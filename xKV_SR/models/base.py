################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

# Base LLM class

import torch
import os
import torch.nn.functional as F
import time
import gc
from tqdm import tqdm
from contextlib import nullcontext

from flash_attn import flash_attn_with_kvcache
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, record_function, ProfilerActivity

from .tensor_op import sample_token, layer_norm, minference_prefill_kernel
from .kv_cache import KV_Cache, ShadowKVCache
from .kv_cache_xkv import ShadowKVCache_xKey, ShadowKVCache_xKV
from .merge_configs import xKVConfig

class LLM:

    def __str__(self) -> str:
        gpu_mem = f"{round(torch.cuda.memory_allocated(self.device) / 1024**3, 2)} GB / {round(torch.cuda.get_device_properties(self.device).total_memory / 1024**3, 2)} GB"
        return f"LLM: {self.model_name}, attn_mode: {self.attn_mode}, max_length: {self.max_length}, batch_size: {self.batch_size}, device: {self.device}, dtype: {self.dtype}, GPU mem: {gpu_mem}"
    
    def _maybe_record_function(self, name):
        """Return record_function context if profiling is enabled, otherwise nullcontext"""
        return record_function(name) if getattr(self, '_profiling_enabled', False) else nullcontext()

    def init_kv_cache(self, sparse_budget: int, chunk_size: int, config, rank: int, merge_config: xKVConfig):
        if self.attn_mode == 'full':
            self.kv_cache = KV_Cache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        elif self.attn_mode.lower() == 'shadowkv':
            self.kv_cache = ShadowKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, chunk_size=chunk_size, rank=rank)
        elif self.attn_mode.lower() == 'xk_sr':
            self.kv_cache = ShadowKVCache_xKey(config, merge_config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'xkv_sr':
            self.kv_cache = ShadowKVCache_xKV(config, merge_config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, chunk_size=chunk_size)
        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

    def print_kv_stats(self):
        self.kv_cache.print_stats()
    
    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        past_len = self.kv_cache.get_kv_len()
        position_ids = torch.arange(past_len, past_len + input_len, device=self.device, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)
        return position_ids

    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor):

        hidden_states = F.embedding(input_ids, self.embed_tokens)

        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)
        hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)
        
        if hidden_states.shape[1] > 16: # prefill
            hidden_states = hidden_states[:, -1:, :]
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        self.kv_cache.clear()
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))

        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"
        return logits

    @torch.inference_mode()
    def prefill_cont(self, input_ids: torch.LongTensor):
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))
        return logits
    
    def encode(self, text: str, template=None, truncation=False):
        if template == 'chat':
            text = self.chat_template.format(msg=text)
            input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if self.tokenizer.bos_token_id is not None:
                assert self.tokenizer.bos_token_id not in input_ids, f"bos_token_id found in input_ids"
            return input_ids
        if template == 'ctx':
            text = self.ctx_template.format(ctx=text)
        if template == 'prefix':
            text = self.prefix_template.format(ctx=text)
        input_ids = self.tokenizer(text, return_tensors="pt", truncation=truncation).input_ids.to(self.device)
        return input_ids

    @torch.inference_mode()
    def layer_compute(self, 
            buffer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim
        )
        
        if isinstance(self.kv_cache, KV_Cache):
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
            key_states, value_states = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
            
            if self.minference == True and q_len > 1:
                hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
            else:
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        elif isinstance(self.kv_cache, (ShadowKVCache, ShadowKVCache_xKey, ShadowKVCache_xKV)):

            if q_len > 1024: # prefill
                with self._maybe_record_function("batch_prefill"):
                    # svd unrope key and save
                    if isinstance(self.kv_cache, ShadowKVCache_xKey):
                        self.kv_cache.get_svd(key_states, layer_idx, fake_svd=self.fake_svd)
                    elif isinstance(self.kv_cache, ShadowKVCache_xKV):
                        self.kv_cache.get_svd(key_states, value_states, layer_idx, fake_svd=self.fake_svd)
                    else:
                        self.kv_cache.get_svd(key_states, layer_idx)
                    query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                    self.kv_cache.prefill_kv_cache(value_states, layer_idx, key_states, query_states[:, :, -1:])
                    
                    if self.minference == True:
                        hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
                    else:
                        hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)
            else: # decode

                # rope query and key
                with self._maybe_record_function("rope_query_key"):
                    query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

                # update kv cache to buffer
                with self._maybe_record_function("update_kv_cache"):
                    self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)

                # get retrieval idx
                with self._maybe_record_function("get_retrieval_position_ids"):
                    position_ids = self.kv_cache.get_retrieval_position_ids(layer_idx=layer_idx, query_states=query_states)

                # multi-stream
                curr_stream = torch.cuda.current_stream()
                get_value_stream = self.kv_cache.copy_stream

                with self._maybe_record_function("get_value_cache_offload_stream"):
                    with torch.cuda.stream(get_value_stream):
                        get_value_stream.wait_stream(curr_stream)
                        value_states = self.kv_cache.get_value_cache(layer_idx, position_ids)

                # gather key cache from GPU and RoPE it (should be hidden by CPU offloading time)
                with self._maybe_record_function("get_key_cache"):
                    key_states = self.kv_cache.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, rope_func=self.apply_rotary_pos_emb_single)

                # flash attention
                with self._maybe_record_function("flash_attn_with_kvcache"):
                    hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

        hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
        
        if bsz*q_len > 64*1024: # [bsz, seq, 128]
            output = torch.empty_like(hidden_states)
            prop_iter = bsz * q_len // (8*1024)
            prefill_chunk_size = bsz * q_len // prop_iter
            prefill_iter = (q_len + prefill_chunk_size - 1) // prefill_chunk_size
            for i in range(prefill_iter):
                start = i*prefill_chunk_size
                end = (i+1)*prefill_chunk_size
                output[:, start:end] = self.post_attention_compute(hidden_states[:, start:end], residual[:, start:end], buffer)
            
            hidden_states = output

        else:
            hidden_states = self.post_attention_compute(hidden_states, residual, buffer)
        
        return hidden_states

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool = False):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = 0.9, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False, enable_profiler: bool = False, profiler_output_dir: str = "./profiler_logs", profiler_wait_steps: int = 2, profiler_warmup_steps: int = 2, profiler_active_steps: int = 6):
        """accuracy eval usage, not for throughput eval
        
        Args:
            enable_profiler: Enable torch profiler to trace the inference workload
            profiler_output_dir: Directory to save profiler traces (default: ./profiler_logs)
            profiler_wait_steps: Number of steps to wait before profiling (default: 2)
            profiler_warmup_steps: Number of warmup steps before active profiling (default: 2)
            profiler_active_steps: Number of steps to actively profile (default: 6)
        """
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill(input_ids)
        else:
            if input_ids.size(1) + self.kv_cache.get_kv_len() >= self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill_cont(input_ids)
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        pos = 0
        generated_ids = []
        generated_ids.extend(next_token[0].tolist())
        
        self.kv_cache.H2D()

        if benchmark == True:
            start = time.time()
        
        # Profiler configuration - start after prefill (skip prefill to reduce file size)
        if enable_profiler:
            self._profiling_enabled = True
            os.makedirs(profiler_output_dir, exist_ok=True)
            
            profiler_schedule = torch.profiler.schedule(
                wait=profiler_wait_steps,
                warmup=profiler_warmup_steps,
                active=profiler_active_steps,
                repeat=1
            )
            
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=profiler_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
                record_shapes=True,
                profile_memory=False,  # Disable memory profiling
                with_stack=True
            )
            prof.start()
            print(f"\nProfiler enabled: will profile {profiler_active_steps} decode steps after {profiler_wait_steps + profiler_warmup_steps} warmup (skipping prefill)")

        
        while n < gen_len:
            with self._maybe_record_function("decode_step"):
                logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
                next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.extend(next_token[0].tolist())
            
            # Step profiler
            if enable_profiler:
                prof.step()
                # Stop profiler after wait + warmup + active steps
                if n == profiler_wait_steps + profiler_warmup_steps + profiler_active_steps:
                    self._profiling_enabled = False
                    prof.stop()
                    print(f"\nProfiler stopped after {profiler_active_steps} active decode steps")
                    print(f"Traces saved to: {profiler_output_dir}")
                    print(f"View with: tensorboard --logdir={profiler_output_dir}")
                    enable_profiler = False
            
            if verbose == True:
                generated_text = (
                    self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    ).strip().split(" ")
                )
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            if next_token[0] == self.tokenizer.eos_token_id:
                break
            if self.tokenizer.decode(next_token[0]) == "<|eot_id|>": # llama-3
                break
            if self.tokenizer.decode(next_token[0]) == "<|im_end|>": # yi
                break
            if next_token[0] in [151329, 151336, 151338]: # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|endoftext|>": # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|end|>": # phi
                break

        if verbose == True and n!=0:
            print(" ".join(generated_text[pos:]), end=" ", flush=True)
        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s, {round(n / (end - start), 2)} tokens/s | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        # TODO(max410011): Uncomment these lines during memory usage evaluation
        # gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

        return [self.tokenizer.decode(generated_ids, skip_special_tokens=True)]
    
    @torch.inference_mode()
    def batch_prefill(self, input_ids: torch.Tensor, benchmark: bool = False):
        self.kv_cache.clear()
        batch_size = input_ids.size(0)
        
        assert batch_size == self.batch_size, f"batch_size mismatch, got {batch_size}, expected {self.batch_size}"
        
        if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
        
        logits = torch.zeros(batch_size, 1, self.vocab_size, device=self.device, dtype=torch.float32)

        if input_ids.shape[-1] > 120*1024 and input_ids.shape[-1] < 200*1024:
            T = 8
        else:
            T = 4
        # # NOTE(max410011): Modify mini-batch to 1 to reduce peak memory usage during prefill
        # T = 1
        # for bsz in range(0, batch_size, T):
        for bsz in tqdm(range(0, batch_size, T), desc=f"Prefilling (batch size={batch_size})"):
            req_input_ids = input_ids[bsz:bsz+T]
            logits[bsz:bsz+T].copy_(self.inference(input_ids=req_input_ids, position_ids=self.get_ctx(req_input_ids)))
        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"

        return logits


    @torch.inference_mode()
    def warmup(self):

        a = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        b = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        for _ in range(100):
            torch.bmm(a, b)
        del a, b

        # print("Warmup done")

    @torch.inference_mode()
    def batch_generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = -1, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False, enable_profiler: bool = False, profiler_output_dir: str = "./profiler_logs", profiler_wait_steps: int = 2, profiler_warmup_steps: int = 2, profiler_active_steps: int = 6):
        """throughput eval usage
        
        Args:
            enable_profiler: Enable torch profiler to trace the inference workload
            profiler_output_dir: Directory to save profiler traces (default: ./profiler_logs)
            profiler_wait_steps: Number of steps to wait before profiling (default: 2)
            profiler_warmup_steps: Number of warmup steps before active profiling (default: 2)
            profiler_active_steps: Number of steps to actively profile (default: 6)
        """
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.batch_prefill(input_ids)
        else:
            logits = self.prefill_cont(input_ids)
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        generated_ids = []
        generated_ids.append(next_token[:, -1].tolist())
        
        self.kv_cache.H2D()
        self.warmup()

        if benchmark == True:
            start = time.time()
        
        # Profiler configuration - start after prefill (skip prefill to reduce file size)
        if enable_profiler:
            self._profiling_enabled = True
            os.makedirs(profiler_output_dir, exist_ok=True)
            
            profiler_schedule = torch.profiler.schedule(
                wait=profiler_wait_steps,
                warmup=profiler_warmup_steps,
                active=profiler_active_steps,
                repeat=1
            )
            
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=profiler_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
                record_shapes=True,
                profile_memory=False,  # Disable memory profiling
                with_stack=True
            )
            prof.start()
            print(f"\nProfiler enabled: will profile {profiler_active_steps} decode steps after {profiler_wait_steps + profiler_warmup_steps} warmup (skipping prefill)")

        
        while n < gen_len:
            with self._maybe_record_function("decode_step"):
                logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
                next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.append(next_token[:, -1].tolist())
            
            # Step profiler
            if enable_profiler:
                prof.step()
                # Stop profiler after wait + warmup + active steps
                if n == profiler_wait_steps + profiler_warmup_steps + profiler_active_steps:
                    self._profiling_enabled = False
                    prof.stop()
                    print(f"\nProfiler stopped after {profiler_active_steps} active decode steps")
                    print(f"Traces saved to: {profiler_output_dir}")
                    print(f"View with: tensorboard --logdir={profiler_output_dir}")
                    enable_profiler = False

        if benchmark == True:
            end = time.time()
            # print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s | Throughput: {round(self.batch_size * n / (end - start), 2)} tokens/s, Latency: {round((end - start)*1000 / n, 2)} ms/step | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        # TODO(max410011): Uncomment these lines during memory usage evaluation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        generated_ids = torch.LongTensor(generated_ids).t().tolist()

        if benchmark == True:
            return self.decode(generated_ids, skip_special_tokens=True), self.batch_size * n / (end - start)

        return self.decode(generated_ids, skip_special_tokens=True)