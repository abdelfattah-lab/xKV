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


import torch
import torch.nn.functional as F
import gc
import os

import transformers
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
transformers.logging.set_verbosity_error()

import vllm
from minference.configs.model2path import MODEL2PATH

from .tensor_op import layer_norm, apply_rotary_pos_emb, apply_rotary_pos_emb_single, apply_rotary_pos_emb_cuda
from .prompt_template import Templates, Chat_Templates, Prefix_Templates
from .base import LLM
from .merge_configs import generate_consecutive_palu_config

class LlamaLayer:
    def __init__(self, layer_idx) -> None:
        
        self.wqkv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.gate_up_proj :torch.Tensor = None 
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx

    def init_parameters(self, hf_layer: LlamaDecoderLayer):

        self.wqkv :torch.Tensor= torch.cat((hf_layer.self_attn.q_proj.weight.detach(), hf_layer.self_attn.k_proj.weight.detach(), hf_layer.self_attn.v_proj.weight.detach()), dim=0)
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()
        self.q_size = hf_layer.self_attn.q_proj.weight.shape[0]
        self.kv_size = hf_layer.self_attn.k_proj.weight.shape[0]

        self.gate_up_proj = torch.cat((hf_layer.mlp.gate_proj.weight.detach(), hf_layer.mlp.up_proj.weight.detach()), dim=0)
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon
    
    def init_gpu(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wqkv = self.wqkv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_up_proj = self.gate_up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)

class Llama(LLM):
    def __init__(self, 
        model_name: str = "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        batch_size :int = 1,
        max_length :int = 64*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        attn_mode: str = 'full',
        # minference
        minference=False,
        # ShadowKV
        sparse_budget: int = 2048,
        rank: int = 160,
        chunk_size: int = 8,
        # xKV
        start_layer_idx: int = 0,
        end_layer_idx: int = -1,
        group_size: int = 4,
        rank_k: int = 384,
        rank_v: int = 576,
        fake_svd: bool = False,
        profile_layers: int = 0,
    ) -> None:
        # assert batch_size == 1, "Batch size must be 1"
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = LlamaConfig.from_pretrained(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.vocab_size = self.config.vocab_size

        # profile-only-N-layer
        self.profile_layer = profile_layers
        if self.profile_layer:
            print(f"Profile {self.profile_layer} layers.")

        self.init_parameters()
        self.attn_mode = attn_mode
        self.minference = minference

        self.merge_config = generate_consecutive_palu_config(
            start_layer=start_layer_idx,
            end_layer=end_layer_idx if end_layer_idx != -1 else self.num_layers,
            group_size=group_size,
            rank_k=rank_k,
            rank_v=rank_v
        )

        if 'llama-3' in model_name.lower():
            self.ctx_template = Templates['llama-3']
            self.chat_template = Chat_Templates['llama-3']
            self.prefix_template = Prefix_Templates['llama-3']
        elif 'yi' in model_name.lower():
            self.ctx_template = Templates['yi']
            self.chat_template = Chat_Templates['yi']
            self.prefix_template = Prefix_Templates['yi']
        else:
            raise ValueError(f"Invalid model name {model_name}")

        self.fake_svd = fake_svd
        self.init_kv_cache(sparse_budget, chunk_size, self.config, rank=rank, merge_config=self.merge_config)

        if self.minference:
            import json
            self.minference_parttern = []
            for layer_idx in range(self.num_layers):
                self.minference_parttern.append({int(ii): jj for ii, jj in json.load(open(MODEL2PATH[self.model_name]))[layer_idx].items()})


    def _set_cos_sin_cache(self, inv_freq: torch.Tensor):
        t = torch.arange(self.max_length + 1024, device=self.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb_cuda(x, self.cos_sin_cache, position_ids)

    @torch.inference_mode()
    def apply_rotary_pos_emb_single_torch(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb_single(x, self.cos_cache, self.sin_cache, position_ids)

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        vllm._custom_ops.rotary_embedding(position_ids, q, k, 128, self.cos_sin_cache, True)
        bsz = q.shape[0]
        q = q.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return q, k

    def init_parameters(self):
        hf_model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        try:
            cos_cache = hf_model.model.rotary_emb.cos_cached[:self.max_length+1024].to(self.device).to(self.dtype)
            sin_cache = hf_model.model.rotary_emb.sin_cached[:self.max_length+1024].to(self.device).to(self.dtype)
        except:
            cos_cache, sin_cache = self._set_cos_sin_cache(hf_model.model.rotary_emb.inv_freq.to(self.device))
        self.cos_sin_cache = torch.cat((cos_cache[:, :64], sin_cache[:, :64]), dim=-1)
        self.cos_cache, self.sin_cache = cos_cache, sin_cache
        del cos_cache, sin_cache

        self.layers :list[LlamaLayer] = []

        # If profiling one layer, only initialize the first layer to keep KV cache semantics simple
        max_layers = self.profile_layer if self.profile_layer else len(hf_model.model.layers)
        for idx, hf_layer in enumerate(hf_model.model.layers[:max_layers]):
            layer = LlamaLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()

        # Keep Llama.num_layers in sync with actual layers and update config for KV caches
        self.num_layers = len(self.layers)
        # Ensure KV caches allocate per our actual layer count
        try:
            self.config.num_hidden_layers = self.num_layers
        except Exception:
            pass

    @torch.inference_mode()
    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: LlamaLayer,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        qkv = F.linear(hidden_states, buffer.wqkv)
        query_states, key_states, value_states = qkv.split([buffer.q_size, buffer.kv_size, buffer.kv_size], dim=-1)
        # TODO(max410011): Uncomment these lines during memory usage evaluation
        # del qkv
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        # gc.collect()

        return query_states, key_states, value_states.view(value_states.shape[0], -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    @torch.inference_mode()
    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: LlamaLayer
    ):  
        hidden_states = F.linear(attn_output, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        
        hidden_states = F.linear(hidden_states, buffer.gate_up_proj)
        d = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        torch.ops._C.silu_and_mul(out, hidden_states)
        
        hidden_states = F.linear(out, buffer.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states


