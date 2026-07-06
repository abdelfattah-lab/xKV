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

import gc
import torch
import math
from torch import nn
from models.merge_configs import xKVConfig

from ops.shadowkv import batch_gather_gemm_rotary_pos_emb_cuda


def fast_svd(tensor, rank):
    # Input shape: [bsz, seqlen, hidden_dim (nh * hd * gs)]
    original_dtype = tensor.dtype

    # NOTE(brian1009): Have deterministic issue but faster
    U_trunc, S_trunc, V_trunc = torch.svd_lowrank(tensor.float(), q=rank)
    Vt_trunc = V_trunc.transpose(1, 2)

    SVh_trunc = torch.matmul(torch.diag_embed(S_trunc), Vt_trunc)
    
    A = U_trunc.to(original_dtype)  # [bsz, seqlen, rank]
    B = SVh_trunc.to(original_dtype)  # [bsz, rank, hidden_dim (nh * hd * gs)]
    
    return A, B


class KV_Cache:
    """Full Attention"""
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16) -> None:

        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.randn(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache = torch.randn(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0

        # batch prefill record
        self.prefilled_batch = 0
        self.batch_size = batch_size

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int
            ):

        bsz, _, incoming, _ = new_v_cache.shape # [bsz, num_kv_heads, incoming, head_dim]

        if bsz == self.batch_size:
            self.prefilled_batch = 0

        self.k_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_k_cache)
        self.v_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.kv_offset:self.kv_offset + incoming].copy_(new_v_cache)

        key = self.k_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]
        value = self.v_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]

        if incoming > 1: # prefill
            key = key.to(self.device)
            value = value.to(self.device)

        if layer_idx == self.num_layers - 1:
            self.prefilled_batch += bsz
            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming
        
        return key.to(self.device), value.to(self.device)
    
    def print_stats(self):
        print(f"KVCache | max_length {self.max_length} | dtype {self.dtype} | cached {self.kv_offset}")

    def H2D(self):
        # TODO: Uncomment these lines during memory usage evaluation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.k_cache = self.k_cache.to(self.device)
        self.v_cache = self.v_cache.to(self.device)

    def clear(self):
        self.kv_offset = 0
        self.prefilled_batch = 0

    def get_kv_len(self):
        return self.kv_offset


class ShadowKVCache_xKey_CPU:
    """ShadowKV_xKey, can be used for Llama-3-8B, Llama-3.1-8B, GLM-4-9B, Yi-200K"""
    def __init__(self, 
        config: object,
        merge_config: xKVConfig,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size: int = 8,
        ) -> None:
        
        self.config = config
        self.merge_config = merge_config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # ShadowKV setting
        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        self.local_chunk = 4
        self.outlier_chunk = int((self.sparse_budget // 1024) * 24)
        
        # xKV setting
        self.group_size = merge_config.group_size
        self.rank_k = merge_config.rank_k

        self.v_cache_cpu = torch.randn(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length // self.chunk_size,
            self.config.hidden_size // self.config.num_attention_heads * self.chunk_size,
            device='cpu',
            dtype=self.dtype,
            pin_memory=True
        )

        self.k_cache_buffer = torch.randn(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 128 + (self.outlier_chunk+self.local_chunk)*self.chunk_size,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.randn(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 128 + (self.outlier_chunk+self.local_chunk)*self.chunk_size,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.select_sets = self.sparse_budget // self.chunk_size
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        self.temp = torch.randn(
            self.batch_size,
            self.num_key_value_heads,
            self.select_sets,
            self.chunk_size*self.head_dim,
            device=self.device,
            dtype=self.dtype
        ).contiguous()

        # batch prefill record
        self.prefilled_batch = 0

        # v offload kernels
        self.block_num = int(self.batch_size * self.num_key_value_heads)
        self.offsets = torch.zeros(self.block_num*(sparse_budget // chunk_size), device=self.device, dtype=torch.int32).contiguous()
        self.cnts = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.signals = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.position_ids = torch.empty(self.num_layers, self.batch_size, self.num_key_value_heads, self.select_sets, device=self.device, dtype=torch.int64).fill_(-1).contiguous()

        # k compute kernels
        self.output = torch.randn(
            self.batch_size,
            self.num_key_value_heads,
            sparse_budget,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        ).contiguous()

        # multi-stream
        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        print(f"ShadowKV_xKey_CPU | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | group_size {self.group_size} | rank_k {self.rank_k} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        num_landmarks = k_landmark.shape[-2]
        bsz = k_landmark.shape[0]
        if layer_idx == 0 and self.prefilled_batch == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.randn(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device=self.device, dtype=self.dtype)
            self.k_landmark_idx = torch.randn(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device=self.device, dtype=torch.long)

            # for fused gemm kernel
            self.gemm_o = torch.randn(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device=self.device, dtype=torch.bfloat16).contiguous()
            self.softmax_o = torch.randn(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device=self.device, dtype=torch.bfloat16).contiguous()
            self.norm = torch.randn(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device=self.device, dtype=torch.float).contiguous()
            self.sum = torch.randn(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device=self.device, dtype=torch.float).contiguous()
        
        self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark)
        self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark_idx)

    ##### Decoding #####
    def get_retrieval_position_ids(self, layer_idx, query_states):
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1

        # gemm_softmax
        torch.ops._shadowkv.batch_gemm_softmax(
            query_states.contiguous(),
            self.k_landmark[layer_idx].contiguous(),
            self.gemm_o,
            self.norm,
            self.sum,
            self.softmax_o,
            self.batch_size * self.num_key_value_heads,
            self.num_key_value_groups * self.incoming_q_len,
            self.k_landmark[layer_idx].shape[-2],
            self.head_dim,
            1 / math.sqrt(128),
            0
        )
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(self.softmax_o.view(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, -1), dim=-2) # [bsz, 8, chunks]
        else:
            chunk_attn = self.softmax_o.view(self.batch_size, self.num_key_value_heads, -1)

        # [bsz, 8, seq] --> [bsz, 8, select_sets(sparse_budget // chunk_size)]
        merged_results = torch.topk(chunk_attn.view(self.batch_size, self.num_key_value_heads, -1), k=self.select_sets, dim=-1).indices
        # use merged_results to gather the position_ids: [bsz, 8, chunks] --> [bsz, 8, select_sets]
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        torch.ops._shadowkv.reorder_keys_and_compute_offsets(self.position_ids[layer_idx], selected_chunks, self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, self.select_sets)
        self.cnts.fill_(165)  # set hit rate to 65%
        return self.position_ids[layer_idx]

    def get_value_cache(self, layer_idx, position_ids):

        # cpu_v_length is the kernel's PER-HEAD stride into v_cache_cpu, so it must describe
        # the tensor's actual allocated row count (max_length//chunk_size rows), not the
        # prefill-derived max_ctx_chunks_len. The upstream ShadowKV code passes
        # max_ctx_chunks_len*head_dim here, which silently misaligns every head after the
        # first whenever max_length != prefill rounded to chunk_size (e.g. bench uses
        # max_length = seqlen+128 -> each head h reads its data shifted by h*16 chunks).
        cpu_per_head_stride = self.v_cache_cpu.shape[-2] * self.v_cache_cpu.shape[-1]
        torch.ops._shadowkv.gather_copy_with_offsets(self.v_cache_cpu[layer_idx], self.v_cache_buffer[layer_idx], self.temp, self.offsets, self.cnts, self.signals, self.batch_size, self.num_key_value_heads, int(cpu_per_head_stride), int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, cos_sin_cache):

        # gather key cache and rope them
        u = self.U[layer_idx // self.group_size] # [bsz, 128k, rank]
        sv = self.SV[layer_idx] # [bsz, 8, 128, rank]

        # print(f"avg cnts: {self.cnts.float().mean()} hit rate: {self.cnts.float().mean() / (self.sparse_budget / 8.0) * 100:.2f}%")
        torch.ops._shadowkv.gather_copy_d2d_with_offsets(self.k_cache_buffer[layer_idx], self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)
        batch_gather_gemm_rotary_pos_emb_cuda(u, sv, cos_sin_cache, position_ids, self.output, self.chunk_size, self.k_cache_buffer[layer_idx], self.sparse_start, self.sparse_end, self.cnts)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def H2D(self):
        # TODO: Uncomment these lines during memory usage evaluation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.SV = self.SV.to(self.device)
        self.U = self.U.to(self.device)
        self.k_landmark = self.k_landmark.to(self.device)
        self.k_landmark_idx = self.k_landmark_idx.to(self.device)

        self.gemm_o = self.gemm_o.to(self.device)
        self.softmax_o = self.softmax_o.to(self.device)
        self.norm = self.norm.to(self.device)
        self.sum = self.sum.to(self.device)

        self.temp = self.temp.to(self.device)
        self.output = self.output.to(self.device)

        # TODO: Uncomment these lines during memory usage evaluation
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        incoming = new_k_cache.shape[-2]
        self.v_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_v_cache, non_blocking=True)
        self.k_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_k_cache, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.kv_offset = 0
        self.gen_offset = 0
        self.prefill_local = 0

        self.prefilled_batch = 0

    def get_kv_len(self):
        return self.kv_offset


class ShadowKVCache_xKV:
    """ShadowKV_xKV, can be used for Llama-3-8B, Llama-3.1-8B, GLM-4-9B, Yi-200K"""
    def __init__(self, 
        config: object,
        merge_config: xKVConfig,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size: int = 8,
        ) -> None:
        
        self.config = config
        self.merge_config = merge_config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # ShadowKV setting
        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        self.local_chunk = 4
        self.outlier_chunk = int((self.sparse_budget // 1024) * 24)
        
        # xKV setting
        self.group_size = merge_config.group_size
        self.rank_k = merge_config.rank_k
        self.rank_v = merge_config.rank_v

        self.k_cache_buffer = torch.randn(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 128 + (self.outlier_chunk+self.local_chunk)*self.chunk_size,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.randn(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 128 + (self.outlier_chunk+self.local_chunk)*self.chunk_size,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.gen_offset = 0

        # Defer landmark, fused gemm, and decomposed K/V buffers until prefill or explicit pre-init
        self.k_landmark = None
        self.k_landmark_idx = None
        self.gemm_o = None
        self.softmax_o = None
        self.norm = None
        self.sum = None

        self.U_k = None
        self.SV_k = None
        self.U_v = None
        self.SV_v = None

        self.select_sets = self.sparse_budget // self.chunk_size
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        self.temp = torch.randn(
            self.batch_size,
            self.num_key_value_heads,
            self.select_sets,
            self.chunk_size * self.head_dim,
            device=self.device,
            dtype=self.dtype,
        ).contiguous()

        # batch prefill record
        self.prefilled_batch = 0

        # v offload kernels
        self.block_num = int(self.batch_size * self.num_key_value_heads)
        self.offsets = torch.zeros(
            self.block_num * (sparse_budget // chunk_size), device=self.device, dtype=torch.int32
        ).contiguous()
        self.cnts = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous().fill_(150)
        self.signals = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.position_ids = torch.empty(
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            self.select_sets,
            device=self.device,
            dtype=torch.int64,
        ).fill_(-1).contiguous()

        # k, v compute kernels
        self.output = (
            torch.randn(
                self.batch_size,
                self.num_key_value_heads,
                sparse_budget,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            ).contiguous()
        )

        # multi-stream
        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        print(f"ShadowKV_xKV_CPU | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | group_size {self.group_size} | rank_k {self.rank_k} | rank_v {self.rank_v} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    ##### Decoding #####
    def get_retrieval_position_ids(self, layer_idx, query_states):
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1

        # gemm_softmax
        torch.ops._shadowkv.batch_gemm_softmax(
            query_states.contiguous(),
            self.k_landmark[layer_idx].contiguous(),
            self.gemm_o,
            self.norm,
            self.sum,
            self.softmax_o,
            self.batch_size * self.num_key_value_heads,
            self.num_key_value_groups * self.incoming_q_len,
            self.k_landmark[layer_idx].shape[-2],
            self.head_dim,
            1 / math.sqrt(128),
            0
        )
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(self.softmax_o.view(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, -1), dim=-2) # [bsz, 8, chunks]
        else:
            chunk_attn = self.softmax_o.view(self.batch_size, self.num_key_value_heads, -1)

        # [bsz, 8, seq] --> [bsz, 8, select_sets(sparse_budget // chunk_size)]
        merged_results = torch.topk(chunk_attn.view(self.batch_size, self.num_key_value_heads, -1), k=self.select_sets, dim=-1).indices
        # use merged_results to gather the position_ids: [bsz, 8, chunks] --> [bsz, 8, select_sets]
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        torch.ops._shadowkv.reorder_keys_and_compute_offsets(self.position_ids[layer_idx], selected_chunks, self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, self.select_sets)
        self.cnts.fill_(165)  # set hit rate to 65%
        return self.position_ids[layer_idx]

    def get_value_cache(self, layer_idx, position_ids, cos_sin_cache):
        # gather value cache
        u = self.U_v[layer_idx // self.group_size]  # [bsz, 128k, rank]
        sv = self.SV_v[layer_idx]  # [bsz, 8, rank, 128]

        torch.ops._shadowkv.gather_copy_d2d_with_offsets(self.v_cache_buffer[layer_idx], self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)
        result = batch_gather_gemm_rotary_pos_emb_cuda(u, sv, cos_sin_cache, self.position_ids[layer_idx], self.output, self.chunk_size, self.v_cache_buffer[layer_idx], self.sparse_start, self.sparse_end, self.cnts, no_rope=True)
        self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, cos_sin_cache):
        # gather key cache and rope them
        u = self.U_k[layer_idx // self.group_size] # [bsz, 128k, rank]
        sv = self.SV_k[layer_idx] # [bsz, 8, 128, rank]

        torch.ops._shadowkv.gather_copy_d2d_with_offsets(self.k_cache_buffer[layer_idx], self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)
        batch_gather_gemm_rotary_pos_emb_cuda(u, sv, cos_sin_cache, self.position_ids[layer_idx], self.output, self.chunk_size, self.k_cache_buffer[layer_idx], self.sparse_start, self.sparse_end, self.cnts)

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def H2D(self):
        # TODO: Uncomment these lines during memory usage evaluation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.U_k = self.U_k.to(self.device)
        self.U_v = self.U_v.to(self.device)
        self.SV_k = self.SV_k.to(self.device)
        self.SV_v = self.SV_v.to(self.device)
        self.k_landmark = self.k_landmark.to(self.device)
        self.k_landmark_idx = self.k_landmark_idx.to(self.device)

        self.gemm_o = self.gemm_o.to(self.device)
        self.softmax_o = self.softmax_o.to(self.device)
        self.norm = self.norm.to(self.device)
        self.sum = self.sum.to(self.device)

        self.temp = self.temp.to(self.device)
        self.output = self.output.to(self.device)

        # TODO: Uncomment these lines during memory usage evaluation
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):

        incoming = new_k_cache.shape[-2]
        self.v_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_v_cache, non_blocking=True)
        self.k_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_k_cache, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U_k = None
        self.U_v = None
        self.SV_k = None
        self.SV_v = None

        self.kv_offset = 0
        self.gen_offset = 0
        self.prefill_local = 0

        self.prefilled_batch = 0

    def get_kv_len(self):
        return self.kv_offset
