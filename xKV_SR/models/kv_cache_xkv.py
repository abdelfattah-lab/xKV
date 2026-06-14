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
from .merge_configs import xKVConfig

from .tensor_op import batch_gather_gemm_rotary_pos_emb_cuda
from xKV_SR.kernels import shadowkv


def fake_svd(tensor, rank):
    """Perform fake SVD: SVD -> Truncate -> Multiply back."""
    # Input shape: [bs, sl, hidden_dim (nh * hd * gs)]
    original_dtype = tensor.dtype
    
    # Step 1: Perform SVD NOTE(brian1009): Have deterministic issue but faster
    # U_trunc, S_trunc, V_trunc = torch.svd_lowrank(tensor, q=rank)
    # Vt_trunc = V_trunc.transpose(1, 2)
    
    U, S, V_h = torch.linalg.svd(tensor.float(), full_matrices=False)
    U_trunc = U[:, :, :rank]
    S_trunc = S[:, :rank]
    Vh_trunc = V_h[:, :rank, :]
    
    # Step 2: Multiply back to approximate the original tensor
    approx_tensor = torch.matmul(U_trunc, torch.matmul(torch.diag_embed(S_trunc), Vh_trunc)) # (bs, sl, hidden_dim)
    approx_tensor = approx_tensor.to(original_dtype)

    return approx_tensor

def svd(tensor, rank):
    # Input shape: [bsz, seqlen, hidden_dim (nh * hd * gs)]
    original_dtype = tensor.dtype

    U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
    U_trunc = U[:, :, :rank]
    S_trunc = S[:, :rank]
    Vh_trunc = Vh[:, :rank, :]
    SVh_trunc = torch.matmul(torch.diag_embed(S_trunc), Vh_trunc)
    
    A = U_trunc.to(original_dtype)  # [bsz, seqlen, rank]
    B = SVh_trunc.to(original_dtype)  # [bsz, rank, hidden_dim (nh * hd * gs)]
    
    return A, B

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

class ShadowKVCache_xKey:
    """ShadowKV, only for accuracy measurement and understanding, not for efficiency, please refer to ShadowKV_CPU for the efficient implementation"""
    def __init__(self, 
        config: object,
        merge_config: xKVConfig,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=8,
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
        
        assert self.batch_size == 1, "ShadowKV class only supports batch_size=1, please use ShadowKV_CPU class for batch_size > 1"

        self.key_cache = []
        self.key_temp_buffer = []

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )


        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = []
        self.SV = []

        self.fake_svd = False
        self.errors = []
        self.relative_errors = []

        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        print(f"ShadowKV_xKey | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | group_size {self.group_size} | rank_k {self.rank_k} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    def get_svd(self, key, layer_idx, fake_svd=False):
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        self.fake_svd = fake_svd
        if key.shape[1] == self.num_key_value_heads:
            # [bsz, 8, prefill, 128] -> [bsz, prefill, 1024]
            key = key.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads * self.head_dim)

        # Update the key cache
        if fake_svd:
            self.key_cache.append(key.clone())
        else:
            self.key_temp_buffer.append(key.clone())

        # Apply cross-layer SVD if we have updated the last layer in the group
        if layer_idx % self.group_size == (self.group_size - 1):
            self.grouped_layer_merging(layer_idx)
    
    @torch.no_grad()
    def grouped_layer_merging(self, last_layer_idx):
        """Perform real/fake SVD on grouped layers, inferring dimensions from the tensors."""
        start_layer_idx, end_layer_idx = (last_layer_idx - self.group_size + 1), last_layer_idx

        # Step 1: Collect keys for the layers in the group
        if self.fake_svd:
            keys = [self.key_cache[i] for i in range(start_layer_idx, end_layer_idx + 1)]
        else:
            keys = self.key_temp_buffer
        
        # Step 2: Concatenate along the nh * hd dimension
        combined_key = torch.cat(keys, dim=2)  # [bsz, prefill, nh * hd * gs]

        if self.fake_svd:
            # Step 3: Apply fake SVD (truncate and multiply back)
            combined_key_approx = fake_svd(combined_key, rank=self.rank_k)

            # Step 4: Convert back to [bsz, heads, seq_len, head_dim] shape and split
            combined_key_approx = combined_key_approx.view(self.batch_size, -1, self.num_key_value_heads * self.group_size, self.head_dim).transpose(1, 2)
            key_layers = torch.split(combined_key_approx, self.num_key_value_heads, dim=1)
            
            for idx, layer_idx in enumerate(range(start_layer_idx, end_layer_idx + 1)):
                self.key_cache[layer_idx] = key_layers[idx]
        else:
            # Step 3: Apply real SVD
            #U_trunc, SVh_trunc = svd(combined_key, self.rank_k)
            U_trunc, SVh_trunc = fast_svd(combined_key, self.rank_k)
            #U_trunc, SVh_trunc = svd_randomized_batched_bf16(combined_key, self.rank_k)

            # Step 4: Reshape, split and store U, SV for each layer
            self.U.append(U_trunc)  # [bsz, seqlen, rank]
            
            # [bsz, rank, nh * hd * gs] -> [bsz, nh * gs, rank, hd]
            SVh_trunc = SVh_trunc.view(self.batch_size, self.rank_k, self.num_key_value_heads * self.group_size, self.head_dim).transpose(1, 2)
            SVhs = torch.split(SVh_trunc, self.num_key_value_heads, dim=1)  # [bsz, nh, rank, hd]
            self.SV.extend(SVhs)
            
            self.key_temp_buffer = []

    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        num_landmarks = k_landmark.shape[-2]
        if layer_idx == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device=self.device, dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device=self.device, dtype=torch.long)
        
        self.k_landmark[layer_idx].copy_(k_landmark.contiguous())
        self.k_landmark_idx[layer_idx].copy_(k_landmark_idx.contiguous())

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            query: torch.Tensor=None
            ):
        
        incoming = new_v_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        self.select_sets = self.sparse_budget // self.chunk_size
        
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
    
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(self.batch_size, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(self.batch_size, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def get_retrieval_position_ids(self, layer_idx, query_states):
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1
        # print(query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim).shape, self.k_landmark[layer_idx].transpose(2, 3).shape)
        # [bsz, 8, 4, q_len, 128] * [bsz, 8, 128, chunks] --> [bsz, 8, 4, q_len, chunks]
        chunk_attn = torch.einsum('bhgqd,bhdc->bhgqc', query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim), self.k_landmark[layer_idx].transpose(2, 3)).squeeze(2) / math.sqrt(128)
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype) # [bsz, 8, 4, q_len, chunks]
        chunk_attn = chunk_attn.sum(dim = -2) # [bsz, 8, 4, chunks]
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(sparse_budget // chunk_size)]

        # use merged_results to gather the position_ids: [bsz, 8, chunks] --> [bsz, 8, select_sets]
        # print(f"k_landmark_idx {self.k_landmark_idx.shape} merged_results {merged_results.shape}")
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        # print(f"selected_chunks {selected_chunks.shape}")
        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(self.batch_size, self.num_key_value_heads, -1) # [bsz, 8, select_sets * chunk_size]

        return position_ids
        
    def get_value_cache(self, layer_idx, position_ids):
        # gather value cache
        value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func):
        if self.fake_svd:
            k_cache = self.key_cache[layer_idx].clone()
            key_gathered = k_cache.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            key_gathered = rope_func(key_gathered, position_ids)
            self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(key_gathered, non_blocking=True)
            gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len
            
            return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]
            
        # gather key cache and rope them
        u = self.U[layer_idx // self.group_size]  # [bsz, 128k, rank]
        sv = self.SV[layer_idx]  # [bsz, 8, rank, 128]

        # indexing, [bsz, 8, sparse_budget, rank]
        index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1))  # [bsz, 8, sparse_budget, rank]
        u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1)  # [bsz, 8, 128k, rank]
        U_head = torch.gather(u_expand, 2, index_expanded)

        # [bsz, 8, sparse_budget, rank] -matmul- [8, rank, 128] --> [bsz, 8, sparse_budget, 128]
        result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)

        # rope the key cache
        result = rope_func(result, position_ids)

        # send to buffer
        self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

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
        self.key_cache = []
        self.key_temp_buffer = []
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = []
        self.SV = []

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0
    
    def H2D(self):
        pass

    def get_kv_len(self):
        return self.kv_offset


class ShadowKVCache_xKV:
    """ShadowKV, only for accuracy measurement and understanding, not for efficiency, please refer to ShadowKV_CPU for the efficient implementation"""
    def __init__(self, 
        config :object,
        merge_config: xKVConfig,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=8,
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

        assert self.batch_size == 1, "ShadowKV class only supports batch_size=1, please use ShadowKV_CPU class for batch_size > 1"

        self.key_cache = []
        self.value_cache = []
        self.key_temp_buffer = []
        self.value_temp_buffer = []

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )


        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U_k = []
        self.SV_k = []
        self.U_v = []
        self.SV_v = []

        self.fake_svd = False
        self.errors = []
        self.relative_errors = []

        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        print(f"ShadowKV_xKV | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | group_size {self.group_size} | rank_k {self.rank_k} | rank_v {self.rank_v} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    def get_svd(self, key, value, layer_idx, fake_svd=False):
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        self.fake_svd = fake_svd
        if key.shape[1] == self.num_key_value_heads:
            # [bsz, 8, prefill, 128] -> [bsz, prefill, 1024]
            key = key.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads * self.head_dim)
        if value.shape[1] == self.num_key_value_heads:
            # TODO(max410011): Value is reshaped but Key is not; check efficiency.
            # [bsz, 8, prefill, 128] -> [bsz, prefill, 1024]
            value = value.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads * self.head_dim)

        # Update the key and value cache
        if fake_svd:
            self.key_cache.append(key.clone())
            self.value_cache.append(value.clone())
        else:
            self.key_temp_buffer.append(key.clone())
            self.value_temp_buffer.append(value.clone())

        # Apply cross-layer SVD if we have updated the last layer in the group
        if layer_idx % self.group_size == (self.group_size - 1):
            self.grouped_layer_merging(layer_idx)
    
    @torch.no_grad()
    def grouped_layer_merging(self, last_layer_idx):
        """Perform real/fake SVD on grouped layers, inferring dimensions from the tensors."""
        start_layer_idx, end_layer_idx = (last_layer_idx - self.group_size + 1), last_layer_idx 

        # Step 1: Collect keys and values for the layers in the group
        if self.fake_svd:
            keys = [self.key_cache[i] for i in range(start_layer_idx, end_layer_idx + 1)]
            values = [self.value_cache[i] for i in range(start_layer_idx, end_layer_idx + 1)]
        else:
            keys = self.key_temp_buffer
            values = self.value_temp_buffer

        # Step 2: Concatenate along the nh * hd dimension
        combined_key = torch.cat(keys, dim=2)  # [bsz, prefill, nh * hd * gs]
        combined_value = torch.cat(values, dim=2)

        if self.fake_svd:
            # Step 3: Apply fake SVD (truncate and multiply back)
            combined_key_approx = fake_svd(combined_key, rank=self.rank_k)
            combined_value_approx = fake_svd(combined_value, rank=self.rank_v)

            # Step 4: Convert back to [bsz, heads, seq_len, head_dim] shape and split
            combined_key_approx = combined_key_approx.view(self.batch_size, -1, self.num_key_value_heads * self.group_size, self.head_dim).transpose(1, 2)
            combined_value_approx = combined_value_approx.view(self.batch_size, -1, self.num_key_value_heads * self.group_size, self.head_dim).transpose(1, 2)
            key_layers = torch.split(combined_key_approx, self.num_key_value_heads, dim=1)
            value_layers = torch.split(combined_value_approx, self.num_key_value_heads, dim=1)

            for idx, layer_idx in enumerate(range(start_layer_idx, end_layer_idx + 1)):
                self.key_cache[layer_idx] = key_layers[idx]
                self.value_cache[layer_idx] = value_layers[idx]
        else:
            # Step 3: Apply real SVD
            U_trunc_k, SVh_trunc_k = svd(combined_key, self.rank_k)
            U_trunc_v, SVh_trunc_v = svd(combined_value, self.rank_v)
            
            # Step 4: Reshape, split and store U, SV for each layer
            self.U_k.append(U_trunc_k)  # [bsz, seqlen, rank]
            self.U_v.append(U_trunc_v)
            
            # [bsz, rank, nh * hd * gs] -> [bsz, nh * gs, rank, hd]
            SVh_trunc_k = SVh_trunc_k.view(self.batch_size, self.rank_k, self.num_key_value_heads * self.group_size, self.head_dim).transpose(1, 2)
            SVh_trunc_v = SVh_trunc_v.view(self.batch_size, self.rank_v, self.num_key_value_heads * self.group_size, self.head_dim).transpose(1, 2)
            SVhs_k = torch.split(SVh_trunc_k, self.num_key_value_heads, dim=1)  # [bsz, num_heads, rank, head_dim]
            SVhs_v = torch.split(SVh_trunc_v, self.num_key_value_heads, dim=1)
            self.SV_k.extend(SVhs_k)
            self.SV_v.extend(SVhs_v)
            
            self.key_temp_buffer = []
            self.value_temp_buffer = []

    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        num_landmarks = k_landmark.shape[-2]
        if layer_idx == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device=self.device, dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device=self.device, dtype=torch.long)
        
        self.k_landmark[layer_idx].copy_(k_landmark.contiguous())
        self.k_landmark_idx[layer_idx].copy_(k_landmark_idx.contiguous())

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            query: torch.Tensor=None
            ):
        
        incoming = new_v_cache.shape[-2] # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        self.select_sets = self.sparse_budget // self.chunk_size
        
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
    
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(self.batch_size, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(self.batch_size, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def get_retrieval_position_ids(self, layer_idx, query_states):
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1
        # print(query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim).shape, self.k_landmark[layer_idx].transpose(2, 3).shape)
        # [bsz, 8, 4, q_len, 128] * [bsz, 8, 128, chunks] --> [bsz, 8, 4, q_len, chunks]
        chunk_attn = torch.einsum('bhgqd,bhdc->bhgqc', query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim), self.k_landmark[layer_idx].transpose(2, 3)).squeeze(2) / math.sqrt(128)
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype) # [bsz, 8, 4, q_len, chunks]
        chunk_attn = chunk_attn.sum(dim = -2) # [bsz, 8, 4, chunks]
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(sparse_budget // chunk_size)]

        # use merged_results to gather the position_ids: [bsz, 8, chunks] --> [bsz, 8, select_sets]
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]

        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(self.batch_size, self.num_key_value_heads, -1) # [bsz, 8, select_sets * chunk_size]

        return position_ids
        
    def get_value_cache(self, layer_idx, position_ids):
        if self.fake_svd:
            value_gathered = self.value_cache[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(value_gathered, non_blocking=True)
            gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len
            
            return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]
            
        # Use SVD reconstruction for value cache
        u = self.U_v[layer_idx // self.group_size]  # [bsz, 128k, rank]
        sv = self.SV_v[layer_idx]  # [bsz, 8, rank, 128]
        
        # indexing, [bsz, 8, sparse_budget, rank]
        index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1))  # [bsz, 8, sparse_budget, rank]
        u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1)  # [bsz, 8, 128k, rank]
        U_head = torch.gather(u_expand, 2, index_expanded)

        # [bsz, 8, sparse_budget, rank] -matmul- [8, rank, 128] --> [bsz, 8, sparse_budget, 128]
        result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)

        # send to buffer
        self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func):
        if self.fake_svd:
            k_cache = self.key_cache[layer_idx].clone()
            key_gathered = k_cache.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            key_gathered = rope_func(key_gathered, position_ids)
            self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(key_gathered, non_blocking=True)
            gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len
            
            return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]
            
        # gather key cache and rope them
        u = self.U_k[layer_idx // self.group_size]  # [bsz, 128k, rank]
        sv = self.SV_k[layer_idx]  # [bsz, 8, rank, 128]

        # indexing, [bsz, 8, sparse_budget, rank]
        index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1))  # [bsz, 8, sparse_budget, rank]
        u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1)  # [bsz, 8, 128k, rank]
        U_head = torch.gather(u_expand, 2, index_expanded)

        # [bsz, 8, sparse_budget, rank] -matmul- [8, rank, 128] --> [bsz, 8, sparse_budget, 128]
        result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)

        # rope the key cache
        result = rope_func(result, position_ids)

        # send to buffer
        self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

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
        self.key_cache = []
        self.value_cache = []
        self.key_temp_buffer = []
        self.value_temp_buffer = []
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U_k = []
        self.SV_k = []
        self.U_v = []
        self.SV_v = []

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0
    
    def H2D(self):
        pass

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

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length // self.chunk_size,
            self.config.hidden_size // self.config.num_attention_heads * self.chunk_size,
            device='cpu',
            dtype=self.dtype,
            pin_memory=True
        )

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 128 + (self.outlier_chunk+self.local_chunk)*self.chunk_size,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
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
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.key_temp_buffer = None
        self.U = None
        self.SV = None

        self.select_sets = self.sparse_budget // self.chunk_size
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        self.temp = torch.zeros(
            self.batch_size, 
            self.num_key_value_heads, 
            self.select_sets, 
            self.chunk_size*self.head_dim, 
            device='cpu', 
            dtype=self.dtype
        ).contiguous()

        # batch prefill record
        self.prefilled_batch = 0

        # v offload kernels
        self.block_num = int(self.batch_size * self.num_key_value_heads)
        self.offsets = torch.zeros(self.block_num*(sparse_budget // chunk_size), device=self.device, dtype=torch.int32).contiguous()
        self.cnts = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.signals = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.position_ids = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.select_sets, device=self.device, dtype=torch.int64).fill_(-1).contiguous()

        # k compute kernels
        self.output = torch.zeros(
            self.batch_size, 
            self.num_key_value_heads, 
            sparse_budget, 
            self.head_dim, 
            device='cpu', 
            dtype=self.dtype
        ).contiguous()

        # multi-stream
        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        print(f"ShadowKV_xKey_CPU | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | group_size {self.group_size} | rank_k {self.rank_k} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    ##### Encoding #####
    def get_svd(self, new_k_cache, layer_idx):
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        if new_k_cache.shape[1] == self.num_key_value_heads:
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads * self.head_dim)
        else:
            k_cache = new_k_cache  # [bsz, prefill, 1024]

        # NOTE(max410011): mini-batch for prefill (4 or 8)
        bsz = k_cache.shape[0]

        if layer_idx == 0 and self.prefilled_batch == 0:
            # init U, SV and temp buffer on GPU for faster SVD computation
            self.U = torch.zeros(self.num_layers // self.group_size, self.batch_size, k_cache.shape[1], self.rank_k, device=self.device, dtype=self.dtype)
            self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.head_dim, self.rank_k, device=self.device, dtype=self.dtype)
            self.key_temp_buffer = torch.zeros(bsz, k_cache.shape[1], self.num_key_value_heads * self.head_dim * self.group_size, device=self.device, dtype=self.dtype)

        # Store temp key cache for cross-layer SVD into the correct slice for this layer within the group
        start = (layer_idx % self.group_size) * self.num_key_value_heads * self.head_dim
        end = start + self.num_key_value_heads * self.head_dim
        self.key_temp_buffer[:, :, start:end].copy_(k_cache)  # [bsz, prefill, 1024 * gs]

        # Apply cross-layer SVD if we have updated the last layer in the group
        if layer_idx % self.group_size == (self.group_size - 1):
            self.grouped_layer_merging(layer_idx)

    @torch.no_grad()
    def grouped_layer_merging(self, last_layer_idx):
        """Perform SVD on grouped layers, inferring dimensions from the tensors."""
        start_layer_idx, end_layer_idx = (last_layer_idx - self.group_size + 1), last_layer_idx

        # Step 1: Apply real SVD directly on the concatenated feature dimension
        U_trunc, SVh_trunc = fast_svd(self.key_temp_buffer, self.rank_k)

        # Step 2: Reshape, split and store U, SV for each layer
        bsz = U_trunc.shape[0]
        self.U[last_layer_idx // self.group_size][self.prefilled_batch:self.prefilled_batch + bsz].copy_(U_trunc)  # [bsz, seqlen, rank]

        # [bsz, rank, nh * hd * gs] -> [bsz, nh * gs, rank, hd]
        SVh_trunc = SVh_trunc.view(bsz, self.rank_k, self.num_key_value_heads * self.group_size, self.head_dim).transpose(1, 2)
        SVh_trunc = SVh_trunc.transpose(-1, -2)  # use for kernel [bsz, nh * gs, hd, rank]
        SVhs = torch.split(SVh_trunc, self.num_key_value_heads, dim=1)  # [bsz, nh, hd, rank]
        for i, layer_idx in enumerate(range(start_layer_idx, end_layer_idx + 1)):
            self.SV[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(SVhs[i])  # [bsz, nh, hd, rank]
        
        del U_trunc, SVh_trunc, SVhs

    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        num_landmarks = k_landmark.shape[-2]
        bsz = k_landmark.shape[0]
        if layer_idx == 0 and self.prefilled_batch == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device='cpu', dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device='cpu', dtype=torch.long)

            # for fused gemm kernel
            self.gemm_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            self.softmax_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            self.norm = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
            self.sum = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
        
        self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark)
        self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark_idx)

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            last_query_states=None
            ):
        
        bsz, _, incoming, _ = new_v_cache.shape # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        max_ctx_chunks = incoming // self.chunk_size
        self.max_ctx_chunks_len = max_ctx_chunks * self.chunk_size
        self.v_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :max_ctx_chunks].copy_(new_v_cache[:, :, :self.max_ctx_chunks_len].reshape(bsz, self.num_key_value_heads, max_ctx_chunks, self.chunk_size*self.head_dim), non_blocking=True) # [bsz, num_kv_heads, max_ctx_chunks, chunk_size*head_dim]

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        # ensure self.chunks is even
        self.chunks = self.chunks - self.chunks % 8
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
    
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget

        self.kernel_offset = self.sparse_start * self.head_dim
        self.kernel_stride = self.v_cache_buffer[layer_idx].shape[-2] * self.head_dim
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(bsz, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(bsz, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(bsz, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        # fill cache for the first time
        chunk_attn = torch.einsum('bhgd,bhcd->bhgc', last_query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim), self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(last_query_states.device)) / math.sqrt(128) # [bsz, 8, 4, chunks]
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype)
        chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(sparse_budget // chunk_size)]
        selected_chunks = self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(last_query_states.device).gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(selected_chunks)
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].max() < self.chunks, f"position_ids exceed the max_length {self.position_ids[layer_idx].max()}"
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].min() >= 0, f"position_ids exceed the min_length {self.position_ids[layer_idx].min()}"
        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(bsz, self.num_key_value_heads, -1)
        value_ = new_v_cache.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        key_ = key_states_roped.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.sparse_start:self.sparse_end].copy_(key_, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            # self.kv_offset += incoming
            self.prefilled_batch += bsz

            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming

                assert torch.any(self.position_ids == -1) == False, f"The cache for offloading is not built correctly, {self.position_ids}"

    ##### Decoding #####
    def get_retrieval_position_ids(self, layer_idx, query_states):
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1

        # gemm_softmax
        shadowkv.batch_gemm_softmax(
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

        # [bsz, 8, seq] --> [bsz, 8, select_sets(sparse_budget // chunk_size)]
        merged_results = torch.topk(chunk_attn.view(self.batch_size, self.num_key_value_heads, -1), k=self.select_sets, dim=-1).indices
        # use merged_results to gather the position_ids: [bsz, 8, chunks] --> [bsz, 8, select_sets]
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        shadowkv.reorder_keys_and_compute_offsets(self.position_ids[layer_idx], selected_chunks, self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, self.select_sets)

        return self.position_ids[layer_idx]

    def get_value_cache(self, layer_idx, position_ids):

        shadowkv.gather_copy_with_offsets(self.v_cache_cpu[layer_idx], self.v_cache_buffer[layer_idx], self.temp, self.offsets, self.cnts, self.signals, self.batch_size, self.num_key_value_heads, int(self.max_ctx_chunks_len*self.head_dim), int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):

        # gather key cache and rope them
        u = self.U[layer_idx // self.group_size] # [bsz, 128k, rank]
        sv = self.SV[layer_idx] # [bsz, 8, 128, rank]

        # print(f"avg cnts: {self.cnts.float().mean()} hit rate: {self.cnts.float().mean() / (self.sparse_budget / 8.0) * 100:.2f}%")
        shadowkv.gather_copy_d2d_with_offsets(self.k_cache_buffer[layer_idx], self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)
        batch_gather_gemm_rotary_pos_emb_cuda(u, sv, cos_sin_cache, position_ids, self.output, self.chunk_size, self.k_cache_buffer[layer_idx], self.sparse_start, self.sparse_end, self.cnts)

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def H2D(self):
        # TODO(max410011): Uncomment these lines during memory usage evaluation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Free temp buffer no longer needed after prefill (saves ~1GB CPU memory)
        del self.key_temp_buffer
        self.key_temp_buffer = None

        # U and SV are already on GPU (computed there during prefill)
        # self.SV = self.SV.to(self.device)  # Already on GPU
        # self.U = self.U.to(self.device)    # Already on GPU
        self.k_landmark = self.k_landmark.to(self.device)
        self.k_landmark_idx = self.k_landmark_idx.to(self.device)

        self.gemm_o = self.gemm_o.to(self.device)
        self.softmax_o = self.softmax_o.to(self.device)
        self.norm = self.norm.to(self.device)
        self.sum = self.sum.to(self.device)

        self.temp = self.temp.to(self.device)
        self.output = self.output.to(self.device)

        # TODO(max410011): Uncomment these lines during memory usage evaluation
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
        self.key_temp_buffer = None
        self.U = None
        self.SV = None

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0

        self.prefilled_batch = 0

    def get_kv_len(self):
        return self.kv_offset


class ShadowKVCache_xKV_CPU:
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


        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 128 + (self.outlier_chunk+self.local_chunk)*self.chunk_size,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
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
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.key_temp_buffer = None
        self.value_temp_buffer = None
        self.U_k = None
        self.SV_k = None
        self.U_v = None
        self.SV_v = None

        self.select_sets = self.sparse_budget // self.chunk_size
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        self.temp = torch.zeros(
            self.batch_size, 
            self.num_key_value_heads, 
            self.select_sets, 
            self.chunk_size*self.head_dim, 
            device='cpu', 
            dtype=self.dtype
        ).contiguous()

        # batch prefill record
        self.prefilled_batch = 0

        # v offload kernels
        # FIXME(max410011): Remove unused variables
        self.block_num = int(self.batch_size * self.num_key_value_heads)
        self.offsets = torch.zeros(self.block_num*(sparse_budget // chunk_size), device=self.device, dtype=torch.int32).contiguous()
        self.cnts = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.signals = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.position_ids = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.select_sets, device=self.device, dtype=torch.int64).fill_(-1).contiguous()

        # k, v compute kernels
        self.output = torch.zeros(
            self.batch_size, 
            self.num_key_value_heads, 
            sparse_budget, 
            self.head_dim, 
            device='cpu', 
            dtype=self.dtype
        ).contiguous()

        # multi-stream
        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        print(f"ShadowKV_xKV_CPU | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} | group_size {self.group_size} | rank_k {self.rank_k} | rank_v {self.rank_v} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    ##### Encoding #####
    def get_svd(self, new_k_cache, new_v_cache, layer_idx):
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        if new_k_cache.shape[1] == self.num_key_value_heads:
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            k_cache = new_k_cache.transpose(1, 2).reshape(new_k_cache.shape[0], -1, self.num_key_value_heads*self.head_dim)
        else:
            k_cache = new_k_cache  # [bsz, prefill, 1024]
        if new_v_cache.shape[1] == self.num_key_value_heads:
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            v_cache = new_v_cache.transpose(1, 2).reshape(new_v_cache.shape[0], -1, self.num_key_value_heads*self.head_dim)
        else:
            v_cache = new_v_cache  # [bsz, prefill, 1024]

        # NOTE(max410011): mini-batch for prefill (4 or 8)
        bsz = k_cache.shape[0]

        if layer_idx == 0 and self.prefilled_batch == 0:
            # init U, SV, temp buffer on GPU for faster SVD computation
            self.U_k = torch.zeros(self.num_layers // self.group_size, self.batch_size, k_cache.shape[1], self.rank_k, device=self.device, dtype=self.dtype)
            self.SV_k = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.head_dim, self.rank_k, device=self.device, dtype=self.dtype)
            self.U_v = torch.zeros(self.num_layers // self.group_size, self.batch_size, v_cache.shape[1], self.rank_v, device=self.device, dtype=self.dtype)

            self.SV_v = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.head_dim, self.rank_v, device=self.device, dtype=self.dtype)
            self.key_temp_buffer = torch.zeros(bsz, k_cache.shape[1], self.num_key_value_heads * self.head_dim * self.group_size, device=self.device, dtype=self.dtype)
            self.value_temp_buffer = torch.zeros(bsz, v_cache.shape[1], self.num_key_value_heads * self.head_dim * self.group_size, device=self.device, dtype=self.dtype)

        # Store temp key/value cache for cross-layer SVD into the correct slice for this layer within the group
        start = (layer_idx % self.group_size) * self.num_key_value_heads * self.head_dim
        end = start + self.num_key_value_heads * self.head_dim
        self.key_temp_buffer[:, :, start:end].copy_(k_cache)  # [bsz, prefill, 1024 * gs]
        self.value_temp_buffer[:, :, start:end].copy_(v_cache)  # [bsz, prefill, 1024 * gs]

        # Apply cross-layer SVD if we have updated the last layer in the group
        if layer_idx % self.group_size == (self.group_size - 1):
            self.grouped_layer_merging(layer_idx)


    @torch.no_grad()
    def grouped_layer_merging(self, last_layer_idx):
        """Perform SVD on grouped layers, inferring dimensions from the tensors."""
        start_layer_idx, end_layer_idx = (last_layer_idx - self.group_size + 1), last_layer_idx

        # Step 1: Apply real SVD directly on the concatenated feature dimension
        U_trunc_k, SVh_trunc_k = fast_svd(self.key_temp_buffer, self.rank_k)
        U_trunc_v, SVh_trunc_v = fast_svd(self.value_temp_buffer, self.rank_v)

        # Step 2: Reshape, split and store U, SV for each layer
        bsz = U_trunc_k.shape[0]
        self.U_k[last_layer_idx // self.group_size][self.prefilled_batch:self.prefilled_batch + bsz].copy_(U_trunc_k) # [bsz, seqlen, rank]
        self.U_v[last_layer_idx // self.group_size][self.prefilled_batch:self.prefilled_batch + bsz].copy_(U_trunc_v)

        # [bsz, rank, nh * hd * gs] -> [bsz, nh * gs, rank, hd]
        SVh_trunc_k = SVh_trunc_k.view(bsz, self.rank_k, self.num_key_value_heads * self.group_size, self.head_dim).transpose(1, 2)
        SVh_trunc_v = SVh_trunc_v.view(bsz, self.rank_v, self.num_key_value_heads * self.group_size, self.head_dim).transpose(1, 2)
        SVh_trunc_k = SVh_trunc_k.transpose(-1, -2)  # use for kernel [bsz, nh * gs, hd, rank]
        SVh_trunc_v = SVh_trunc_v.transpose(-1, -2) # use for kernel [bsz, nh * gs, rank, hd]
        SVhs_k = torch.split(SVh_trunc_k, self.num_key_value_heads, dim=1)  # [bsz, nh, hd, rank]
        SVhs_v = torch.split(SVh_trunc_v, self.num_key_value_heads, dim=1)  # [bsz, nh, rank, hd]
        for i, layer_idx in enumerate(range(start_layer_idx, end_layer_idx + 1)):
            self.SV_k[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(SVhs_k[i]) # [bsz, nh, hd, rank]
            self.SV_v[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(SVhs_v[i])

        del U_trunc_k, SVh_trunc_k, SVhs_k
        del U_trunc_v, SVh_trunc_v, SVhs_v

        # TODO(max410011): Uncomment these lines during memory usage evaluation
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        num_landmarks = k_landmark.shape[-2]
        bsz = k_landmark.shape[0]
        if layer_idx == 0 and self.prefilled_batch == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device='cpu', dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device='cpu', dtype=torch.long)

            # for fused gemm kernel
            self.gemm_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            self.softmax_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            self.norm = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
            self.sum = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
        
        self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark)
        self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark_idx)

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            last_query_states=None
            ):
        
        bsz, _, incoming, _ = new_v_cache.shape # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        max_ctx_chunks = incoming // self.chunk_size
        self.max_ctx_chunks_len = max_ctx_chunks * self.chunk_size

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        # ensure self.chunks is even
        self.chunks = self.chunks - self.chunks % 8
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
    
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget

        self.kernel_offset = self.sparse_start * self.head_dim
        self.kernel_stride = self.v_cache_buffer[layer_idx].shape[-2] * self.head_dim
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(bsz, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(bsz, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(bsz, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        # fill cache for the first time
        chunk_attn = torch.einsum('bhgd,bhcd->bhgc', last_query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim), self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(last_query_states.device)) / math.sqrt(128) # [bsz, 8, 4, chunks]
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype)
        chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(sparse_budget // chunk_size)]
        selected_chunks = self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(last_query_states.device).gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(selected_chunks)
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].max() < self.chunks, f"position_ids exceed the max_length {self.position_ids[layer_idx].max()}"
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].min() >= 0, f"position_ids exceed the min_length {self.position_ids[layer_idx].min()}"
        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(bsz, self.num_key_value_heads, -1)
        value_ = new_v_cache.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        key_ = key_states_roped.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.sparse_start:self.sparse_end].copy_(key_, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            # self.kv_offset += incoming
            self.prefilled_batch += bsz

            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming

                assert torch.any(self.position_ids == -1) == False, f"The cache for offloading is not built correctly, {self.position_ids}"

    ##### Decoding #####
    def get_retrieval_position_ids(self, layer_idx, query_states):
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1

        # gemm_softmax
        shadowkv.batch_gemm_softmax(
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

        # [bsz, 8, seq] --> [bsz, 8, select_sets(sparse_budget // chunk_size)]
        merged_results = torch.topk(chunk_attn.view(self.batch_size, self.num_key_value_heads, -1), k=self.select_sets, dim=-1).indices
        # use merged_results to gather the position_ids: [bsz, 8, chunks] --> [bsz, 8, select_sets]
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        shadowkv.reorder_keys_and_compute_offsets(self.position_ids[layer_idx], selected_chunks, self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, self.select_sets)

        return self.position_ids[layer_idx]

    def get_value_cache(self, layer_idx, position_ids, cos_sin_cache):
        # gather value cache
        u = self.U_v[layer_idx // self.group_size]  # [bsz, 128k, rank]
        sv = self.SV_v[layer_idx]  # [bsz, 8, 128, rank]
        
        # FIXME(max410011): Need a batch_gather_gemm cuda kernel
        # position_ids = (position_ids.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=position_ids.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(self.batch_size, self.num_key_value_heads, -1) # [bsz, 8, select_sets * chunk_size]

        # # indexing, [bsz, 8, sparse_budget, rank]
        # index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1))  # [bsz, 8, sparse_budget, rank]
        # u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1)  # [bsz, 8, 128k, rank]
        # U_head = torch.gather(u_expand, 2, index_expanded)

        # # [bsz, 8, sparse_budget, rank] -matmul- [8, rank, 128] --> [bsz, 8, sparse_budget, 128]
        # result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)

        # # send to buffer
        # self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)

        # FIXME(max410011): Need a batch_gather_gemm cuda kernel
        shadowkv.gather_copy_d2d_with_offsets(self.v_cache_buffer[layer_idx], self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)
        batch_gather_gemm_rotary_pos_emb_cuda(u, sv, cos_sin_cache, self.position_ids[layer_idx], self.output, self.chunk_size, self.v_cache_buffer[layer_idx], self.sparse_start, self.sparse_end, self.cnts, no_rope=True)

        
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        # gather key cache and rope them
        u = self.U_k[layer_idx // self.group_size] # [bsz, 128k, rank]
        sv = self.SV_k[layer_idx] # [bsz, 8, 128, rank]

        shadowkv.gather_copy_d2d_with_offsets(self.k_cache_buffer[layer_idx], self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)
        batch_gather_gemm_rotary_pos_emb_cuda(u, sv, cos_sin_cache, self.position_ids[layer_idx], self.output, self.chunk_size, self.k_cache_buffer[layer_idx], self.sparse_start, self.sparse_end, self.cnts)

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def H2D(self):
        # TODO(max410011): Uncomment these lines during memory usage evaluation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Free temp buffers no longer needed after prefill (saves ~2GB GPU memory)
        del self.key_temp_buffer
        self.key_temp_buffer = None
        del self.value_temp_buffer
        self.value_temp_buffer = None

        # U and SV are already on GPU (computed there during prefill)
        # self.U_k = self.U_k.to(self.device)  # Already on GPU
        # self.U_v = self.U_v.to(self.device)  # Already on GPU
        # self.SV_k = self.SV_k.to(self.device)  # Already on GPU
        # self.SV_v = self.SV_v.to(self.device)  # Already on GPU
        self.k_landmark = self.k_landmark.to(self.device)
        self.k_landmark_idx = self.k_landmark_idx.to(self.device)

        self.gemm_o = self.gemm_o.to(self.device)
        self.softmax_o = self.softmax_o.to(self.device)
        self.norm = self.norm.to(self.device)
        self.sum = self.sum.to(self.device)

        self.temp = self.temp.to(self.device)
        self.output = self.output.to(self.device)

        # TODO(max410011): Uncomment these lines during memory usage evaluation
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
        self.key_temp_buffer = None
        self.value_temp_buffer = None
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U_k = None
        self.U_v = None
        self.SV_k = None
        self.SV_v = None

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0

        self.prefilled_batch = 0

    def get_kv_len(self):
        return self.kv_offset
