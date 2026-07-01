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
import time

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
transformers.logging.set_verbosity_error()

from .tensor_op import layer_norm, apply_rotary_pos_emb, apply_rotary_pos_emb_single
from .prompt_template import Templates, Chat_Templates
from .base import LLM
from .merge_configs import generate_consecutive_palu_config

class Phi3Layer:
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

    def init_parameters(self, hf_layer):

        self.wqkv :torch.Tensor= hf_layer.self_attn.qkv_proj.weight.detach()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()

        self.gate_up_proj = hf_layer.mlp.gate_up_proj.weight.detach()
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

class Phi3(LLM):
    def __init__(self, 
        model_name: str = "microsoft/Phi-3-mini-128k-instruct",
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
        group_size: int = 1,
        rank_k: int = 512,
        rank_v: int = 768,
        fake_svd: bool = False,
    ) -> None:
        
        assert batch_size == 1, "Batch size must be 1"
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.model_name = model_name

        hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype, trust_remote_code=True)
        self.config = hf_model.config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False, trust_remote_code=True)
        self.max_length = max_length

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        self.init_parameters(hf_model)
        self.attn_mode = attn_mode
        self.minference = minference

        self.merge_config = generate_consecutive_palu_config(
            start_layer=start_layer_idx,
            end_layer=end_layer_idx if end_layer_idx != -1 else self.num_layers,
            group_size=group_size,
            rank_k=rank_k,
            rank_v=rank_v
        )

        self.ctx_template = Templates['phi']
        self.chat_template = Chat_Templates['phi']

        self.fake_svd = fake_svd
        self.init_kv_cache(sparse_budget, chunk_size, self.config, rank=rank, merge_config=self.merge_config)

    def _set_cos_sin_cache(self, hf_model):
        dummy_x = torch.tensor(1.0, device=self.device).to(self.dtype)
        position_ids = torch.arange(self.max_length, device=self.device, dtype=torch.long).unsqueeze(0)
        cos_cache, sin_cache = hf_model.model.layers[0].self_attn.rotary_emb(dummy_x, position_ids)
        return cos_cache[0], sin_cache[0]

    def init_parameters(self, hf_model):
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        self.cos_cache, self.sin_cache = self._set_cos_sin_cache(hf_model)
        self.layers :list[Phi3Layer] = []

        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = Phi3Layer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: Phi3Layer,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        qkv = F.linear(hidden_states, buffer.wqkv)
        
        query_pos = num_heads * head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + num_key_value_heads * head_dim]
        value_states = qkv[..., query_pos + num_key_value_heads * head_dim :]
        
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        return query_states, key_states, value_states
    
    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: Phi3Layer
    ):  
        hidden_states = F.linear(attn_output, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        up_states = F.linear(hidden_states, buffer.gate_up_proj)
        gate_proj, up = up_states.chunk(2, dim=-1)
        gate = F.silu(gate_proj)
        hidden_states = gate * up
        hidden_states = F.linear(hidden_states, buffer.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states
    
    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb_single(x, self.cos_cache, self.sin_cache, position_ids)

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb(q, k, self.cos_cache, self.sin_cache, position_ids)