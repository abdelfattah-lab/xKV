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
from transformers import AutoModel, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

transformers.logging.set_verbosity_error()

import vllm

from .tensor_op import layer_norm
from .prompt_template import Templates, Chat_Templates
from .base import LLM
from .merge_configs import generate_consecutive_palu_config

class GLMLayer:
    def __init__(self, layer_idx) -> None:
        
        self.wqkv :torch.Tensor = None
        self.bqkv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx

    def init_parameters(self, hf_layer: LlamaDecoderLayer):

        self.wqkv: torch.Tensor = hf_layer.self_attention.query_key_value.weight.detach()
        self.bqkv: torch.Tensor = hf_layer.self_attention.query_key_value.bias.detach()
        self.wo :torch.Tensor= hf_layer.self_attention.dense.weight.detach()

        self.up_proj = hf_layer.mlp.dense_h_to_4h.weight.detach()
        self.down_proj = hf_layer.mlp.dense_4h_to_h.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.eps

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.eps
    
    def init_gpu(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wqkv = self.wqkv.to(device, non_blocking=True)
        self.bqkv = self.bqkv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)

class GLMConfig:
    def __init__(self, config) -> None:
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.multi_query_group_num
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads

class GLM(LLM):
    def __init__(self, 
        model_name: str = "THUDM/glm-4-9b-chat-1m",
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
        
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        hf_model = AutoModel.from_pretrained(self.model_name, torch_dtype=self.dtype, trust_remote_code=True)
        self.config = hf_model.config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False, trust_remote_code=True)
        self.max_length = max_length
        
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.multi_query_group_num
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.seq_length
        self.rope_ratio = self.config.rope_ratio

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

        self.ctx_template = Templates['glm']
        self.chat_template = Chat_Templates['glm']
        self.prefix_template = Templates['glm']

        self.vocab_size = self.config.vocab_size

        self.fake_svd = fake_svd
        self.init_kv_cache(sparse_budget, chunk_size, GLMConfig(self.config), rank=rank, merge_config=self.merge_config)

    def _set_cos_sin_cache(self, hf_model):
        return hf_model.transformer.rotary_pos_emb(self.max_length + 1024).to(self.device).transpose(-1, -2).contiguous().view(-1, 64)

    def init_parameters(self, hf_model):
        self.embed_tokens = hf_model.transformer.embedding.word_embeddings.weight.detach().to(self.device)
        self.lm_head = hf_model.transformer.output_layer.weight.detach().to(self.device)
        self.norm_weight = hf_model.transformer.encoder.final_layernorm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.transformer.encoder.final_layernorm.eps
        self.cos_sin_cache = self._set_cos_sin_cache(hf_model)

        self.layers :list[GLMLayer] = []

        for idx, hf_layer in enumerate(hf_model.transformer.encoder.layers):
            layer = GLMLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.transformer.encoder.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: GLMLayer,
        num_heads: int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        
        mixed_x_layer = F.linear(input=hidden_states, weight=buffer.wqkv, bias=buffer.bqkv)
        
        (query_states, key_states, value_states) = mixed_x_layer.split(
                [
                    num_heads * head_dim,
                    num_key_value_heads * head_dim,
                    num_key_value_heads * head_dim,
                ],
                dim=-1,
            )

        return query_states, key_states, value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: GLMLayer
    ):  
        hidden_states = F.linear(attn_output, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)

        hidden_states = F.linear(hidden_states, buffer.up_proj)
        d = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        vllm._custom_ops.silu_and_mul(out, hidden_states)
        
        hidden_states = F.linear(out, buffer.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        vllm._custom_ops.rotary_embedding(position_ids, q, k, 128, self.cos_sin_cache, False)
        bsz = q.shape[0]
        q = q.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return q, k

    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3: # x: [bsz, seq, 1024]
            x = x.view(x.size(0), x.size(1), -1, 128).transpose(1, 2) # [bsz, heads, seq, 128]
        # print(x.shape, position_ids.shape)
        if len(position_ids.shape) == 1: # position_ids: [seq]
            position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1)
        if len(position_ids.shape) == 2: # position_ids: [bsz, seq]
            position_ids = position_ids.unsqueeze(1).expand(-1, x.size(1), -1)
        rope_cache = self.cos_sin_cache[position_ids] # [max_len, 64] --> [bsz, heads, seq, 64]
        rot_dim = 64
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]

        # print(x.shape, rope_cache.shape)
        x_out2 = torch.stack(
            [
                x[..., 0::2] * rope_cache[..., :32] - x[..., 1::2] * rope_cache[..., 32:],
                x[..., 1::2] * rope_cache[..., :32] + x[..., 0::2] * rope_cache[..., 32:],
            ],
            -1,
        ) # [bsz, heads, seq, 64, 2]

        x_out2 = x_out2.flatten(3)
        return torch.cat((x_out2, x_pass), dim=-1)
