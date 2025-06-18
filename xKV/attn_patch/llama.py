from typing import Optional, Tuple
import types
import torch.nn as nn
import torch


from transformers.models.llama.modeling_llama import (
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaForCausalLM
)

from transformers.cache_utils import Cache
from ..customized_cache.fake_layer_merge_dynamic_cache import FakeLayerMergingCache


def xKV_llama_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    assert not output_attentions, "xKVMistral does not support output_attentions"

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)    
    
    is_prefill = q_len > 1 # assume auto-regressive
        
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    #NOTE(brian1009): Skip the RoPE on key and only apply onto query for now.
    query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
    
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if is_prefill: # prefilling
            #NOTE(brian1009): In our customized cache, we will perform different kind of compression methods
            # To boost performance, we will not use the kv returned from the cache, but instead use the original KV
            assert isinstance(past_key_value, FakeLayerMergingCache)
            past_key_value.update(key_states, value_states, self.layer_idx, mode='prefill', cos=cos, sin=sin)
            key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
        else:
            key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, mode='decode')
    
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def enable_llama_xKV_eval(
    model: LlamaForCausalLM,
):
    for i, layer in enumerate(model.model.layers):
        module = layer.self_attn
        
        if not isinstance(module, LlamaSdpaAttention):
            raise ValueError("Only LlamaSdpaAttention is supported for now")
        
        module.forward = types.MethodType(
            xKV_llama_sdpa_forward, module
        )