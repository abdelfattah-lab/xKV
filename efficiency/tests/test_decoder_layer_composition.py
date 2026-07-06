"""
Verifies bench_decode_layer.py's `_layer_forward` -- the full decoder-layer plumbing
(RMSNorm -> Q/K/V proj -> RoPE -> attention dispatch -> O proj -> residual -> RMSNorm ->
SwiGLU MLP -> residual) -- against an independent plain-PyTorch reference built from the
same weights and hidden_states.

Attention itself is exercised via mode="fa" with a KNOWN (not random) dense KV cache, since
the attention math for every mode is already thoroughly verified elsewhere (test_xkv_kernel_
correctness.py, test_batch_gather_gemm_correctness.py, test_decode_dispatch_correctness.py).
This test's job is to check the surrounding layer composition -- projections, RoPE, residuals,
MLP -- which none of the other tests touch.

Usage:
  pytest efficiency/tests/test_decoder_layer_composition.py -v
  python efficiency/tests/test_decoder_layer_composition.py
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.kv_cache import KV_Cache
from bench_decode_attn import _llama8b_config, _preinit_full
from bench_decode_layer import _make_weights, _rms_norm, _layer_forward, _attn_fa, _RMS_EPS

REL_ERR_TOL = 0.03


def _rms_norm_ref(x, weight, eps=_RMS_EPS):
    x32 = x.float()
    norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * weight.float())


def _rope_ref(t, cos, sin):
    """t: [..., nd]; cos/sin: [nd/2] (matches _layer_forward's position-0-placeholder RoPE)."""
    nd2 = t.shape[-1] // 2
    t1, t2 = t[..., :nd2], t[..., nd2:]
    return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_layer_forward_matches_reference():
    torch.manual_seed(0)
    dev, dtype = torch.device("cuda:0"), torch.bfloat16
    bsz, seqlen = 2, 1024
    cfg = _llama8b_config(num_layers=1)
    nq, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    nd = cfg.hidden_size // nq
    kv_group_num = nq // nkv

    weights = _make_weights(cfg, dev, dtype)
    hidden_states = torch.randn(bsz, 1, cfg.hidden_size, device=dev, dtype=dtype)
    cos_sin = torch.randn(seqlen + 128, nd, device=dev, dtype=dtype)

    kv = KV_Cache(cfg, batch_size=bsz, max_length=seqlen + 128, device=str(dev), dtype=dtype)
    K = torch.randn(bsz, nkv, seqlen, nd, device=dev, dtype=dtype)
    V = torch.randn(bsz, nkv, seqlen, nd, device=dev, dtype=dtype)
    kv.k_cache[0, :, :, :seqlen].copy_(K)
    kv.v_cache[0, :, :, :seqlen].copy_(V)
    _preinit_full(kv, seqlen)
    attn_fn = lambda q: _attn_fa(q, kv, 0)

    out = _layer_forward(hidden_states, weights, cfg, attn_fn, cos_sin)

    # ── independent plain-PyTorch reference, step for step ────────────────────
    residual = hidden_states.float()
    x = _rms_norm_ref(hidden_states, weights["input_norm_w"])

    q = F.linear(x, weights["q_proj"].float()).view(bsz, 1, nq, nd).transpose(1, 2)
    k = F.linear(x, weights["k_proj"].float()).view(bsz, 1, nkv, nd).transpose(1, 2)

    cos = cos_sin[0, :nd // 2].float()
    sin = cos_sin[0, nd // 2:].float()
    q = _rope_ref(q, cos, sin)
    k = _rope_ref(k, cos, sin)   # k here isn't used downstream (fa mode reads from the pre-populated kv cache)

    # dense attention over the KNOWN K/V (roped the same way _attn_fa's flash_attn_with_kvcache
    # would internally treat a pre-populated, already-in-final-form cache: no further RoPE)
    q_squeezed = q[:, :, 0, :]   # [bsz, nq, nd]
    K_exp = K.float().repeat_interleave(kv_group_num, dim=1)
    V_exp = V.float().repeat_interleave(kv_group_num, dim=1)
    scale = nd ** -0.5
    scores = torch.einsum("bhd,bhsd->bhs", q_squeezed, K_exp) * scale
    probs = torch.softmax(scores, dim=-1)
    attn_out = torch.einsum("bhs,bhsd->bhd", probs, V_exp).reshape(bsz, 1, nq * nd)

    attn_out = F.linear(attn_out, weights["o_proj"].float())
    hidden_states2 = residual + attn_out

    residual2 = hidden_states2
    x2 = _rms_norm_ref(hidden_states2.to(dtype), weights["post_norm_w"])
    gate = F.silu(F.linear(x2, weights["gate_proj"].float()))
    up = F.linear(x2, weights["up_proj"].float())
    mlp_out = F.linear(gate * up, weights["down_proj"].float())
    ref = residual2 + mlp_out

    rel_err = (out.float() - ref).norm() / ref.norm()
    assert rel_err < REL_ERR_TOL, f"_layer_forward relative error {rel_err:.4f} too high"


if __name__ == "__main__":
    try:
        test_layer_forward_matches_reference()
        print("[PASS] layer_forward composition")
    except AssertionError as e:
        print(f"[FAIL] layer_forward composition: {e}")
