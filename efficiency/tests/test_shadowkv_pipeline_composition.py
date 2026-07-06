"""
Integration test for the REAL xkey_sr / xkv_sr / shadowkv reconstruction pipeline as wired
together by ShadowKVCache_xKey_CPU / ShadowKVCache_xKV .get_key_cache() / .get_value_cache()
in models/kv_cache.py: buffer bookkeeping (sparse_start/sparse_end, gather_copy_d2d_with_offsets)
composed with the low-rank GEMM+RoPE kernel (batch_gather_gemm_rotary_pos_emb_cuda, unit-tested
in isolation in test_batch_gather_gemm_correctness.py).

This test drives the REAL class methods, not a reimplementation, so it validates that the
buffer-management layer and the reconstruction kernel compose correctly end to end.

Two deliberate scoping choices, both documented inline at point of use:

1. `get_retrieval_position_ids()` (the top-k landmark scoring that decides *which* chunks to
   retrieve) is bypassed -- valid position_ids are injected directly into kv.position_ids
   instead of being produced by that call. Empirically, calling the real scoring op on the
   *synthetic* landmark data bench_decode_attn.py's `_preinit_xkey_sr`/`_preinit_xkv` generate
   (random, meant only for latency timing) returns all -1 (never-selected placeholder) --
   `reorder_keys_and_compute_offsets` appears to assume landmark/offset state from a real
   prior prefill that the benchmark's synthetic init does not replicate. Since evaluating
   *which* chunks a real prefill's landmarks would select is a retrieval-quality/approximation
   question anyway (not a kernel-correctness one -- see test_xkv_kernel_correctness.py's scope
   note for the same reasoning applied to the dense kernel), this test instead directly
   supplies a valid, arbitrary set of chunk indices and checks that get_key_cache/get_value_cache
   reconstruct exactly those chunks correctly.
2. kv.cnts is forced to 0 (see test_batch_gather_gemm_correctness.py's module docstring and
   the same override used there): a "hit" means "already correctly cached from a previous
   decode step"; a single isolated call has no such prior state, so forcing 0 hits (fresh
   reconstruction of every requested chunk) is required to get a fully-defined, checkable
   result from one call.

Usage:
  pytest efficiency/tests/test_shadowkv_pipeline_composition.py -v
  python efficiency/tests/test_shadowkv_pipeline_composition.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.kv_cache import ShadowKVCache_xKey_CPU, ShadowKVCache_xKV
from models.merge_configs import generate_consecutive_palu_config
from bench_decode_attn import _llama8b_config, _preinit_xkey_sr, _preinit_xkv

THETA = 10_000_000.0
REL_ERR_TOL = 0.02


def _sin_cos_ref(positions: torch.Tensor, theta: float, dim: int):
    dim2 = dim // 2
    freqs = torch.arange(dim2, dtype=torch.float32, device=positions.device) * 2 / dim
    freqs = theta ** freqs
    angle = positions[:, None].float() / freqs[None, :]
    return torch.cos(angle), torch.sin(angle)


def _apply_rope_splithalf(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    dim2 = x.shape[-1] // 2
    x0, x1 = x[..., :dim2], x[..., dim2:]
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)


def _reconstruct_selected(U, SV, position_ids, chunk_size, cos_sin_full, apply_rope):
    """Ground-truth reconstruction (+optional RoPE) for exactly the chunks in position_ids.
    U: [bsz, seq_len, rank]   SV: [bsz, heads, head_dim, rank]   position_ids: [bsz, heads, n_chunks]
    """
    head_dim = SV.shape[-2]
    recon_full = torch.einsum("btr,bhdr->bhtd", U.float(), SV.float())   # [bsz, heads, seq_len, head_dim]

    tok_idx = (position_ids[..., None] * chunk_size + torch.arange(chunk_size, device=U.device))
    tok_idx = tok_idx.reshape(*position_ids.shape[:2], -1)
    selected = torch.gather(recon_full, 2, tok_idx[..., None].expand(-1, -1, -1, head_dim))
    if not apply_rope:
        return selected
    cos_full, sin_full = cos_sin_full
    return _apply_rope_splithalf(selected, cos_full[tok_idx], sin_full[tok_idx])


def _inject_position_ids(kv, layer_idx, bsz, nkv):
    """Bypass the real top-k landmark retrieval (see module docstring) with a valid,
    arbitrary set of chunk indices, and set the state get_key_cache/get_value_cache expect
    from a prior get_retrieval_position_ids() call."""
    select_sets = kv.select_sets
    position_ids = torch.stack([
        torch.randperm(kv.chunks, device=kv.device)[:select_sets] for _ in range(bsz * nkv)
    ]).view(bsz, nkv, select_sets).long()
    kv.position_ids[layer_idx].copy_(position_ids)
    kv.cnts.zero_()
    kv.incoming_q_len = 1
    return position_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_size,rank_k", [(4, 384), (1, 96)], ids=["xkey_sr_config", "shadowkv_config"])
def test_xkey_sr_get_key_cache_composition(group_size, rank_k):
    """ShadowKVCache_xKey_CPU (xkey_sr and shadowkv both use this class): real get_key_cache
    buffer bookkeeping composed with the real reconstruction kernel."""
    torch.manual_seed(0)
    dev, dtype = torch.device("cuda:0"), torch.bfloat16
    bsz, prefill_len, sparse_budget = 2, 4096, 512

    cfg = _llama8b_config(num_layers=4)
    merge_cfg = generate_consecutive_palu_config(
        start_layer=0, end_layer=cfg.num_hidden_layers - 1, group_size=group_size, rank_k=rank_k, rank_v=576,
    )
    kv = ShadowKVCache_xKey_CPU(cfg, merge_cfg, max_length=prefill_len + 128, device=dev,
                                 dtype=dtype, batch_size=bsz, sparse_budget=sparse_budget)
    _preinit_xkey_sr(kv, cfg, prefill_len)

    nkv, nd = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    n_groups = kv.num_layers // kv.group_size
    kv.U = torch.randn(n_groups, bsz, prefill_len, kv.rank_k, device=dev, dtype=dtype) * kv.rank_k ** -0.5
    kv.SV = torch.randn(kv.num_layers, bsz, nkv, nd, kv.rank_k, device=dev, dtype=dtype) * kv.rank_k ** -0.5

    layer_idx = 0
    position_ids = _inject_position_ids(kv, layer_idx, bsz, nkv)

    cos, sin = _sin_cos_ref(torch.arange(kv.max_ctx_chunks_len, device=dev), THETA, nd)
    cos_sin = torch.cat([cos, sin], dim=-1).to(dtype)

    out = kv.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, cos_sin_cache=cos_sin)
    assert out.shape[0] == bsz and out.shape[1] == nkv and out.shape[2] >= kv.sparse_end  # sanity: shapes compose

    u = kv.U[layer_idx // kv.group_size]
    sv = kv.SV[layer_idx]
    ref = _reconstruct_selected(u, sv, position_ids, kv.chunk_size, (cos, sin), apply_rope=True)

    got = kv.k_cache_buffer[layer_idx][:, :, kv.sparse_start:kv.sparse_end].float()
    rel_err = (got - ref).norm() / ref.norm()
    assert rel_err < REL_ERR_TOL, f"xkey_sr/shadowkv get_key_cache composition relative error {rel_err:.4f} too high"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_xkv_sr_get_key_and_value_cache_composition():
    """ShadowKVCache_xKV (xkv_sr): real get_key_cache AND get_value_cache buffer bookkeeping
    composed with the real reconstruction kernel (xkv_sr keeps both K and V low-rank on GPU)."""
    torch.manual_seed(0)
    dev, dtype = torch.device("cuda:0"), torch.bfloat16
    bsz, prefill_len, sparse_budget, group_size, rank_k, rank_v = 2, 4096, 512, 4, 384, 576

    cfg = _llama8b_config(num_layers=4)
    merge_cfg = generate_consecutive_palu_config(
        start_layer=0, end_layer=cfg.num_hidden_layers - 1, group_size=group_size, rank_k=rank_k, rank_v=rank_v,
    )
    kv = ShadowKVCache_xKV(cfg, merge_cfg, max_length=prefill_len + 128, device=dev,
                           dtype=dtype, batch_size=bsz, sparse_budget=sparse_budget)
    _preinit_xkv(kv, cfg, prefill_len)

    nkv, nd = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    n_groups = kv.num_layers // kv.group_size
    kv.U_k = torch.randn(n_groups, bsz, prefill_len, rank_k, device=dev, dtype=dtype) * rank_k ** -0.5
    kv.SV_k = torch.randn(kv.num_layers, bsz, nkv, nd, rank_k, device=dev, dtype=dtype) * rank_k ** -0.5
    kv.U_v = torch.randn(n_groups, bsz, prefill_len, rank_v, device=dev, dtype=dtype) * rank_v ** -0.5
    kv.SV_v = torch.randn(kv.num_layers, bsz, nkv, nd, rank_v, device=dev, dtype=dtype) * rank_v ** -0.5

    layer_idx = 0
    position_ids = _inject_position_ids(kv, layer_idx, bsz, nkv)

    cos, sin = _sin_cos_ref(torch.arange(kv.max_ctx_chunks_len, device=dev), THETA, nd)
    cos_sin = torch.cat([cos, sin], dim=-1).to(dtype)

    k_out = kv.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, cos_sin_cache=cos_sin)
    v_out = kv.get_value_cache(layer_idx=layer_idx, position_ids=position_ids, cos_sin_cache=cos_sin)
    assert k_out.shape[0] == bsz and k_out.shape[1] == nkv and k_out.shape[2] >= kv.sparse_end
    assert v_out.shape[0] == bsz and v_out.shape[1] == nkv and v_out.shape[2] >= kv.sparse_end

    u_k, sv_k = kv.U_k[layer_idx // kv.group_size], kv.SV_k[layer_idx]
    u_v, sv_v = kv.U_v[layer_idx // kv.group_size], kv.SV_v[layer_idx]
    ref_k = _reconstruct_selected(u_k, sv_k, position_ids, kv.chunk_size, (cos, sin), apply_rope=True)
    ref_v = _reconstruct_selected(u_v, sv_v, position_ids, kv.chunk_size, (cos, sin), apply_rope=False)

    got_k = kv.k_cache_buffer[layer_idx][:, :, kv.sparse_start:kv.sparse_end].float()
    got_v = kv.v_cache_buffer[layer_idx][:, :, kv.sparse_start:kv.sparse_end].float()
    rel_err_k = (got_k - ref_k).norm() / ref_k.norm()
    rel_err_v = (got_v - ref_v).norm() / ref_v.norm()
    assert rel_err_k < REL_ERR_TOL, f"xkv_sr get_key_cache composition relative error {rel_err_k:.4f} too high"
    assert rel_err_v < REL_ERR_TOL, f"xkv_sr get_value_cache composition relative error {rel_err_v:.4f} too high"


if __name__ == "__main__":
    for group_size, rank_k in [(4, 384), (1, 96)]:
        try:
            test_xkey_sr_get_key_cache_composition(group_size, rank_k)
            print(f"[PASS] xkey_sr/shadowkv composition (group_size={group_size}, rank_k={rank_k})")
        except AssertionError as e:
            print(f"[FAIL] xkey_sr/shadowkv composition (group_size={group_size}, rank_k={rank_k}): {e}")
    try:
        test_xkv_sr_get_key_and_value_cache_composition()
        print("[PASS] xkv_sr composition")
    except AssertionError as e:
        print(f"[FAIL] xkv_sr composition: {e}")
