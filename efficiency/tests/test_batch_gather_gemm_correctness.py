"""
Numerical correctness check for ops/shadowkv/tensor_op.batch_gather_gemm_rotary_pos_emb_cuda,
the CUDA gather + low-rank GEMM (+ optional fused RoPE) kernel shared by the xkey_sr, xkv_sr,
and shadowkv decode paths (via ShadowKVCache_xKey_CPU / ShadowKVCache_xKV .get_key_cache /
.get_value_cache in models/kv_cache.py).

Semantics were confirmed empirically (not guessed) before writing this test, using
identifiable inputs (e.g. a[token] = token_index, identity B, cos=1/sin=0 sanity checks):
  - `position_ids[b, h, c]` is a CHUNK index; each chunk expands to `chunk_size` consecutive
    absolute token positions [chunk_idx*chunk_size, chunk_idx*chunk_size + chunk_size).
  - Reconstruction is `recon[b, h, t, d] = sum_r a[b, t, r] * b_[b, h, d, r]`.
  - `no_rope=True` returns the reconstructed values directly via the `output` tensor,
    unrotated (used for V, which needs no RoPE).
  - `no_rope=False` additionally applies RoPE and writes into `cache[:, :, sparse_start:sparse_end]`.
    `cos_sin[pos, :64]` is cos and `cos_sin[pos, 64:]` is sin for absolute token position `pos`,
    and the rotation is the same split-half convention as ops/fused_attention/rope/v1.py:
      k0_pe = k0*cos - k1*sin ;  k1_pe = k1*cos + k0*sin

Scope: this only tests the reconstruction+RoPE math for an explicit, hand-supplied
position_ids tensor. It does NOT test the landmark top-k retrieval heuristic that decides
*which* chunks to select in production -- that's an approximation-quality question (by
design, ShadowKV only ever looks at a subset of the context), not a kernel-correctness one.
See test_shadowkv_pipeline_composition.py for a test that exercises the real retrieval path.

Usage:
  pytest efficiency/tests/test_batch_gather_gemm_correctness.py -v
  python efficiency/tests/test_batch_gather_gemm_correctness.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from ops.shadowkv.tensor_op import batch_gather_gemm_rotary_pos_emb_cuda

THETA = 10_000_000.0
REL_ERR_TOL = 0.02
COS_SIM_TOL = 0.99


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


def _run_case(batch, heads, head_dim, rank, seq_len, chunk_size, num_chunks, seed=0):
    torch.manual_seed(seed)
    dev, dtype = torch.device("cuda:0"), torch.bfloat16

    a = torch.randn(batch, seq_len, rank, device=dev, dtype=dtype) * rank ** -0.5
    b = torch.randn(batch, heads, head_dim, rank, device=dev, dtype=dtype) * rank ** -0.5

    max_chunks = seq_len // chunk_size
    position_ids = torch.stack([
        torch.randperm(max_chunks, device=dev)[:num_chunks]
        for _ in range(batch * heads)
    ]).view(batch, heads, num_chunks).long()

    sparse_budget = num_chunks * chunk_size
    cnts = torch.zeros(batch * heads, device=dev, dtype=torch.int32)

    # fp32 reference reconstruction over the exact same (batch,head,chunk) selection
    a32, b32 = a.float(), b.float()
    recon_full = torch.einsum("btr,bhdr->bhtd", a32, b32)   # [batch, heads, seq_len, head_dim]

    tok_idx = (position_ids[..., None] * chunk_size + torch.arange(chunk_size, device=dev)).reshape(batch, heads, -1)
    ref_norope = torch.gather(
        recon_full, 2, tok_idx[..., None].expand(-1, -1, -1, head_dim)
    )  # [batch, heads, sparse_budget, head_dim]

    # ── no_rope=True: raw reconstruction, returned via `output` ──────────────
    output = torch.zeros(batch, heads, sparse_budget, head_dim, device=dev, dtype=dtype)
    cache_dummy = torch.zeros(batch, heads, sparse_budget + 128, head_dim, device=dev, dtype=dtype)
    got_norope = batch_gather_gemm_rotary_pos_emb_cuda(
        a, b, torch.zeros(seq_len + 128, head_dim, device=dev, dtype=dtype),
        position_ids, output, chunk_size, cache_dummy, 0, sparse_budget, cnts, no_rope=True,
    ).float()

    rel_err_norope = (got_norope - ref_norope).norm() / ref_norope.norm()

    # ── no_rope=False: reconstruction + RoPE, written into `cache` ───────────
    cos, sin = _sin_cos_ref(torch.arange(seq_len, device=dev), THETA, head_dim)
    cos_sin = torch.cat([cos, sin], dim=-1).to(dtype)   # [seq_len, head_dim]

    cos_tok = cos[tok_idx]   # [batch, heads, sparse_budget, head_dim/2]
    sin_tok = sin[tok_idx]
    ref_roped = _apply_rope_splithalf(ref_norope, cos_tok, sin_tok)

    output2 = torch.zeros(batch, heads, sparse_budget, head_dim, device=dev, dtype=dtype)
    cache = torch.zeros(batch, heads, sparse_budget + 128, head_dim, device=dev, dtype=dtype)
    batch_gather_gemm_rotary_pos_emb_cuda(
        a, b, cos_sin, position_ids, output2, chunk_size, cache, 0, sparse_budget, cnts, no_rope=False,
    )
    got_roped = cache[:, :, :sparse_budget].float()

    rel_err_roped = (got_roped - ref_roped).norm() / ref_roped.norm()

    return rel_err_norope.item(), rel_err_roped.item()


_CASES = {
    "small":            dict(batch=1, heads=2, head_dim=128, rank=64,  seq_len=256,   chunk_size=8, num_chunks=6),
    "multi_batch_head": dict(batch=3, heads=8, head_dim=128, rank=384, seq_len=2048,  chunk_size=8, num_chunks=32),
    "shadowkv_config":  dict(batch=2, heads=8, head_dim=128, rank=96,  seq_len=4096,  chunk_size=8, num_chunks=64),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", _CASES.values(), ids=_CASES.keys())
def test_batch_gather_gemm_matches_dense_reference(case):
    rel_err_norope, rel_err_roped = _run_case(**case)
    assert rel_err_norope < REL_ERR_TOL, f"no_rope relative error {rel_err_norope:.4f} exceeds tolerance"
    assert rel_err_roped < REL_ERR_TOL, f"roped relative error {rel_err_roped:.4f} exceeds tolerance"


if __name__ == "__main__":
    for name, case in _CASES.items():
        rel_err_norope, rel_err_roped = _run_case(**case)
        ok = rel_err_norope < REL_ERR_TOL and rel_err_roped < REL_ERR_TOL
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: no_rope_err={rel_err_norope:.6f}  roped_err={rel_err_roped:.6f}")
