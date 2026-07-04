"""
Numerical correctness check for ops/fused_attention/rope/v1.py (decode_attention_fwd).

The xKV benchmarks (bench_decode_attn.py etc.) only measure latency using random
tensors -- they never check that the Triton kernel's low-rank K reconstruction +
fused RoPE + multi-split attention actually computes the right numbers. This test
builds a reference computation in plain PyTorch/fp32 and compares against the kernel.

K and V are constructed to be EXACTLY low-rank (K = k_A @ k_B, V = v_A @ v_B), so
there is no SVD approximation error to account for -- any mismatch here is a kernel
bug (wrong indexing/strides/RoPE math/split-reduction), not an algorithmic
approximation error.

Scope: only covers the xkv dense kernel (rope/v1.py). xkey_sr/xkv_sr/shadowkv use a
separate sparse-retrieval CUDA pipeline (batch_gather_gemm_rotary_pos_emb_cuda,
torch.ops._shadowkv.*) that is not exercised here.

Usage:
  pytest efficiency/tests/test_xkv_kernel_correctness.py -v
  python efficiency/tests/test_xkv_kernel_correctness.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from ops.fused_attention.rope.v1 import decode_attention_fwd

HEAD_DIM = 128
THETA    = 10_000_000.0
REL_ERR_TOL = 0.02     # bf16 kernel vs fp32 reference; observed ~0.003 in practice
COS_SIM_TOL = 0.99


def _sin_cos_ref(positions: torch.Tensor, theta: float, dim: int = HEAD_DIM):
    dim2 = dim // 2
    freqs = torch.arange(dim2, dtype=torch.float32, device=positions.device) * 2 / dim
    freqs = theta ** freqs
    angle = positions[:, None].float() / freqs[None, :]
    return torch.cos(angle), torch.sin(angle)


def _apply_rope_splithalf(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    dim2 = x.shape[-1] // 2
    x0, x1 = x[..., :dim2], x[..., dim2:]
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)


def _run_case(batch, num_q_heads, num_kv_heads, rank_k, rank_v, kv_len, num_splits, seed=0):
    torch.manual_seed(seed)
    dev, dtype = torch.device("cuda:0"), torch.bfloat16
    kv_group_num = num_q_heads // num_kv_heads
    sm_scale = HEAD_DIM ** -0.5

    k_A = torch.randn(batch, kv_len, rank_k, device=dev, dtype=dtype) * rank_k ** -0.5
    k_B = torch.randn(batch, num_kv_heads, rank_k, HEAD_DIM, device=dev, dtype=dtype) * rank_k ** -0.5
    v_A = torch.randn(batch, kv_len, rank_v, device=dev, dtype=dtype) * rank_v ** -0.5
    v_B = torch.randn(batch, num_kv_heads, rank_v, HEAD_DIM, device=dev, dtype=dtype) * rank_v ** -0.5
    q_raw = torch.randn(batch, num_q_heads, HEAD_DIM, device=dev, dtype=dtype)

    num_kv_splits = torch.full((batch,), num_splits, dtype=torch.int32, device=dev)

    # ── fp32 reference: reconstruct K/V densely, apply RoPE, do plain attention ──
    k_A32, k_B32 = k_A.float(), k_B.float()
    v_A32, v_B32 = v_A.float(), v_B.float()
    q32 = q_raw.float()

    K_full = torch.einsum("bsr,bhrd->bhsd", k_A32, k_B32)
    V_full = torch.einsum("bsr,bhrd->bhsd", v_A32, v_B32)

    cos_k, sin_k = _sin_cos_ref(torch.arange(kv_len, device=dev), THETA)
    K_roped = _apply_rope_splithalf(K_full, cos_k[None, None], sin_k[None, None])

    q_pos = torch.tensor([kv_len], device=dev)   # new token generated right after the cache
    cos_q, sin_q = _sin_cos_ref(q_pos, THETA)
    Q_roped = _apply_rope_splithalf(q32, cos_q[None], sin_q[None])

    K_exp = K_roped.repeat_interleave(kv_group_num, dim=1)
    V_exp = V_full.repeat_interleave(kv_group_num, dim=1)

    scores = torch.einsum("bhd,bhsd->bhs", Q_roped, K_exp) * sm_scale
    probs  = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum("bhs,bhsd->bhd", probs, V_exp)

    # ── kernel computation (bf16, Q pre-roped as real call sites do) ──────────
    q_kernel_in = Q_roped.to(dtype)
    kernel_out = decode_attention_fwd(
        q_kernel_in, k_A, k_B, v_A, v_B,
        num_kv_splits, num_splits, sm_scale=sm_scale, theta=THETA,
    ).float()

    abs_err = (kernel_out - ref_out).abs()
    rel_err = (abs_err.norm() / ref_out.norm()).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        kernel_out.reshape(-1, HEAD_DIM), ref_out.reshape(-1, HEAD_DIM), dim=-1
    )
    return rel_err, cos_sim.min().item()


_CASES = {
    "multi_split":       dict(batch=2, num_q_heads=32, num_kv_heads=8, rank_k=384, rank_v=576, kv_len=4096, num_splits=4),
    "single_split_odd_len": dict(batch=2, num_q_heads=32, num_kv_heads=8, rank_k=384, rank_v=576, kv_len=4001, num_splits=1),
    "production_scale_60k": dict(batch=4, num_q_heads=32, num_kv_heads=8, rank_k=384, rank_v=576, kv_len=60_000, num_splits=30),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", _CASES.values(), ids=_CASES.keys())
def test_xkv_kernel_matches_dense_reference(case):
    rel_err, min_cos_sim = _run_case(**case)
    assert rel_err < REL_ERR_TOL, f"relative L2 error {rel_err:.4f} exceeds tolerance {REL_ERR_TOL}"
    assert min_cos_sim > COS_SIM_TOL, f"min cosine similarity {min_cos_sim:.4f} below tolerance {COS_SIM_TOL}"


if __name__ == "__main__":
    for name, case in _CASES.items():
        rel_err, min_cos_sim = _run_case(**case)
        status = "PASS" if rel_err < REL_ERR_TOL and min_cos_sim > COS_SIM_TOL else "FAIL"
        print(f"[{status}] {name}: rel_err={rel_err:.6f}  min_cos_sim={min_cos_sim:.6f}")
