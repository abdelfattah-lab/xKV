"""
The one true end-to-end test: real (correlated, not hand-constructed-exact-low-rank) K/V ->
real cross-layer CholQR SVD compression (svd_api.run_svd, the same function
bench_svd_overhead.py times but never checks the accuracy of) -> real decode_attention_fwd
Triton kernel -> attention output.

Every other test in this suite either uses data DEFINED to be exactly low-rank (test_xkv_
kernel_correctness.py, test_batch_gather_gemm_correctness.py -- isolates kernel math from
compression quality) or never calls real SVD at all (all the bench_decode_*.py scripts use
torch.randn directly as U/SV factors; models/kv_cache.py's fast_svd() is dead code -- grep
confirms nothing in efficiency/ calls it). This test is the only one that exercises the
actual compression algorithm feeding the actual attention kernel.

Two things are checked, with different meanings:

1. PASS/FAIL (test_kernel_reconstructs_real_svd_output): does decode_attention_fwd correctly
   reconstruct K/V from what a REAL SVD run actually produces (not idealized exact-low-rank
   factors)? This is a kernel-correctness question and should be tight regardless of how
   lossy the SVD approximation itself is -- the kernel's job is just to correctly compute
   attention from whatever U/B factors it's given.

2. INFORMATIONAL ONLY (reported, not asserted): how much does the SVD approximation itself
   change the attention output vs the original uncompressed K/V? This measures real
   compression quality, which fundamentally depends on how much cross-layer low-rank
   structure the input actually has -- real transformer K/V (the paper's actual claim)
   compresses far better than i.i.d. random Gaussian K/V with only pairwise correlation
   (used here, since no real model weights are available in this harness). A high number
   here is expected for synthetic data and is NOT a sign of a bug; it is reported so a human
   can judge it, not compared against a pass/fail threshold that would be meaningless without
   real LLM activations.

Usage:
  pytest efficiency/tests/test_full_pipeline_real_svd.py -v
  python efficiency/tests/test_full_pipeline_real_svd.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from svd.svd_api import SVDConfig, run_svd
from ops.fused_attention.rope.v1 import decode_attention_fwd

REL_ERR_TOL = 0.1  # looser than the exact-low-rank kernel tests: real SVD compounds bf16+fp16 rounding


def _build_case(seed=0):
    torch.manual_seed(seed)
    dev, dtype = torch.device("cuda:0"), torch.bfloat16
    seqlen, nkv, nq, head_dim, W = 2048, 8, 32, 128, 2
    hidden_dim = W * nkv * head_dim
    rank_k, rank_v = 256, 384
    kv_group_num = nq // nkv

    # two adjacent "layers'" K/V: correlated (matching the paper's actual motivation for
    # cross-layer compression), not identical, and NOT hand-constructed to be exactly low-rank
    K0 = torch.randn(1, seqlen, nkv, head_dim, device=dev, dtype=dtype) * 0.1
    K1 = 0.85 * K0 + 0.15 * torch.randn_like(K0) * 0.1
    V0 = torch.randn(1, seqlen, nkv, head_dim, device=dev, dtype=dtype) * 0.1
    V1 = 0.85 * V0 + 0.15 * torch.randn_like(V0) * 0.1

    K_cat = torch.stack([K0, K1], dim=3).reshape(1, seqlen, hidden_dim)
    V_cat = torch.stack([V0, V1], dim=3).reshape(1, seqlen, hidden_dim)

    Uk, Sk, Vhk = run_svd(K_cat.to(torch.float16), SVDConfig(method="cholqr", rank=rank_k, n_iter=4),
                          power_dtype="fp16", orth="chol")
    Uv, Sv, Vhv = run_svd(V_cat.to(torch.float16), SVDConfig(method="cholqr", rank=rank_v, n_iter=4),
                          power_dtype="fp16", orth="chol")

    # split the joint SVD's per-layer "B" factor for layer 0 (U/k_A is shared across the group)
    Vhk_r = Vhk.float().reshape(1, rank_k, nkv, W, head_dim)
    Vhv_r = Vhv.float().reshape(1, rank_v, nkv, W, head_dim)
    k_A = Uk.to(dtype)
    k_B = (Sk.float()[:, :, None, None] * Vhk_r[:, :, :, 0, :]).permute(0, 2, 1, 3).to(dtype)
    v_A = Uv.to(dtype)
    v_B = (Sv.float()[:, :, None, None] * Vhv_r[:, :, :, 0, :]).permute(0, 2, 1, 3).to(dtype)

    K0_recon = torch.einsum("btr,bhrd->bhtd", k_A.float(), k_B.float())[0]   # [nkv, seqlen, head_dim]
    V0_recon = torch.einsum("btr,bhrd->bhtd", v_A.float(), v_B.float())[0]

    q = torch.randn(1, nq, head_dim, device=dev, dtype=dtype)
    return dict(k_A=k_A, k_B=k_B, v_A=v_A, v_B=v_B,
                K0=K0[0].permute(1, 0, 2), V0=V0[0].permute(1, 0, 2),   # [nkv, seqlen, head_dim]
                K0_recon=K0_recon, V0_recon=V0_recon, q=q, nq=nq, nkv=nkv,
                head_dim=head_dim, kv_group_num=kv_group_num)


def _dense_attention(q, K, V, kv_group_num, head_dim):
    """q: [1,nq,hd]  K,V: [nkv, seqlen, hd]"""
    K_exp = K.repeat_interleave(kv_group_num, dim=0)[None]
    V_exp = V.repeat_interleave(kv_group_num, dim=0)[None]
    scale = head_dim ** -0.5
    scores = torch.einsum("bhd,bhsd->bhs", q.float(), K_exp.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bhsd->bhd", probs, V_exp.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kernel_reconstructs_real_svd_output():
    case = _build_case()
    num_kv_splits = torch.full((1,), 4, dtype=torch.int32, device=case["q"].device)
    out = decode_attention_fwd(
        case["q"], case["k_A"], case["k_B"], case["v_A"], case["v_B"],
        num_kv_splits, 4, sm_scale=case["head_dim"] ** -0.5,
    )
    ref = _dense_attention(case["q"], case["K0_recon"].to(out.dtype), case["V0_recon"].to(out.dtype),
                           case["kv_group_num"], case["head_dim"])
    rel_err = (out.float() - ref).norm() / ref.norm()
    assert rel_err < REL_ERR_TOL, f"kernel vs real-SVD-reconstruction relative error {rel_err:.4f} too high"


def report_compression_quality():
    """Informational only -- not a pass/fail check. See module docstring."""
    case = _build_case()
    num_kv_splits = torch.full((1,), 4, dtype=torch.int32, device=case["q"].device)
    out = decode_attention_fwd(
        case["q"], case["k_A"], case["k_B"], case["v_A"], case["v_B"],
        num_kv_splits, 4, sm_scale=case["head_dim"] ** -0.5,
    )
    ref_orig = _dense_attention(case["q"], case["K0"], case["V0"], case["kv_group_num"], case["head_dim"])
    diff = (out.float() - ref_orig).norm() / ref_orig.norm()
    return diff.item()


if __name__ == "__main__":
    try:
        test_kernel_reconstructs_real_svd_output()
        print("[PASS] kernel correctly reconstructs real SVD output")
    except AssertionError as e:
        print(f"[FAIL] {e}")
    compression_diff = report_compression_quality()
    print(f"[INFO] compressed-attention vs original-uncompressed-attention diff: {compression_diff:.4f} "
          f"(synthetic correlated-random K/V, not real LLM activations -- informational only)")
