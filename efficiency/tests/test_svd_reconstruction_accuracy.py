"""
Reconstruction-accuracy check for the real CholQR SVD used by bench_svd_overhead.py
(svd_api.run_svd -> random_cholesky_v6.randomized_svd_fp16).

bench_svd_overhead.py is the ONLY place in efficiency/ that calls real SVD -- and it only
measures TIMING, never reconstruction error. All the other benchmarks (bench_decode_attn.py,
bench_decode_layer.py, bench_e2e_throughput.py) use torch.randn directly as U/SV factors;
they never call SVD at all (models/kv_cache.py's fast_svd() is defined but unreachable dead
code -- verified via grep, nothing in efficiency/ calls it). So this is the only test in the
suite that touches the actual compression algorithm's numerical quality.

Correctness signal: for a rank-r truncated SVD, the best *possible* reconstruction error is
given by the Eckart-Young theorem: sqrt(sum of squared singular values beyond rank r), computed
from an exact torch.linalg.svd. A correct low-rank SVD implementation should land close to
that optimal bound -- checking against the optimum (not just "error is small") is what
actually tests whether the algorithm is finding the right (top) subspace, not merely
producing *a* low-rank factorization.

Usage:
  pytest efficiency/tests/test_svd_reconstruction_accuracy.py -v
  python efficiency/tests/test_svd_reconstruction_accuracy.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from svd.svd_api import SVDConfig, run_svd

# how much worse than the theoretically-optimal rank-r truncation we tolerate
OPTIMALITY_TOL = 1.5


def _optimal_rank_r_error(tensor: torch.Tensor, rank: int) -> float:
    _, s, _ = torch.linalg.svd(tensor.float(), full_matrices=False)
    tail = s[..., rank:]
    return (tail.pow(2).sum(dim=-1).sqrt().mean()).item()


def _run_case(seqlen, hidden_dim, rank, n_iter=4, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("cuda:0")
    tensor = torch.randn(1, seqlen, hidden_dim, device=dev, dtype=torch.bfloat16)

    cfg = SVDConfig(method="cholqr", rank=rank, n_iter=n_iter)
    U, S, Vh = run_svd(tensor.to(torch.float16), cfg, power_dtype="fp16", orth="chol")
    recon = (U.float() @ torch.diag_embed(S.float()) @ Vh.float())

    actual_err = (recon - tensor.float()).norm(dim=(-2, -1)).mean().item()
    optimal_err = _optimal_rank_r_error(tensor, rank)
    return actual_err, optimal_err


_CASES = {
    "W2_64k":  dict(seqlen=65_536,  hidden_dim=2 * 8 * 128, rank=256),   # window=2, base_rank=128
    "W4_64k":  dict(seqlen=65_536,  hidden_dim=4 * 8 * 128, rank=512),   # window=4, base_rank=128
    "W2_160k": dict(seqlen=163_840, hidden_dim=2 * 8 * 128, rank=256),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", _CASES.values(), ids=_CASES.keys())
def test_cholqr_reconstruction_near_optimal(case):
    actual_err, optimal_err = _run_case(**case)
    assert actual_err < optimal_err * OPTIMALITY_TOL, (
        f"CholQR reconstruction error {actual_err:.2f} is more than {OPTIMALITY_TOL}x "
        f"the theoretical optimum {optimal_err:.2f} (full-SVD rank-r truncation)"
    )


if __name__ == "__main__":
    for name, case in _CASES.items():
        actual_err, optimal_err = _run_case(**case)
        ratio = actual_err / optimal_err
        ok = ratio < OPTIMALITY_TOL
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: actual={actual_err:.3f} optimal={optimal_err:.3f} ratio={ratio:.3f}")
