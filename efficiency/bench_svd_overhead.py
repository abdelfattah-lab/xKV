"""
SVD overhead benchmark for xKV prefill.

Measures the time to compute cross-layer SVD during prefill for xKV, across
different sequence lengths and window sizes (W=2, W=4).

For each group of W layers, the concatenated KV cache has shape:
  K_cat ∈ R^{L × (W * Hkv * head_dim)}

Total SVD time = n_groups × per_group_SVD_time
  n_groups = num_layers / W  (e.g. 16 groups for W=2, 8 groups for W=4)

Model config: Llama-3.1-8B (32 layers, 8 KV heads, head_dim=128)
Seqlens: 64k, 128k, 160k, 256k

Methods:
  cholqr  : custom CholQR + power iterations
  lowrank : torch.svd_lowrank (reference, slower)

Usage:
  python bench_svd_overhead.py
  python bench_svd_overhead.py --method cholqr --seqlens 65536 131072
  python bench_svd_overhead.py --n_iter 2 --output_dir results/efficiency
"""
import sys, os, json, csv, argparse
from pathlib import Path
from datetime import datetime

import torch

# ── Locate svd_methods (bundled in efficiency/) ───────────────────────────────
_KVSVD_ROOT = Path(os.environ.get("KVSVD_ROOT", Path(__file__).parent))
_CHOLQR_AVAILABLE = False
try:
    if str(_KVSVD_ROOT) not in sys.path:
        sys.path.insert(0, str(_KVSVD_ROOT))
    from svd.svd_api import run_svd, SVDConfig
    _CHOLQR_AVAILABLE = True
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
NUM_LAYERS   = 32   # Llama-3.1-8B
NUM_KV_HEADS = 8
HEAD_DIM     = 128

SEQLENS            = [65_536, 131_072, 163_840, 262_144]
WIN_SIZES          = [2, 4]
BASE_RANK_PER_LAYER = 128  # rank per layer; total rank = BASE_RANK_PER_LAYER × W (8× compression)


# ── SVD implementations ───────────────────────────────────────────────────────

def svd_cholqr(tensor: torch.Tensor, rank: int, n_iter: int = 2):
    """Custom CholQR-based randomized SVD."""
    if not _CHOLQR_AVAILABLE:
        raise RuntimeError("cholqr kernel not available; set KVSVD_ROOT to the kv-svd directory")
    cfg = SVDConfig(rank=rank, n_iter=n_iter)
    return run_svd(tensor.to(torch.float16), cfg, power_dtype="fp16", orth="chol")


def svd_lowrank(tensor: torch.Tensor, rank: int, n_iter: int = 2):
    """torch.svd_lowrank baseline (slower, no custom kernel)."""
    U, S, Vh = torch.svd_lowrank(tensor.float(), q=rank, niter=n_iter)
    SVh = torch.matmul(torch.diag_embed(S), Vh.transpose(-2, -1))
    return U.to(tensor.dtype), SVh.to(tensor.dtype)


# ── Benchmark one (seqlen, W) combination ─────────────────────────────────────

@torch.inference_mode()
def benchmark_svd(seqlen: int, window_size: int, base_rank: int = BASE_RANK_PER_LAYER,
                  method: str = "cholqr", n_iter: int = 2,
                  warmup: int = 3, iters: int = 10) -> dict:
    """
    Time SVD for a single layer-group, then multiply by n_groups for total time.

    total_time = per_group_time × (NUM_LAYERS // window_size)
    """
    device = torch.device("cuda:0")
    dtype  = torch.bfloat16

    n_groups   = NUM_LAYERS // window_size
    hidden_dim = window_size * NUM_KV_HEADS * HEAD_DIM   # W × 8 × 128
    rank       = base_rank * window_size                  # 8× compression: rank/hidden_dim = 1/8

    # Single group input: [1, seqlen, hidden_dim]
    tensor = torch.randn(1, seqlen, hidden_dim, device=device, dtype=dtype)

    if method == "cholqr":
        fn = lambda: svd_cholqr(tensor, rank=rank, n_iter=n_iter)
    elif method == "lowrank":
        fn = lambda: svd_lowrank(tensor, rank=rank, n_iter=n_iter)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs (per group)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 1000.0)   # → seconds

    per_group_s = sum(times) / len(times)
    total_s     = per_group_s * n_groups   # scale to all groups

    return {
        "seqlen": seqlen,
        "window_size": window_size,
        "n_groups": n_groups,
        "hidden_dim": hidden_dim,
        "rank": rank,
        "method": method,
        "n_iter": n_iter,
        "per_group_s": per_group_s,
        "total_s": total_s,
        "avg_ms": total_s * 1000.0,
    }


# ── Full sweep ───────────────────────────────────────────────────────────────

def run_sweep(seqlens=SEQLENS, win_sizes=WIN_SIZES,
                    method="cholqr", n_iter=2,
                    base_rank=BASE_RANK_PER_LAYER, warmup=3, iters=10,
                    output_dir=None) -> list[dict]:

    print(f"\n{'='*65}")
    print(f"  xKV SVD Overhead Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  method={method}  n_iter={n_iter}  base_rank={base_rank}  (rank=base_rank×W)")
    print(f"{'='*65}\n")
    print(f"  {'Seqlen':>8}  {'W':>3}  {'Groups':>7}  {'SVD time (s)':>14}  {'% of est. prefill':>18}")
    print(f"  {'-'*65}")

    # Estimated prefill times for Llama-3.1-8B on A100 (seconds), used for overhead %.
    PREFILL_EST_A100 = {65_536: 9.97, 131_072: 32.00, 163_840: 47.84, 262_144: 113.67}

    results = []
    for sl in seqlens:
        for W in win_sizes:
            r = benchmark_svd(seqlen=sl, window_size=W, base_rank=base_rank,
                              method=method, n_iter=n_iter,
                              warmup=warmup, iters=iters)
            ref_prefill = PREFILL_EST_A100.get(sl)
            pct = (r["total_s"] / ref_prefill * 100) if ref_prefill else None
            r["pct_of_prefill_a100"] = pct
            results.append(r)
            pct_str = f"{pct:.2f}% (A100 ref)" if pct else "—"
            print(f"  {sl//1024:>7}k  {W:>3}  {r['n_groups']:>7}  {r['total_s']:>14.4f}s  {pct_str:>18}")

    # Summary table
    print(f"\n  Summary (A100 reference prefill times for overhead %):")
    print(f"  {'Seqlen':>8}  {'SVD W=2 (s)':>12}  {'SVD W=4 (s)':>12}")
    print(f"  {'-'*45}")
    by_sl: dict[int, dict[int, float]] = {}
    for r in results:
        by_sl.setdefault(r["seqlen"], {})[r["window_size"]] = r["total_s"]
    for sl in seqlens:
        w2 = by_sl.get(sl, {}).get(2, float("nan"))
        w4 = by_sl.get(sl, {}).get(4, float("nan"))
        print(f"  {sl//1024:>7}k  {w2:>12.4f}  {w4:>12.4f}")

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"svd_overhead_{ts}.json")
        csv_path  = os.path.join(output_dir, f"svd_overhead_{ts}.csv")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        fields = ["seqlen", "window_size", "n_groups", "rank", "method",
                  "n_iter", "per_group_s", "total_s", "avg_ms", "pct_of_prefill_a100"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"\nResults saved → {json_path}")
        print(f"             → {csv_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser("xKV SVD overhead benchmark")
    p.add_argument("--seqlens", nargs="+", type=int, default=SEQLENS)
    p.add_argument("--win_sizes", nargs="+", type=int, default=WIN_SIZES)
    p.add_argument("--method",  default="cholqr", choices=["cholqr", "lowrank"])
    p.add_argument("--n_iter",    type=int, default=2, help="Power iterations for randomized SVD")
    p.add_argument("--base_rank", type=int, default=BASE_RANK_PER_LAYER,
                   help="Base rank per layer; actual rank = base_rank × W (maintains 8× compression)")
    p.add_argument("--warmup",  type=int, default=3)
    p.add_argument("--iters",   type=int, default=10)
    p.add_argument("--output_dir", default="results/efficiency")
    args = p.parse_args()

    run_sweep(seqlens=args.seqlens, win_sizes=args.win_sizes,
                    method=args.method, n_iter=args.n_iter, base_rank=args.base_rank,
                    warmup=args.warmup, iters=args.iters,
                    output_dir=args.output_dir)
