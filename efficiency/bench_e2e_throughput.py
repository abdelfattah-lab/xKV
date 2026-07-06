"""
E2E generation throughput benchmark.

Compares throughput across attention methods at long sequence lengths:
  - FA2 runs at its max batch size before OOM (on A100 80GB: bs=4 for both 60k/122k).
  - xKV methods run at larger batch sizes enabled by compressed KV cache.
  - Total throughput (tokens/s) = batch_size × 1000 / est_32layer_ms

For each mode and sequence length, this script:
  1. Sets batch size: FA2→max bs before OOM, xKV methods→max supported bs.
  2. Runs the full decoder layer benchmark (attention + MLP + all projections)
     using a 4-layer loop, divided back to per-layer latency.
  3. Projects to 32 layers (Llama-3.1-8B) to estimate tokens/s.

Usage:
  python bench_e2e_throughput.py
  python bench_e2e_throughput.py --seqlen 122000
  python bench_e2e_throughput.py --mode fa xkv_sr xkey_sr
"""
import sys, os, argparse, json, csv
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import torch
from bench_decode_layer import benchmark_layer_one, ALL_MODES
from bench_decode_attn import _SHADOWKV_GROUP_SIZE, _SHADOWKV_RANK_K, SWEEP_CONFIGS

# ── Constants ─────────────────────────────────────────────────────────────────

_NUM_LAYERS   = 32        # Llama-3.1-8B
_NKV          = 8         # num_key_value_heads
_ND           = 128       # head_dim
_PROBE_LAYERS = 4         # layers to loop for timing (then divide back)

# FA2 OOM boundary on A100 80GB: bs=4 for both seqlens.
#   seqlen=60k  → bs=4  (4 × 7.86 GB = 31.4 GB; bs=8 → 62.9 GB > 56 GB usable → OOM)
#   seqlen=122k → bs=4  (4 × 15.99 GB = 64 GB > 56 GB → OOM at bs=8)
# xKV methods use the maximum batch size enabled by compressed KV cache.
_FA2_OOM_BS:  dict[int, int] = {60_000: 4, 122_000: 4}
_MAX_BS:      dict[int, int] = {sl: max(bss) for sl, bss in SWEEP_CONFIGS}


def _kv_mem_per_item_gb(mode: str, seqlen: int, rank_k: int, rank_v: int,
                        group_size: int) -> float:
    """GPU KV-cache memory (GB) consumed per batch item in each mode."""
    n_grp = _NUM_LAYERS // group_size
    rk_sk = _SHADOWKV_RANK_K      # per-layer rank for shadowkv (group_size=1)
    bytes_per_elem = 2             # BF16

    if mode == "fa":
        # Full K + V for all layers
        return _NUM_LAYERS * seqlen * _NKV * _ND * 2 * bytes_per_elem / 1e9

    elif mode == "xkv":
        # U_k + SV_k + U_v + SV_v
        u  = n_grp * seqlen * (rank_k + rank_v)
        sv = _NUM_LAYERS * _NKV * _ND * (rank_k + rank_v)
        return (u + sv) * bytes_per_elem / 1e9

    elif mode == "xkv_sr":
        # Same structure as xkv (U on GPU, sparse buffer small)
        u  = n_grp * seqlen * (rank_k + rank_v)
        sv = _NUM_LAYERS * _NKV * _ND * (rank_k + rank_v)
        return (u + sv) * bytes_per_elem / 1e9

    elif mode == "xkey_sr":
        # Only K side on GPU; V is CPU-offloaded
        u_k  = n_grp * seqlen * rank_k
        sv_k = _NUM_LAYERS * _NKV * _ND * rank_k
        return (u_k + sv_k) * bytes_per_elem / 1e9

    elif mode == "shadowkv":
        # Per-layer key SVD (group_size=1, rank=96); V on CPU
        u_k  = _NUM_LAYERS * seqlen * rk_sk        # gs=1 → n_grp = num_layers
        sv_k = _NUM_LAYERS * _NKV * _ND * rk_sk
        return (u_k + sv_k) * bytes_per_elem / 1e9

    else:
        raise ValueError(f"Unknown mode: {mode}")


def _bench_batch_size(mode: str, seqlen: int) -> int:
    """Return batch size for throughput comparison.

    FA2 uses its OOM boundary batch size on A100 80GB (bs=4 for both seqlens).
    xKV methods use the maximum batch size enabled by compressed KV cache.
    Falls back to the closest seqlen in SWEEP_CONFIGS if exact match missing.
    """
    lut = _FA2_OOM_BS if mode == "fa" else _MAX_BS
    if seqlen in lut:
        return lut[seqlen]
    closest = min(lut.keys(), key=lambda s: abs(s - seqlen))
    return lut[closest]


# ── Main sweep ────────────────────────────────────────────────────────────────

# Default seqlens
THROUGHPUT_SEQLENS = [sl for sl, _ in SWEEP_CONFIGS]
THROUGHPUT_MODES   = ["fa", "xkey_sr", "xkv_sr", "shadowkv"]


def run_throughput_sweep(
    modes: list = THROUGHPUT_MODES,
    seqlens: list = THROUGHPUT_SEQLENS,
    rank_k: int = 384,
    rank_v: int = 576,
    group_size: int = 4,
    sparse_budget: int = 2048,
    warmup: int = 3,
    iters: int = 10,
    output_dir: str | None = None,
) -> list[dict]:

    total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    results = []

    print(f"\n{'='*78}")
    print(f"  xKV E2E Throughput Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name(0)}  ({total_gpu_gb:.1f} GB)")
    print(f"  FA2 batch sizes (OOM boundary on A100 80GB): "
          + ", ".join(f"seqlen={s//1000}k→bs={b}" for s, b in _FA2_OOM_BS.items()
                      if s in seqlens))
    print(f"  xKV batch sizes (max supported): "
          + ", ".join(f"seqlen={s//1000}k→bs={b}" for s, b in _MAX_BS.items()
                      if s in seqlens))
    print(f"  Throughput = batch_size × 1000 / est_32layer_ms")
    print(f"  rank_k={rank_k}  rank_v={rank_v}  group_size={group_size}"
          f"  sparse_budget={sparse_budget}  probe_layers={_PROBE_LAYERS}")
    print(f"{'='*78}\n")

    for seqlen in seqlens:
        print(f"  ── seqlen = {seqlen//1000}k ──")
        fa_mem = _kv_mem_per_item_gb("fa", seqlen, rank_k, rank_v, group_size)
        fa_toks: float | None = None

        for mode in modes:
            bs = _bench_batch_size(mode, seqlen)
            mem = _kv_mem_per_item_gb(mode, seqlen, rank_k, rank_v, group_size)

            label = f"{mode:8s}  bs={bs:2d}  ({mem:.2f}GB/item)"
            if mode != "fa":
                fa_total = fa_mem * bs
                print(f"  Running {label}  [FA2 at bs={bs} → {fa_total:.1f}GB, OOM]")
                print(f"  Timing  {label} ...", end="", flush=True)
            else:
                print(f"  Running {label} ...", end="", flush=True)
            try:
                r = benchmark_layer_one(
                    mode=mode, seqlen=seqlen, batch_size=bs,
                    rank_k=rank_k, rank_v=rank_v,
                    group_size=group_size, sparse_budget=sparse_budget,
                    warmup=warmup, iters=iters,
                    num_layers=_PROBE_LAYERS,
                )
                est32_ms  = r["avg_ms_per_layer"] * _NUM_LAYERS
                toks_per_s = bs * 1000.0 / est32_ms

                if mode == "fa":
                    fa_toks = toks_per_s
                speedup = toks_per_s / fa_toks if fa_toks else None

                r["batch_size"]             = bs
                r["kv_mem_per_item_gb"]     = mem
                r["est_32layer_ms"]         = est32_ms
                r["est_throughput_toks_per_s"] = toks_per_s
                r["throughput_speedup_vs_fa"]  = speedup
                results.append(r)

                sp_str = f"  {speedup:.2f}× vs fa" if speedup else ""
                print(f"  {toks_per_s:.0f} tok/s  (est32={est32_ms:.0f}ms){sp_str}")

            except Exception as exc:
                print(f"  FAILED: {exc}")
                results.append({"mode": mode, "seqlen": seqlen, "batch_size": bs,
                                "error": str(exc)})
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"{'='*78}")
    print(f"  E2E Throughput Summary (estimated, single-GPU, 32-layer projection)")
    print(f"  FA2 baseline = FA2 at max bs before OOM (A100 80GB)")
    print(f"{'='*78}")
    print(f"  {'Mode':<10} {'SeqLen':>8} {'BS':>4} {'mem/item':>10}"
          f" {'est32ms':>9} {'tok/s':>8} {'vs FA':>8}")
    print(f"  {'-'*72}")
    for r in results:
        if "error" in r:
            print(f"  {'ERROR':<10} {r['seqlen']//1000:>7}k  FAILED")
            continue
        sp  = f"{r.get('throughput_speedup_vs_fa', 0):.2f}×"
        print(f"  {r['mode']:<10} {r['seqlen']//1000:>7}k"
              f" {r['batch_size']:>4} {r.get('kv_mem_per_item_gb', 0):>9.2f}GB"
              f" {r.get('est_32layer_ms', 0):>9.0f} {r.get('est_throughput_toks_per_s', 0):>8.0f}"
              f" {sp:>8}")
    print(f"{'='*78}\n")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"e2e_throughput_{ts}.json")
        csv_path  = os.path.join(output_dir, f"e2e_throughput_{ts}.csv")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        fields = ["mode", "seqlen", "batch_size", "kv_mem_per_item_gb",
                  "avg_ms_per_layer", "est_32layer_ms", "est_throughput_toks_per_s",
                  "throughput_speedup_vs_fa", "rank_k", "rank_v", "group_size",
                  "sparse_budget"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"Results saved → {json_path}")
        print(f"             → {csv_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser("xKV E2E throughput benchmark")
    p.add_argument("--mode",  nargs="+", default=THROUGHPUT_MODES,
                   help="Modes to benchmark (space-separated)")
    p.add_argument("--seqlen", nargs="+", type=int, default=THROUGHPUT_SEQLENS,
                   help="Sequence lengths to benchmark (default: from SWEEP_CONFIGS)")
    p.add_argument("--rank_k",  type=int, default=384)
    p.add_argument("--rank_v",  type=int, default=576)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--sparse_budget", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters",  type=int, default=10)
    p.add_argument("--output_dir", default="results/efficiency")
    args = p.parse_args()

    run_throughput_sweep(
        modes=args.mode,
        seqlens=args.seqlen,
        rank_k=args.rank_k,
        rank_v=args.rank_v,
        group_size=args.group_size,
        sparse_budget=args.sparse_budget,
        warmup=args.warmup,
        iters=args.iters,
        output_dir=args.output_dir,
    )
