"""
Decode-step attention latency benchmark for xKV.

Tests attention modes on Llama-3.1-8B config (32 Q-heads, 8 KV-heads, head_dim=128):
  - fa        : FlashAttention-2 (full KV cache baseline)
  - xkv       : xKV dense reconstruction (cross-layer SVD, all tokens reconstructed)
  - xkey_sr   : xK-SR (cross-layer key SVD + CPU-offloaded values, selective recon)
  - xkv_sr    : xKV-SR (cross-layer K+V SVD on GPU, selective reconstruction)

Default settings:
  seqlen=60k  → batch_sizes [8, 16, 32]
  seqlen=122k → batch_sizes [4, 8, 16]
  rank_k=384, rank_v=576, group_size=4, sparse_budget=2048

Usage:
  python bench_decode_attn.py
  python bench_decode_attn.py --seqlen 60000 --batch_size 16 --mode all
  python bench_decode_attn.py --mode fa --seqlen 60000 --batch_size 8 --iters 50
"""
import sys, os, argparse, json, csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import torch
from flash_attn import flash_attn_with_kvcache

from models.kv_cache import KV_Cache, ShadowKVCache_xKey_CPU, ShadowKVCache_xKV
from models.merge_configs import generate_consecutive_palu_config
from ops.fused_attention.rope.v1   import decode_attention_fwd as xkv_attn


# ── Llama-3.1-8B model config ─────────────────────────────────────────────────
# num_hidden_layers=4 (one full group_size=4 group) keeps KV cache allocation
# manageable while preserving the correct per-layer tensor shapes for benchmarking.
def _llama8b_config(num_layers: int = 4):
    return SimpleNamespace(
        hidden_size=4096,
        num_hidden_layers=num_layers,
        num_attention_heads=32,
        num_key_value_heads=8,
    )


# ── Pre-init helpers ──────────────────────────────────────────────────────────

@torch.inference_mode()
def _preinit_full(kv: KV_Cache, prefill_len: int):
    kv.kv_offset = min(prefill_len, kv.max_length)


@torch.inference_mode()
def _preinit_xkey_sr(kv: ShadowKVCache_xKey_CPU, cfg, prefill_len: int):
    dev, dtype = torch.device("cuda:0"), kv.dtype
    bsz = kv.batch_size
    nkv, nd = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    nq   = cfg.num_attention_heads

    max_ctx = max(prefill_len // kv.chunk_size, 1)
    kv.max_ctx_chunks_len = max_ctx * kv.chunk_size
    chunks = max_ctx - kv.local_chunk
    chunks -= chunks % 8
    chunks = max(chunks, 8)
    kv.chunks = chunks
    kv.prefill_local = prefill_len - chunks * kv.chunk_size

    oc = kv.outlier_chunk
    if oc >= chunks:
        oc = max(chunks // 2, 1)
    kv.sparse_start  = kv.prefill_local + oc * kv.chunk_size
    kv.sparse_end    = kv.sparse_start + kv.sparse_budget
    kv.kernel_offset = kv.sparse_start * nd
    kv.kernel_stride = kv.v_cache_buffer.shape[-2] * nd

    kv.U   = torch.randn(kv.num_layers // kv.group_size, bsz, prefill_len, kv.rank_k, device=dev, dtype=dtype)
    kv.SV  = torch.randn(kv.num_layers, bsz, nkv, nd, kv.rank_k, device=dev, dtype=dtype)

    nlm = max(chunks - oc, 1)
    kv.k_landmark     = torch.randn(kv.num_layers, bsz, nkv, nlm, nd, device=dev, dtype=dtype)
    kv.k_landmark_idx = torch.randint(0, chunks, (kv.num_layers, bsz, nkv, nlm), device=dev, dtype=torch.long)

    ngrp  = nq // nkv
    tiles = (nlm + 256 - 1) // 256
    kv.gemm_o   = torch.randn(bsz, nkv, ngrp, nlm, device=dev, dtype=torch.bfloat16).contiguous()
    kv.softmax_o= torch.randn(bsz, nkv, ngrp, nlm, device=dev, dtype=torch.bfloat16).contiguous()
    kv.norm     = torch.randn(bsz*nkv, ngrp, tiles, device=dev, dtype=torch.float).contiguous()
    kv.sum      = torch.randn(bsz*nkv, ngrp, tiles, device=dev, dtype=torch.float).contiguous()

    kv.position_ids.fill_(-1)
    kv.v_cache_cpu.normal_()   # faults in CPU L3 cache to simulate warm post-prefill state
    kv.k_cache_buffer.normal_()
    kv.v_cache_buffer.normal_()


@torch.inference_mode()
def _preinit_xkv(kv: ShadowKVCache_xKV, cfg, prefill_len: int):
    dev, dtype = torch.device("cuda:0"), kv.dtype
    bsz = kv.batch_size
    nkv, nd = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    nq   = cfg.num_attention_heads

    max_ctx = max(prefill_len // kv.chunk_size, 1)
    kv.max_ctx_chunks_len = max_ctx * kv.chunk_size
    chunks = max_ctx - kv.local_chunk
    chunks -= chunks % 8
    chunks = max(chunks, 8)
    kv.chunks = chunks
    kv.prefill_local = prefill_len - chunks * kv.chunk_size

    oc = kv.outlier_chunk
    if oc >= chunks:
        oc = max(chunks // 2, 1)
    kv.sparse_start  = kv.prefill_local + oc * kv.chunk_size
    kv.sparse_end    = kv.sparse_start + kv.sparse_budget
    kv.kernel_offset = kv.sparse_start * nd
    kv.kernel_stride = kv.v_cache_buffer.shape[-2] * nd

    kv.U_k  = torch.randn(kv.num_layers // kv.group_size, bsz, prefill_len, kv.rank_k, device=dev, dtype=dtype)
    kv.SV_k = torch.randn(kv.num_layers, bsz, nkv, nd, kv.rank_k, device=dev, dtype=dtype)
    kv.U_v  = torch.randn(kv.num_layers // kv.group_size, bsz, prefill_len, kv.rank_v, device=dev, dtype=dtype)
    kv.SV_v = torch.randn(kv.num_layers, bsz, nkv, nd, kv.rank_v, device=dev, dtype=dtype)

    nlm = max(chunks - oc, 1)
    kv.k_landmark     = torch.randn(kv.num_layers, bsz, nkv, nlm, nd, device=dev, dtype=dtype)
    kv.k_landmark_idx = torch.randint(0, chunks, (kv.num_layers, bsz, nkv, nlm), device=dev, dtype=torch.long)

    ngrp  = nq // nkv
    tiles = (nlm + 256 - 1) // 256
    kv.gemm_o   = torch.randn(bsz, nkv, ngrp, nlm, device=dev, dtype=torch.bfloat16).contiguous()
    kv.softmax_o= torch.randn(bsz, nkv, ngrp, nlm, device=dev, dtype=torch.bfloat16).contiguous()
    kv.norm     = torch.randn(bsz*nkv, ngrp, tiles, device=dev, dtype=torch.float).contiguous()
    kv.sum      = torch.randn(bsz*nkv, ngrp, tiles, device=dev, dtype=torch.float).contiguous()

    kv.k_cache_buffer.normal_()
    kv.v_cache_buffer.normal_()


# ── Attention dispatch ────────────────────────────────────────────────────────

@torch.inference_mode()
def _decode_fa(q, kv: KV_Cache, layer_idx: int, **_):
    seqlen = kv.kv_offset
    k = kv.k_cache[layer_idx][:, :, :seqlen]
    v = kv.v_cache[layer_idx][:, :, :seqlen]
    return flash_attn_with_kvcache(
        q=q.transpose(1, 2), k_cache=k.transpose(1, 2), v_cache=v.transpose(1, 2), causal=True,
    )


@torch.inference_mode()
def _decode_xkv(q, kv_tensors: dict, **_):
    """xKV dense: fused Triton kernel with RoPE on K (rope/v1)."""
    k_A   = kv_tensors["k_A"]
    k_B   = kv_tensors["k_B"]
    v_A   = kv_tensors["v_A"]
    v_B   = kv_tensors["v_B"]
    num_kv_splits = kv_tensors["num_kv_splits"]
    max_kv_splits = kv_tensors["max_kv_splits"]
    q_in = q[:, :, 0, :]
    return xkv_attn(q_in, k_A, k_B, v_A, v_B, num_kv_splits, max_kv_splits)


@torch.inference_mode()
def _decode_xkey_sr(q, kv: ShadowKVCache_xKey_CPU, layer_idx: int, cos_sin: torch.Tensor,
                 curr_stream, value_stream, **_):
    pos_ids = kv.get_retrieval_position_ids(layer_idx=layer_idx, query_states=q)
    with torch.cuda.stream(value_stream):
        value_stream.wait_stream(curr_stream)
        v = kv.get_value_cache(layer_idx, pos_ids)
    k = kv.get_key_cache(layer_idx=layer_idx, position_ids=pos_ids,
                         cos_sin_cache=cos_sin)
    curr_stream.wait_stream(value_stream)
    return flash_attn_with_kvcache(
        q=q.transpose(1, 2), k_cache=k.transpose(1, 2), v_cache=v.transpose(1, 2), causal=True,
    )


@torch.inference_mode()
def _decode_xkv_sr(q, kv: ShadowKVCache_xKV, layer_idx: int, cos_sin: torch.Tensor, **_):
    pos_ids = kv.get_retrieval_position_ids(layer_idx=layer_idx, query_states=q)
    v = kv.get_value_cache(layer_idx, pos_ids, cos_sin)
    k = kv.get_key_cache(layer_idx=layer_idx, position_ids=pos_ids,
                         cos_sin_cache=cos_sin)
    return flash_attn_with_kvcache(
        q=q.transpose(1, 2), k_cache=k.transpose(1, 2), v_cache=v.transpose(1, 2), causal=True,
    )


# ── Single benchmark run ──────────────────────────────────────────────────────

@torch.inference_mode()
def benchmark_one(mode: str, seqlen: int, batch_size: int,
                  rank_k: int, rank_v: int, group_size: int, sparse_budget: int,
                  warmup: int, iters: int, layer_idx: int = 0) -> dict:
    cfg    = _llama8b_config(num_layers=4)   # 4 = one group; realistic per-layer shapes
    dev    = torch.device("cuda:0")
    dtype  = torch.bfloat16
    nq, nkv, nd = cfg.num_attention_heads, cfg.num_key_value_heads, \
                  cfg.hidden_size // cfg.num_attention_heads

    merge_cfg = generate_consecutive_palu_config(
        start_layer=0, end_layer=cfg.num_hidden_layers - 1,
        group_size=group_size, rank_k=rank_k, rank_v=rank_v,
    )
    max_len = seqlen + 128   # small headroom

    q         = torch.randn(batch_size, nq, 1, nd, device=dev, dtype=dtype)
    cos_sin   = torch.randn(max_len, nd, device=dev, dtype=dtype)

    # ── build + pre-init cache ────────────────────────────────────────────────
    if mode == "fa":
        kv = KV_Cache(cfg, batch_size=batch_size, max_length=max_len, device=str(dev), dtype=dtype)
        _preinit_full(kv, seqlen)
        fn = lambda: _decode_fa(q, kv, layer_idx)

    elif mode == "xkv":
        kv_len   = seqlen
        nkv_spl  = max(1, kv_len // 2048)
        kv_data  = {
            "k_A": torch.randn(batch_size, kv_len, rank_k, device=dev, dtype=dtype),
            "k_B": torch.randn(batch_size, nkv,  rank_k, nd, device=dev, dtype=dtype),
            "v_A": torch.randn(batch_size, kv_len, rank_v, device=dev, dtype=dtype),
            "v_B": torch.randn(batch_size, nkv,  rank_v, nd, device=dev, dtype=dtype),
            "num_kv_splits": torch.full((batch_size,), nkv_spl, dtype=torch.int32, device=dev),
            "max_kv_splits": nkv_spl,
        }
        fn = lambda: _decode_xkv(q, kv_data)

    elif mode == "xkey_sr":
        # Pre-flight: U needs [n_groups, bsz, seqlen, rank_k] on GPU
        n_grp = cfg.num_hidden_layers // group_size
        _gpu_need = n_grp * batch_size * seqlen * rank_k * 2  # bytes (bf16)
        _gpu_free_actual, _ = torch.cuda.mem_get_info(dev)
        _gpu_cache = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        _gpu_free = _gpu_free_actual + _gpu_cache  # actual free + PyTorch reusable cache
        if _gpu_need > _gpu_free * 0.9:
            raise RuntimeError(
                f"Estimated OOM before CPU init: U needs {_gpu_need/1e9:.2f} GB, "
                f"only {_gpu_free/1e9:.2f} GB usable on GPU "
                f"({_gpu_free_actual/1e9:.2f} free + {_gpu_cache/1e9:.2f} cache)"
            )
        kv = ShadowKVCache_xKey_CPU(cfg, merge_cfg, max_length=max_len, device=dev,
                                     dtype=dtype, batch_size=batch_size,
                                     sparse_budget=sparse_budget)
        _preinit_xkey_sr(kv, cfg, seqlen)
        kv.H2D()
        curr_s  = torch.cuda.current_stream()
        val_s   = kv.copy_stream
        fn = lambda: _decode_xkey_sr(q, kv, layer_idx, cos_sin, curr_s, val_s)

    elif mode == "xkv_sr":
        # Pre-flight: U_k + U_v need [n_groups, bsz, seqlen, rank_k/v] on GPU
        n_grp = cfg.num_hidden_layers // group_size
        _gpu_need = n_grp * batch_size * seqlen * (rank_k + rank_v) * 2
        _gpu_free_actual, _ = torch.cuda.mem_get_info(dev)
        _gpu_cache = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        _gpu_free = _gpu_free_actual + _gpu_cache  # actual free + PyTorch reusable cache
        if _gpu_need > _gpu_free * 0.9:
            raise RuntimeError(
                f"Estimated OOM before init: U_k+U_v needs {_gpu_need/1e9:.2f} GB, "
                f"only {_gpu_free/1e9:.2f} GB usable on GPU "
                f"({_gpu_free_actual/1e9:.2f} free + {_gpu_cache/1e9:.2f} cache)"
            )
        kv = ShadowKVCache_xKV(cfg, merge_cfg, max_length=max_len, device=dev,
                                dtype=dtype, batch_size=batch_size,
                                sparse_budget=sparse_budget)
        _preinit_xkv(kv, cfg, seqlen)
        kv.H2D()
        fn = lambda: _decode_xkv_sr(q, kv, layer_idx, cos_sin)

    elif mode == "shadowkv":
        # ShadowKV: per-layer key SVD (group_size=1, rank_k=96), values on CPU
        merge_cfg_sk = generate_consecutive_palu_config(
            start_layer=0, end_layer=cfg.num_hidden_layers - 1,
            group_size=_SHADOWKV_GROUP_SIZE, rank_k=_SHADOWKV_RANK_K, rank_v=rank_v,
        )
        n_grp = cfg.num_hidden_layers // _SHADOWKV_GROUP_SIZE
        _gpu_need = n_grp * batch_size * seqlen * _SHADOWKV_RANK_K * 2
        _gpu_free_actual, _ = torch.cuda.mem_get_info(dev)
        _gpu_cache = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        _gpu_free = _gpu_free_actual + _gpu_cache
        if _gpu_need > _gpu_free * 0.9:
            raise RuntimeError(
                f"Estimated OOM: U needs {_gpu_need/1e9:.2f} GB, "
                f"only {_gpu_free/1e9:.2f} GB usable on GPU"
            )
        kv = ShadowKVCache_xKey_CPU(cfg, merge_cfg_sk, max_length=max_len, device=dev,
                                     dtype=dtype, batch_size=batch_size,
                                     sparse_budget=sparse_budget)
        _preinit_xkey_sr(kv, cfg, seqlen)
        kv.H2D()
        curr_s = torch.cuda.current_stream()
        val_s  = kv.copy_stream
        fn = lambda: _decode_xkey_sr(q, kv, layer_idx, cos_sin, curr_s, val_s)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ── warmup ───────────────────────────────────────────────────────────────
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # ── timed loop ───────────────────────────────────────────────────────────
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    avg_ms = sum(times) / len(times)
    out_rank_k     = _SHADOWKV_RANK_K     if mode == "shadowkv" else rank_k
    out_group_size = _SHADOWKV_GROUP_SIZE if mode == "shadowkv" else group_size
    return {"mode": mode, "seqlen": seqlen, "batch_size": batch_size,
            "rank_k": out_rank_k, "rank_v": rank_v, "group_size": out_group_size,
            "sparse_budget": sparse_budget, "avg_ms": avg_ms}


# ── Full sweep ───────────────────────────────────────────────────────────────

SWEEP_CONFIGS = [
    # seqlen,  batch_sizes
    (60_000,  [8, 16, 32]),
    (122_000, [4, 8, 16]),
]

ALL_MODES = ["fa", "xkv", "xkey_sr", "xkv_sr", "shadowkv"]

# ShadowKV = per-layer key SVD (group_size=1, rank_k=96) + CPU values + sparse attn
_SHADOWKV_GROUP_SIZE = 1
_SHADOWKV_RANK_K = 96


def run_sweep(modes=ALL_MODES, warmup=5, iters=20,
                    rank_k=384, rank_v=576, group_size=4, sparse_budget=2048,
                    output_dir=None) -> list[dict]:
    results = []
    fa_baseline: dict[tuple, float] = {}   # (seqlen, batch) → fa avg_ms

    print(f"\n{'='*70}")
    print(f"  xKV Decode Attention Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  rank_k={rank_k}  rank_v={rank_v}  group_size={group_size}  sparse_budget={sparse_budget}")
    print(f"{'='*70}\n")

    for seqlen, batch_sizes in SWEEP_CONFIGS:
        for bsz in batch_sizes:
            for mode in modes:
                label = f"{mode:8s}  seqlen={seqlen//1000}k  bs={bsz}"
                print(f"  Running {label} ...", end="", flush=True)
                try:
                    r = benchmark_one(mode=mode, seqlen=seqlen, batch_size=bsz,
                                      rank_k=rank_k, rank_v=rank_v,
                                      group_size=group_size, sparse_budget=sparse_budget,
                                      warmup=warmup, iters=iters)
                    if mode == "fa":
                        fa_baseline[(seqlen, bsz)] = r["avg_ms"]
                    fa_t = fa_baseline.get((seqlen, bsz), None)
                    speedup = (fa_t / r["avg_ms"]) if fa_t else None
                    r["speedup_vs_fa"] = speedup
                    results.append(r)
                    sp_str = f"  speedup={speedup:.2f}×" if speedup else ""
                    print(f"  {r['avg_ms']:.3f} ms{sp_str}")
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    results.append({"mode": mode, "seqlen": seqlen, "batch_size": bsz,
                                    "error": str(exc), "speedup_vs_fa": None})

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Mode':<10} {'SeqLen':>8} {'Batch':>6} {'ms':>8} {'Speedup':>10}")
    print(f"{'-'*70}")
    for r in results:
        if "error" in r:
            print(f"{r['mode']:<10} {r['seqlen']//1000:>7}k {r['batch_size']:>6}  {'FAILED':>8}")
            continue
        sp = f"{r['speedup_vs_fa']:.2f}×" if r.get("speedup_vs_fa") else "—"
        print(f"{r['mode']:<10} {r['seqlen']//1000:>7}k {r['batch_size']:>6} {r['avg_ms']:>8.3f} {sp:>10}")
    print(f"{'='*70}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"decode_attn_{ts}.json")
        csv_path  = os.path.join(output_dir, f"decode_attn_{ts}.csv")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mode", "seqlen", "batch_size", "avg_ms",
                                               "speedup_vs_fa", "rank_k", "rank_v",
                                               "group_size", "sparse_budget"])
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})
        print(f"Results saved → {json_path}")
        print(f"             → {csv_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser("xKV decode attention benchmark")
    p.add_argument("--mode",   default="all", help="fa|xkv|xkey_sr|xkv_sr|shadowkv|all")
    p.add_argument("--seqlen", type=int, default=None, help="Single seqlen override")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--rank_k",  type=int, default=384)
    p.add_argument("--rank_v",  type=int, default=576)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--sparse_budget", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters",  type=int, default=20)
    p.add_argument("--output_dir", default="results/efficiency")
    args = p.parse_args()

    modes = ALL_MODES if args.mode == "all" else [args.mode]

    if args.seqlen is not None and args.batch_size is not None:
        r = benchmark_one(mode=modes[0], seqlen=args.seqlen, batch_size=args.batch_size,
                          rank_k=args.rank_k, rank_v=args.rank_v,
                          group_size=args.group_size, sparse_budget=args.sparse_budget,
                          warmup=args.warmup, iters=args.iters)
        print(json.dumps(r, indent=2))
    else:
        run_sweep(modes=modes, warmup=args.warmup, iters=args.iters,
                        rank_k=args.rank_k, rank_v=args.rank_v,
                        group_size=args.group_size, sparse_budget=args.sparse_budget,
                        output_dir=args.output_dir)
