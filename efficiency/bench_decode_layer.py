"""
Full Llama decoder layer latency benchmark.

Wraps each attention mode in a complete single-layer decode forward pass
matching LlamaDecoderLayer.forward():

  input_layernorm  →  Q_proj  →  Attention  →  O_proj  →  residual_add
  →  post_attn_layernorm  →  gate_proj/up_proj  →  silu × up  →  down_proj
  →  residual_add

K/V projections are included in the timing (matching real decode cost where the
new token's K/V are projected even though the KV cache is pre-populated).

Tests the same modes and configs as bench_decode_attn.py.

Usage:
  python bench_decode_layer.py
  python bench_decode_layer.py --seqlen 60000 --batch_size 8 --mode all
"""
import sys, os, argparse, json, csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache

from models.kv_cache import KV_Cache, ShadowKVCache_xKey_CPU, ShadowKVCache_xKV
from models.merge_configs import generate_consecutive_palu_config
from ops.fused_attention.rope.v1 import decode_attention_fwd as xkv_attn

# Import KV-cache pre-init helpers from bench_decode_attn
from bench_decode_attn import (
    _llama8b_config, _preinit_full, _preinit_xkey_sr, _preinit_xkv,
    ALL_MODES, SWEEP_CONFIGS,
    _SHADOWKV_GROUP_SIZE, _SHADOWKV_RANK_K,
)


# ── Llama-3.1-8B layer sizes ──────────────────────────────────────────────────
_HIDDEN_SIZE      = 4096
_INTERMEDIATE     = 14336
_RMS_EPS          = 1e-5
_FULL_LAYERS      = 32


# ── RMSNorm (matches LlamaRMSNorm) ───────────────────────────────────────────

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = _RMS_EPS):
    x = x.float()
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * weight).to(weight.dtype)


# ── Layer weight factory ───────────────────────────────────────────────────────

def _make_weights(cfg, dev, dtype):
    h   = cfg.hidden_size           # 4096
    ff  = _INTERMEDIATE             # 14336
    nq  = cfg.num_attention_heads   # 32
    nkv = cfg.num_key_value_heads   # 8
    nd  = h // nq                   # 128

    return {
        # RMSNorm weights
        "input_norm_w":    torch.ones(h, device=dev, dtype=dtype),
        "post_norm_w":     torch.ones(h, device=dev, dtype=dtype),
        # Attention projections (all four, matching LlamaAttention)
        "q_proj": torch.randn(nq  * nd, h, device=dev, dtype=dtype) * (h ** -0.5),
        "k_proj": torch.randn(nkv * nd, h, device=dev, dtype=dtype) * (h ** -0.5),
        "v_proj": torch.randn(nkv * nd, h, device=dev, dtype=dtype) * (h ** -0.5),
        "o_proj": torch.randn(h, nq * nd, device=dev, dtype=dtype) * (h ** -0.5),
        # MLP (SwiGLU)
        "gate_proj": torch.randn(ff, h, device=dev, dtype=dtype) * (h ** -0.5),
        "up_proj":   torch.randn(ff, h, device=dev, dtype=dtype) * (h ** -0.5),
        "down_proj": torch.randn(h, ff, device=dev, dtype=dtype) * (ff ** -0.5),
    }


# ── Attention dispatch (same logic as bench_decode_attn) ─────────────────────

@torch.inference_mode()
def _attn_fa(q, kv, layer_idx, **_):
    seqlen = kv.kv_offset
    k = kv.k_cache[layer_idx][:, :, :seqlen]
    v = kv.v_cache[layer_idx][:, :, :seqlen]
    # returns [bsz, 1, nq, nd]
    return flash_attn_with_kvcache(
        q=q.transpose(1, 2), k_cache=k.transpose(1, 2), v_cache=v.transpose(1, 2),
        causal=True,
    )


@torch.inference_mode()
def _attn_xkv(q, kv_data, **_):
    k_A = kv_data["k_A"]; k_B = kv_data["k_B"]
    v_A = kv_data["v_A"]; v_B = kv_data["v_B"]
    q_in = q[:, :, 0, :]   # [bsz, nq, nd]
    out = xkv_attn(q_in, k_A, k_B, v_A, v_B,
                        kv_data["num_kv_splits"], kv_data["max_kv_splits"])
    return out.unsqueeze(1)  # [bsz, 1, nq, nd]


@torch.inference_mode()
def _attn_xkey_sr(q, kv, layer_idx, cos_sin, curr_s, val_s, **_):
    pos_ids = kv.get_retrieval_position_ids(layer_idx=layer_idx, query_states=q)
    with torch.cuda.stream(val_s):
        val_s.wait_stream(curr_s)
        v = kv.get_value_cache(layer_idx, pos_ids)
    k = kv.get_key_cache(layer_idx=layer_idx, position_ids=pos_ids,
                         cos_sin_cache=cos_sin)
    curr_s.wait_stream(val_s)
    return flash_attn_with_kvcache(
        q=q.transpose(1, 2), k_cache=k.transpose(1, 2), v_cache=v.transpose(1, 2),
        causal=True,
    )


@torch.inference_mode()
def _attn_xkv_sr(q, kv, layer_idx, cos_sin, **_):
    pos_ids = kv.get_retrieval_position_ids(layer_idx=layer_idx, query_states=q)
    v = kv.get_value_cache(layer_idx, pos_ids, cos_sin)
    k = kv.get_key_cache(layer_idx=layer_idx, position_ids=pos_ids,
                         cos_sin_cache=cos_sin)
    return flash_attn_with_kvcache(
        q=q.transpose(1, 2), k_cache=k.transpose(1, 2), v_cache=v.transpose(1, 2),
        causal=True,
    )


# ── Full decoder layer forward ────────────────────────────────────────────────

@torch.inference_mode()
def _layer_forward(hidden_states: torch.Tensor, weights: dict,
                   cfg, attn_fn, cos_sin: torch.Tensor) -> torch.Tensor:
    """
    Full Llama decoder layer forward for a single decode token.
    hidden_states: [bsz, 1, hidden_size]

    Matches LlamaDecoderLayer.forward() + LlamaAttention.forward():
      input_layernorm
      → Q/K/V proj (single decode token)
      → RoPE on Q and K (position = 0, placeholder for timing)
      → attn_fn (mode-specific KV cache lookup)
      → O_proj → residual
      → post_attn_layernorm → MLP (SwiGLU) → residual
    """
    nq  = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    nd  = cfg.hidden_size // nq
    bsz = hidden_states.shape[0]

    # 1. Input RMSNorm
    residual = hidden_states
    x = _rms_norm(hidden_states, weights["input_norm_w"])

    # 2. Q / K / V projections for the current decode token
    q = F.linear(x, weights["q_proj"]).view(bsz, 1, nq,  nd).transpose(1, 2)  # [bsz, nq,  1, nd]
    k = F.linear(x, weights["k_proj"]).view(bsz, 1, nkv, nd).transpose(1, 2)  # [bsz, nkv, 1, nd]
    v = F.linear(x, weights["v_proj"]).view(bsz, 1, nkv, nd).transpose(1, 2)  # [bsz, nkv, 1, nd]

    # 3. RoPE on Q and K for the current position (use position 0 as placeholder)
    #    cos/sin shape: [max_len, nd]; we take position 0 → [nd]
    cos = cos_sin[0, :nd//2]   # half-dim real part
    sin = cos_sin[0, nd//2:]   # half-dim imaginary part
    def _rope(t):               # t: [bsz, heads, 1, nd]
        t1, t2 = t[..., :nd//2], t[..., nd//2:]
        return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)
    q = _rope(q)
    k = _rope(k)

    # 4. Attention (mode-specific) → [bsz, 1, nq, nd]
    attn_out = attn_fn(q)

    # 5. O projection  [bsz, 1, nq*nd] → [bsz, 1, hidden]
    attn_out = attn_out.reshape(bsz, 1, -1)
    attn_out = F.linear(attn_out, weights["o_proj"])

    # 6. Residual
    hidden_states = residual + attn_out

    # 7. Post-attention RMSNorm
    residual = hidden_states
    x = _rms_norm(hidden_states, weights["post_norm_w"])

    # 8. SwiGLU MLP: down(silu(gate) * up)
    gate = F.silu(F.linear(x, weights["gate_proj"]))
    up   = F.linear(x, weights["up_proj"])
    hidden_states = F.linear(gate * up, weights["down_proj"])

    # 9. Residual
    hidden_states = residual + hidden_states

    return hidden_states


# ── Single benchmark run ──────────────────────────────────────────────────────

@torch.inference_mode()
def benchmark_layer_one(mode: str, seqlen: int, batch_size: int,
                        rank_k: int, rank_v: int, group_size: int, sparse_budget: int,
                        warmup: int, iters: int, layer_idx: int = 0,
                        num_layers: int = 1) -> dict:

    cfg   = _llama8b_config(num_layers=4)
    dev   = torch.device("cuda:0")
    dtype = torch.bfloat16
    nq, nkv, nd = cfg.num_attention_heads, cfg.num_key_value_heads, \
                  cfg.hidden_size // cfg.num_attention_heads

    merge_cfg = generate_consecutive_palu_config(
        start_layer=0, end_layer=cfg.num_hidden_layers - 1,
        group_size=group_size, rank_k=rank_k, rank_v=rank_v,
    )
    max_len = seqlen + 128

    weights     = _make_weights(cfg, dev, dtype)
    hidden_in   = torch.randn(batch_size, 1, cfg.hidden_size, device=dev, dtype=dtype)
    cos_sin     = torch.randn(max_len, nd, device=dev, dtype=dtype)

    # ── Build + pre-init cache; bind attn_fn ────────────────────────────────
    if mode == "fa":
        kv = KV_Cache(cfg, batch_size=batch_size, max_length=max_len,
                      device=str(dev), dtype=dtype)
        _preinit_full(kv, seqlen)
        attn_fn = lambda q: _attn_fa(q, kv, layer_idx)

    elif mode == "xkv":
        kv_len  = seqlen
        nkv_spl = max(1, kv_len // 2048)
        kv_data = {
            "k_A": torch.randn(batch_size, kv_len, rank_k, device=dev, dtype=dtype),
            "k_B": torch.randn(batch_size, nkv,  rank_k, nd, device=dev, dtype=dtype),
            "v_A": torch.randn(batch_size, kv_len, rank_v, device=dev, dtype=dtype),
            "v_B": torch.randn(batch_size, nkv,  rank_v, nd, device=dev, dtype=dtype),
            "num_kv_splits": torch.full((batch_size,), nkv_spl, dtype=torch.int32, device=dev),
            "max_kv_splits": nkv_spl,
        }
        attn_fn = lambda q: _attn_xkv(q, kv_data)

    elif mode in ("xkey_sr", "shadowkv"):
        _gs   = _SHADOWKV_GROUP_SIZE if mode == "shadowkv" else group_size
        _rk   = _SHADOWKV_RANK_K     if mode == "shadowkv" else rank_k
        _mcfg = generate_consecutive_palu_config(
            start_layer=0, end_layer=cfg.num_hidden_layers - 1,
            group_size=_gs, rank_k=_rk, rank_v=rank_v,
        )
        n_grp = cfg.num_hidden_layers // _gs
        _gpu_need = n_grp * batch_size * seqlen * _rk * 2
        _free_a, _ = torch.cuda.mem_get_info(dev)
        _free = _free_a + torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        if _gpu_need > _free * 0.9:
            raise RuntimeError(f"Estimated OOM: U needs {_gpu_need/1e9:.2f} GB")
        kv = ShadowKVCache_xKey_CPU(_llama8b_config(4), _mcfg, max_length=max_len,
                                    device=dev, dtype=dtype, batch_size=batch_size,
                                    sparse_budget=sparse_budget)
        _preinit_xkey_sr(kv, cfg, seqlen)
        kv.H2D()
        curr_s = torch.cuda.current_stream()
        val_s  = kv.copy_stream
        attn_fn = lambda q: _attn_xkey_sr(q, kv, layer_idx, cos_sin, curr_s, val_s)

    elif mode == "xkv_sr":
        n_grp = cfg.num_hidden_layers // group_size
        _gpu_need = n_grp * batch_size * seqlen * (rank_k + rank_v) * 2
        _free_a, _ = torch.cuda.mem_get_info(dev)
        _free = _free_a + torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        if _gpu_need > _free * 0.9:
            raise RuntimeError(f"Estimated OOM: U_k+U_v needs {_gpu_need/1e9:.2f} GB")
        kv = ShadowKVCache_xKV(cfg, merge_cfg, max_length=max_len, device=dev,
                               dtype=dtype, batch_size=batch_size,
                               sparse_budget=sparse_budget)
        _preinit_xkv(kv, cfg, seqlen)
        kv.H2D()
        attn_fn = lambda q: _attn_xkv_sr(q, kv, layer_idx, cos_sin)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    def fn():
        h = hidden_in
        for _ in range(num_layers):
            h = _layer_forward(h, weights, cfg, attn_fn, cos_sin)
        return h

    # ── Warmup ───────────────────────────────────────────────────────────────
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # ── Timed loop ───────────────────────────────────────────────────────────
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    avg_ms       = sum(times) / len(times)
    avg_ms_layer = avg_ms / num_layers
    out_rk = _SHADOWKV_RANK_K     if mode == "shadowkv" else rank_k
    out_gs = _SHADOWKV_GROUP_SIZE if mode == "shadowkv" else group_size
    return {"mode": mode, "seqlen": seqlen, "batch_size": batch_size,
            "rank_k": out_rk, "rank_v": rank_v, "group_size": out_gs,
            "sparse_budget": sparse_budget, "num_layers": num_layers,
            "avg_ms": avg_ms, "avg_ms_per_layer": avg_ms_layer}


# ── Layer-count scaling sweep ─────────────────────────────────────────────────

SCALING_LAYER_COUNTS = [1, 2, 4, 8]

def run_scaling_sweep(modes=ALL_MODES, seqlen=60_000, batch_size=8,
                      rank_k=384, rank_v=576, group_size=4, sparse_budget=2048,
                      warmup=3, iters=10, output_dir=None) -> list[dict]:
    """
    For each mode, run 1/2/4/8-layer loops and check linearity.
    Reports per-layer ms and estimated full-model (32-layer) latency.
    """
    results = []

    print(f"\n{'='*70}")
    print(f"  xKV Decoder Layer Scaling Test")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  seqlen={seqlen//1000}k  batch_size={batch_size}")
    print(f"  rank_k={rank_k}  rank_v={rank_v}  group_size={group_size}")
    print(f"  hidden={_HIDDEN_SIZE}  intermediate={_INTERMEDIATE}")
    print(f"{'='*70}\n")

    for mode in modes:
        print(f"  Mode: {mode}")
        for nl in SCALING_LAYER_COUNTS:
            print(f"    {nl} layer{'s' if nl > 1 else ''} ...", end="", flush=True)
            try:
                r = benchmark_layer_one(
                    mode=mode, seqlen=seqlen, batch_size=batch_size,
                    rank_k=rank_k, rank_v=rank_v,
                    group_size=group_size, sparse_budget=sparse_budget,
                    warmup=warmup, iters=iters, num_layers=nl)
                est32 = r["avg_ms_per_layer"] * _FULL_LAYERS
                r["est_32layer_ms"] = est32
                results.append(r)
                print(f"  total={r['avg_ms']:.2f}ms  per-layer={r['avg_ms_per_layer']:.2f}ms"
                      f"  est-32L={est32:.1f}ms")
            except Exception as exc:
                print(f"  FAILED: {exc}")
                results.append({"mode": mode, "num_layers": nl, "error": str(exc)})
        print()

    # ── Linearity check ──────────────────────────────────────────────────────
    print(f"{'='*70}")
    print(f"  Linearity check (per-layer ms, should be constant across #layers)")
    print(f"  {'Mode':<10} {'1L':>8} {'2L':>8} {'4L':>8} {'8L':>8} {'est 32L':>10}")
    print(f"  {'-'*60}")
    by_mode: dict[str, dict[int, float]] = {}
    for r in results:
        if "error" not in r:
            by_mode.setdefault(r["mode"], {})[r["num_layers"]] = r["avg_ms_per_layer"]
    for mode in modes:
        d = by_mode.get(mode, {})
        vals = [f"{d[nl]:.2f}" if nl in d else "—" for nl in SCALING_LAYER_COUNTS]
        est32 = f"{d.get(8, d.get(4, d.get(2, d.get(1, 0)))) * _FULL_LAYERS:.1f}" if d else "—"
        print(f"  {mode:<10} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {est32:>10}")
    print(f"{'='*70}\n")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"layer_scaling_{ts}.json")
        csv_path  = os.path.join(output_dir, f"layer_scaling_{ts}.csv")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mode", "seqlen", "batch_size", "num_layers",
                                               "avg_ms", "avg_ms_per_layer", "est_32layer_ms",
                                               "rank_k", "rank_v", "group_size", "sparse_budget"])
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})
        print(f"Results saved → {json_path}")
        print(f"             → {csv_path}")

    return results


# ── Full sweep ───────────────────────────────────────────────────────────────

def run_layer_sweep(modes=ALL_MODES, warmup=5, iters=20,
                    rank_k=384, rank_v=576, group_size=4, sparse_budget=2048,
                    num_layers=1, output_dir=None) -> list[dict]:
    results = []
    fa_baseline: dict[tuple, float] = {}

    print(f"\n{'='*70}")
    print(f"  xKV Decoder Layer Benchmark  (attention + MLP, {num_layers}-layer loop)")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  rank_k={rank_k}  rank_v={rank_v}  group_size={group_size}  sparse_budget={sparse_budget}")
    print(f"  hidden={_HIDDEN_SIZE}  intermediate={_INTERMEDIATE}")
    if num_layers > 1:
        print(f"  Projecting per-layer average to {_FULL_LAYERS} layers (est. full model)")
    print(f"{'='*70}\n")

    for seqlen, batch_sizes in SWEEP_CONFIGS:
        for bsz in batch_sizes:
            for mode in modes:
                label = f"{mode:8s}  seqlen={seqlen//1000}k  bs={bsz}"
                print(f"  Running {label} ...", end="", flush=True)
                try:
                    r = benchmark_layer_one(
                        mode=mode, seqlen=seqlen, batch_size=bsz,
                        rank_k=rank_k, rank_v=rank_v,
                        group_size=group_size, sparse_budget=sparse_budget,
                        warmup=warmup, iters=iters, num_layers=num_layers)
                    per_layer = r["avg_ms_per_layer"]
                    if mode == "fa":
                        fa_baseline[(seqlen, bsz)] = per_layer
                    fa_t = fa_baseline.get((seqlen, bsz))
                    speedup = (fa_t / per_layer) if fa_t else None
                    est32 = per_layer * _FULL_LAYERS
                    toks  = (bsz * 1000.0) / est32   # tokens/s estimate
                    r["speedup_vs_fa"] = speedup
                    r["est_32layer_ms"] = est32
                    r["est_throughput_toks_per_s"] = toks
                    results.append(r)
                    sp_str = f"  speedup={speedup:.2f}×" if speedup else ""
                    print(f"  {per_layer:.3f} ms/layer  est32={est32:.0f}ms  {toks:.0f}tok/s{sp_str}")
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    results.append({"mode": mode, "seqlen": seqlen, "batch_size": bsz,
                                    "error": str(exc), "speedup_vs_fa": None})

    print(f"\n{'='*78}")
    print(f"{'Mode':<10} {'SeqLen':>8} {'Batch':>6} {'ms/layer':>10} {'est32L ms':>10} {'tok/s':>8} {'Speedup':>10}")
    print(f"{'-'*78}")
    for r in results:
        if "error" in r:
            print(f"{'ERROR':<10} {r['seqlen']//1000:>7}k {r['batch_size']:>6}  {'FAILED':>10}")
            continue
        sp  = f"{r['speedup_vs_fa']:.2f}×" if r.get("speedup_vs_fa") else "—"
        e32 = f"{r.get('est_32layer_ms', 0):.0f}"
        tps = f"{r.get('est_throughput_toks_per_s', 0):.0f}"
        print(f"{r['mode']:<10} {r['seqlen']//1000:>7}k {r['batch_size']:>6}"
              f" {r.get('avg_ms_per_layer', r['avg_ms']):>10.3f} {e32:>10} {tps:>8} {sp:>10}")
    print(f"{'='*78}\n")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"decode_layer_{ts}.json")
        csv_path  = os.path.join(output_dir, f"decode_layer_{ts}.csv")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mode", "seqlen", "batch_size", "num_layers",
                                               "avg_ms", "avg_ms_per_layer", "est_32layer_ms",
                                               "est_throughput_toks_per_s",
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
    p = argparse.ArgumentParser("xKV full decoder layer benchmark")
    p.add_argument("--mode",   default="all", help="fa|xkv|xkey_sr|xkv_sr|shadowkv|all")
    p.add_argument("--seqlen", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--rank_k",  type=int, default=384)
    p.add_argument("--rank_v",  type=int, default=576)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--sparse_budget", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters",  type=int, default=20)
    p.add_argument("--num_layers", type=int, default=1,
                   help="Number of decoder layers to loop (for scaling tests)")
    p.add_argument("--scaling", action="store_true",
                   help="Run scaling sweep over num_layers=[1,2,4,8]")
    p.add_argument("--output_dir", default="results/efficiency")
    args = p.parse_args()

    modes = ALL_MODES if args.mode == "all" else [args.mode]

    if args.scaling:
        sl = args.seqlen or 60_000
        bs = args.batch_size or 8
        run_scaling_sweep(modes=modes, seqlen=sl, batch_size=bs,
                          rank_k=args.rank_k, rank_v=args.rank_v,
                          group_size=args.group_size, sparse_budget=args.sparse_budget,
                          warmup=args.warmup, iters=args.iters,
                          output_dir=args.output_dir)
    elif args.seqlen is not None and args.batch_size is not None:
        r = benchmark_layer_one(
            mode=modes[0], seqlen=args.seqlen, batch_size=args.batch_size,
            rank_k=args.rank_k, rank_v=args.rank_v,
            group_size=args.group_size, sparse_budget=args.sparse_budget,
            warmup=args.warmup, iters=args.iters, num_layers=args.num_layers)
        print(json.dumps(r, indent=2))
    else:
        run_layer_sweep(modes=modes, warmup=args.warmup, iters=args.iters,
                        rank_k=args.rank_k, rank_v=args.rank_v,
                        group_size=args.group_size, sparse_budget=args.sparse_budget,
                        num_layers=args.num_layers, output_dir=args.output_dir)
