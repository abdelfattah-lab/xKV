#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import AutoConfig
"""
Quick KV-cache size and compression calculator.

Models are specified via --model as a Hugging Face model ID:
- layers, hidden_size, num_heads, num_kv_heads are read from HF config

Notation:
- B: batch size
- Layers: number of layers
- kv_head: number of KV heads
- L: sequence length
- D: head dimension (hidden_size / num_heads)
- rank_k, rank_v: low-rank for K and V
- G: cross-layer group size (layers per group)

Baseline elements (both K and V stored in full):
    elems_base = 2 * B * layers * kv_head * L * D

Cross-layer grouped low-rank (U shared across heads & layers within a group):
    num_groups = layers / G
    elems_lr  = B * num_groups * L * (rank_k + rank_v)       # shared U
              + B * layers * kv_head * (rank_k + rank_v) * D # separate SV^T

By default, r is scaled by G (r*=G) unless --no-scale-rank is set.

Special case:
- If rank_v = 0, V-Cache is uncompressed. Only K-Cache uses low-rank.

Example usage:
    python3 scripts/calc_kv_compression.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --seqlen 65536 --bits 16 --rank_k 64 --rank_v 96 --group-sizes 1,2,4

    python3 scripts/calc_kv_compression.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --seqlen 65536 --bits 16 --rank_k 96 --rank_v 144 --group-sizes 1,2,4

    python3 scripts/calc_kv_compression.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --seqlen 65536 --bits 16 --rank_k 128 --rank_v 192 --group-sizes 1,2,4

    python3 scripts/calc_kv_compression.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --seqlen 65536 --bits 16 --rank_k 64 --rank_v 0 --group-sizes 1,2,4
"""


@dataclass
class KVShape:
    batch: int
    layers: int
    kv_heads: int
    seqlen: int
    head_dim: int


def baseline_kv_elements(shape: KVShape) -> int:
    # Stores both K and V: 2 * B * Layers * kv_head * L * D
    return 2 * shape.batch * shape.layers * shape.kv_heads * shape.seqlen * shape.head_dim


def cross_layer_lowrank_elements(shape: KVShape, rank_k: int, rank_v: int, group_size: int) -> int:
    """
    Cross-layer grouped SVD where U is shared across heads and layers within each group ("shared U").
    Layer-wise coefficient matrices are separate SV^T with shape [kv_heads * D, r].

    Elements per group:
    Shared U terms: B * L * (rank_k + rank_v)
    Separate SV^T terms: G * B * kv_heads * D * (rank_k + rank_v)

    If rank_v = 0: V is uncompressed. In that case,
    - Shared U and Separate SV^T terms use only rank_k
    - Add full V cost: B * layers * kv_heads * L * D

    Total across groups (num_groups = layers / G):
        num_groups * U_terms + layers * B * kv_heads * D * (rank_k + rank_v)
    """
    import math

    G = max(1, int(group_size))
    num_groups = math.ceil(shape.layers / G)
    u_terms = shape.batch * shape.seqlen * (rank_k + rank_v)
    separate_svt_terms = shape.batch * shape.kv_heads * shape.head_dim * (rank_k + rank_v)

    # rank_v=0 means V is stored full (uncompressed). Only K contributes to low-rank terms.
    v_full = (shape.batch * shape.layers * shape.kv_heads * shape.seqlen * shape.head_dim) if rank_v == 0 else 0

    return num_groups * u_terms + shape.layers * separate_svt_terms + v_full


def human_readable_size(num_bytes: float) -> str:
    # Prefer power-of-two units (KiB, MiB, GiB, TiB)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    i = 0
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {units[i]}"


def _resolve_spec_from_hf(model: str) -> Dict[str, Any]:
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)

    def pick(*names: str, default: Optional[int] = None) -> Optional[int]:
        for n in names:
            v = getattr(cfg, n, None)
            if v is not None:
                return int(v)
        return default

    layers = pick("num_hidden_layers", "n_layer", default=0)
    hidden = pick("hidden_size", "n_embd", default=0)
    heads = pick("num_attention_heads", "n_head", default=0)
    kv_heads = pick("num_key_value_heads", "n_kv_head", default=heads)
    if not all([layers, hidden, heads, kv_heads]):
        raise RuntimeError(f"Missing fields in HF config for '{model}': layers={layers}, hidden={hidden}, heads={heads}, kv_heads={kv_heads}")
    return {"layers": layers, "hidden_size": hidden, "num_heads": heads, "num_kv_heads": kv_heads, "source": "hf"}


def compute_and_print(
    *,
    model: str,
    seqlen: int = 65536,
    batch: int = 1,
    bits: int = 16,
    rank_k: int = 64,
    rank_v: int = 64,
    group_sizes: list[int] | None = None,
    scale_rank_with_gs: bool = True,
):
    # Always resolve config from HF (no local registry)
    try:
        spec = _resolve_spec_from_hf(model)
    except Exception as e:
        raise SystemExit(f"Failed to resolve config from HF for '{model}': {e}")
    layers = spec["layers"]
    hidden_size = spec["hidden_size"]
    num_heads = spec["num_heads"]
    num_kv_heads = spec["num_kv_heads"]

    head_dim = hidden_size // num_heads
    shape = KVShape(batch=batch, layers=layers, kv_heads=num_kv_heads, seqlen=seqlen, head_dim=head_dim)
    # Convert bits to bytes per element (may be fractional if not divisible by 8)
    bpe = bits / 8.0

    base_elems = baseline_kv_elements(shape)
    base_bytes = base_elems * bpe
    print("=== KV-Cache Compression Calculator ===")
    print(f"Model     : {model} ({spec.get('source','hf')})")
    print(f"Layers    : {layers}")
    print(f"Heads     : total={num_heads}, kv={num_kv_heads}, head_dim={head_dim}")
    print(f"Batch     : {batch}")
    print(f"SeqLen    : {seqlen}")
    print(f"Bits      : {bits}")
    print(f"Ranks     : rank_k={rank_k}, rank_v={rank_v}")
    print()
    print(f"Full KV-Cache Size :  {human_readable_size(base_bytes)}")

    # Cross-layer mode: iterate over group sizes (default: [1,2,4])
    gs_list = group_sizes or [1, 2, 4]
    print("\nCross-layer grouped SVD (U shared across heads & layers in group):")
    for gs in gs_list:
        eff_rank_k = rank_k * (gs if scale_rank_with_gs else 1)
        # Do not scale rv when 0 (V full)
        eff_rank_v = rank_v if rank_v == 0 else rank_v * (gs if scale_rank_with_gs else 1)
        low_elems = cross_layer_lowrank_elements(shape, rank_k=eff_rank_k, rank_v=eff_rank_v, group_size=gs)
        low_bytes = low_elems * bpe
        ratio = base_bytes / max(low_bytes, 1)
        savings = 1.0 - (low_bytes / base_bytes)
        tag = " (V full)" if eff_rank_v == 0 else ""
        print(f"  gs={gs:<2} (rank_k={eff_rank_k}, rank_v={eff_rank_v}){tag} -> {human_readable_size(low_bytes):>10} | ratio={ratio:>5.2f}× | save={savings*100:>5.2f}%")


def main():
    parser = argparse.ArgumentParser(description="KV-cache compression calculator")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HF model ID to read config. E.g.: meta-llama/Meta-Llama-3-8B-Instruct, Qwen/Qwen2.5-7B-Instruct-1M",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size B")
    parser.add_argument("--seqlen", type=int, default=65536, help="Sequence length L")
    parser.add_argument("--bits", type=int, default=16, help="Bits per element for storage (e.g., 16 for fp16/bf16, 8 for fp8/int8, 32 for fp32)")
    parser.add_argument("--rank_k", type=int, default=64, help="Rank for K factorization")
    parser.add_argument("--rank_v", type=int, default=64, help="Rank for V factorization")
    parser.add_argument("--group-sizes", type=str, default="1,2,4", help="Comma-separated group sizes to evaluate in cross-layer mode")
    parser.add_argument("--no-scale-rank", action="store_true", help="Do not scale ranks with group size in cross-layer mode (default scales: r*=gs)")
    # no revision arg; use model default config
    args = parser.parse_args()

    gs_list = [int(x) for x in args.group_sizes.split(",") if x.strip()]

    compute_and_print(
        model=args.model,
        batch=args.batch,
        seqlen=args.seqlen,
        bits=args.bits,
        rank_k=args.rank_k,
        rank_v=args.rank_v,
        group_sizes=gs_list,
        scale_rank_with_gs=not args.no_scale_rank,
    )


if __name__ == "__main__":
    main()
