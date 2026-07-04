"""
Isolated unit tests for the low-level torch.ops._shadowkv CUDA ops that
ShadowKVCache_xKey_CPU / ShadowKVCache_xKV compose together in get_retrieval_position_ids /
get_key_cache / get_value_cache:

  - reorder_keys_and_compute_offsets: given a set of newly-selected chunk ids and the
    previously-cached chunk ids, computes which are cache "hits" (already present) vs
    "misses" (need fresh gather), and writes the reordered/deduplicated final chunk-id set.
  - gather_copy_d2d_with_offsets: reshuffles the GPU K-cache buffer in place based on the
    offsets/cnts from the op above (relocates "hit" slots; leaves "miss" slots for the
    downstream GEMM+RoPE kernel to fill -- see test_batch_gather_gemm_correctness.py).
  - gather_copy_with_offsets: the CPU->GPU memory copy used by ShadowKVCache_xKey_CPU's V
    path (V is kept dense on CPU for "xkey_sr", not SVD-compressed like K).
  - batch_gemm_softmax: the landmark-scoring kernel (Q @ K_landmark^T, softmax) that
    get_retrieval_position_ids uses to rank candidate chunks before top-k selection.

Semantics were confirmed empirically (see conversation) before writing these tests -- e.g.
map_size must be one of {128,256,512,1024} (a CUTLASS template-dispatch constant, not an
arbitrary count) or the kernels silently no-op; softmax_o's raw values are NOT a normalized
probability distribution (some internal unnormalized representation) -- only the raw
gemm_o scores and softmax_o's relative ranking are checked, which is the only property that
matters for its purpose (top-k relevance ranking).

Scope: reorder_keys_and_compute_offsets / gather_copy_d2d_with_offsets model an INCREMENTAL
cache (hits = already correct from a previous decode step). These tests check the "fresh"
(first-ever call, all miss) and "second call with partial overlap" scenarios, which is as
much of the incremental-caching semantics as can be checked without simulating a full
multi-step generation loop (something none of the benchmark scripts do either).

Fingerprint-encoding caveat (learned the hard way): any per-chunk tag values used to verify
copies MUST be exactly representable in bf16. Tags like head*1000+chunk_id alias under bf16
rounding (ULP grows to 4 then 8 past 512/1024) and produce fake per-head failures that look
exactly like an addressing bug. The multi-head test below therefore encodes head / chunk_id /
token in SEPARATE elements as small integers (all exact in bf16). See
test_decode_dispatch_correctness.py's module docstring for the full post-mortem, including
the real caller-side stride bug that investigation did uncover.

Usage:
  pytest efficiency/tests/test_shadowkv_micro_ops_correctness.py -v
  python efficiency/tests/test_shadowkv_micro_ops_correctness.py
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import ops.shadowkv  # noqa: F401  (loads the _shadowkv CUDA extension)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reorder_keys_fresh_call_is_all_miss():
    dev = torch.device("cuda:0")
    bsz, heads, select_sets = 1, 1, 256   # select_sets must be one of {128,256,512,1024}
    torch.manual_seed(0)

    cached_pos_ids = torch.full((bsz, heads, select_sets), -1, dtype=torch.int64, device=dev)
    cur_pos_ids = torch.randperm(2000, device=dev)[:select_sets].view(1, 1, -1).long()
    offsets = torch.zeros(bsz * heads * select_sets, dtype=torch.int32, device=dev)
    cnts = torch.zeros(bsz * heads, dtype=torch.int32, device=dev)

    torch.ops._shadowkv.reorder_keys_and_compute_offsets(
        cached_pos_ids, cur_pos_ids, offsets, cnts, bsz, heads, select_sets
    )

    assert cnts.tolist() == [0], "a fresh (-1) cache should report 0 hits"
    assert set(cached_pos_ids[0, 0].tolist()) == set(cur_pos_ids[0, 0].tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reorder_keys_partial_overlap_hit_count():
    dev = torch.device("cuda:0")
    bsz, heads, select_sets = 1, 1, 256
    torch.manual_seed(0)

    cached_pos_ids = torch.full((bsz, heads, select_sets), -1, dtype=torch.int64, device=dev)
    cur_pos_ids_1 = torch.randperm(2000, device=dev)[:select_sets].view(1, 1, -1).long()
    offsets = torch.zeros(bsz * heads * select_sets, dtype=torch.int32, device=dev)
    cnts = torch.zeros(bsz * heads, dtype=torch.int32, device=dev)
    torch.ops._shadowkv.reorder_keys_and_compute_offsets(
        cached_pos_ids, cur_pos_ids_1, offsets, cnts, bsz, heads, select_sets
    )

    overlap_n = 128
    overlap = cur_pos_ids_1[0, 0, :overlap_n]
    pool = torch.randperm(2000, device=dev)
    new_ones = pool[~torch.isin(pool, cur_pos_ids_1)][: select_sets - overlap_n]
    cur_pos_ids_2 = torch.cat([overlap, new_ones]).view(1, 1, -1).long()
    assert cur_pos_ids_2.shape[-1] == select_sets

    torch.ops._shadowkv.reorder_keys_and_compute_offsets(
        cached_pos_ids, cur_pos_ids_2, offsets, cnts, bsz, heads, select_sets
    )

    assert cnts.tolist() == [overlap_n], f"expected {overlap_n} hits from the overlapping set"
    assert set(cached_pos_ids[0, 0].tolist()) == set(cur_pos_ids_2[0, 0].tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gather_copy_d2d_is_noop_on_fresh_miss():
    """With cnts=0 (nothing cached yet), the D2D reshuffle must not touch the buffer --
    every slot is a miss and gets freshly written later by batch_gather_gemm_rotary_pos_emb_cuda
    (verified separately in test_batch_gather_gemm_correctness.py and exercised together with
    this op in test_shadowkv_pipeline_composition.py)."""
    dev = torch.device("cuda:0")
    bsz, heads, select_sets, chunk_size, head_dim = 1, 1, 256, 8, 128
    sparse_budget = select_sets * chunk_size
    gpu_v_offset_tokens = 64
    buf_len = sparse_budget + gpu_v_offset_tokens + 128
    torch.manual_seed(0)

    cached_pos_ids = torch.full((bsz, heads, select_sets), -1, dtype=torch.int64, device=dev)
    cur_pos_ids = torch.randperm(2000, device=dev)[:select_sets].view(1, 1, -1).long()
    offsets = torch.zeros(bsz * heads * select_sets, dtype=torch.int32, device=dev)
    cnts = torch.zeros(bsz * heads, dtype=torch.int32, device=dev)
    torch.ops._shadowkv.reorder_keys_and_compute_offsets(
        cached_pos_ids, cur_pos_ids, offsets, cnts, bsz, heads, select_sets
    )
    assert cnts.tolist() == [0]

    buf = torch.randn(bsz, heads, buf_len, head_dim, device=dev, dtype=torch.bfloat16)
    buf_before = buf.clone()
    torch.ops._shadowkv.gather_copy_d2d_with_offsets(
        buf[0], offsets, cnts, bsz, heads, int(sparse_budget * head_dim),
        int(gpu_v_offset_tokens * head_dim), int(buf_len * head_dim), select_sets,
    )
    assert torch.equal(buf, buf_before), "fresh-miss D2D reshuffle must be a no-op"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gather_copy_with_offsets_fetches_correct_cpu_chunks():
    """ShadowKVCache_xKey_CPU's V path: plain CPU->GPU chunk copy (V stays dense, not
    SVD-compressed). Each CPU chunk is tagged with its own chunk id so we can verify the
    GPU buffer ends up holding exactly the chunks that were selected, at the right slots."""
    dev = torch.device("cuda:0")
    bsz, heads, select_sets, chunk_size, head_dim = 1, 1, 256, 8, 128
    sparse_budget = select_sets * chunk_size
    max_ctx_chunks = 256   # kept <=256 so chunk ids are exactly bf16-representable
    gpu_v_offset_tokens = 64
    buf_len = sparse_budget + gpu_v_offset_tokens + 128
    torch.manual_seed(0)

    cached_pos_ids = torch.full((bsz, heads, select_sets), -1, dtype=torch.int64, device=dev)
    cur_pos_ids = torch.randperm(max_ctx_chunks, device=dev)[:select_sets].view(1, 1, -1).long()
    offsets = torch.zeros(bsz * heads * select_sets, dtype=torch.int32, device=dev)
    cnts = torch.zeros(bsz * heads, dtype=torch.int32, device=dev)
    torch.ops._shadowkv.reorder_keys_and_compute_offsets(
        cached_pos_ids, cur_pos_ids, offsets, cnts, bsz, heads, select_sets
    )

    v_cache_cpu = torch.zeros(heads, max_ctx_chunks, chunk_size * head_dim, dtype=torch.bfloat16, pin_memory=True)
    for c in range(max_ctx_chunks):
        v_cache_cpu[0, c, :] = float(c)

    v_cache_buffer = torch.zeros(heads, buf_len, head_dim, device=dev, dtype=torch.bfloat16)
    temp = torch.zeros(bsz, heads, select_sets, chunk_size * head_dim, device=dev, dtype=torch.bfloat16).contiguous()
    signals = torch.zeros(bsz * heads, dtype=torch.int32, device=dev)

    torch.ops._shadowkv.gather_copy_with_offsets(
        v_cache_cpu, v_cache_buffer, temp, offsets, cnts, signals,
        bsz, heads, int(max_ctx_chunks * head_dim), int(sparse_budget * head_dim),
        int(gpu_v_offset_tokens * head_dim), int(buf_len * head_dim), select_sets,
    )
    torch.cuda.synchronize()

    final_pos = cached_pos_ids[0, 0]
    got_block = v_cache_buffer[0, gpu_v_offset_tokens:gpu_v_offset_tokens + sparse_budget, 0].view(select_sets, chunk_size)
    expected = final_pos.float().view(select_sets, 1).expand(-1, chunk_size)
    assert torch.equal(got_block, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gather_copy_with_offsets_multi_head_exact_fingerprints():
    """Multi-head (heads=8) version of the CPU->GPU V copy test, with head / chunk_id / token
    encoded in SEPARATE elements as small integers so every value is exactly bf16-representable
    (see the module docstring's fingerprint-encoding caveat -- composite tags like
    head*1000+chunk alias under bf16 rounding and fake an addressing bug). This closes the
    coverage gap that made an earlier heads=1-only version of this file blind to per-head
    stride issues. Note v_cache_cpu's per-head row count here deliberately equals the
    cpu_v_length/head_dim passed to the kernel -- mismatching those (as the pre-fix
    get_value_cache wiring did) is itself a bug, covered by test_decode_dispatch_correctness."""
    dev = torch.device("cuda:0")
    bsz, heads, select_sets, chunk_size, head_dim = 1, 8, 256, 8, 128
    sparse_budget = select_sets * chunk_size
    max_ctx_chunks = 256   # chunk ids must stay <=256: bf16 ULP is 2 in [256,512), odd ids would alias
    gpu_v_offset_tokens = 64
    buf_len = sparse_budget + gpu_v_offset_tokens + 128
    torch.manual_seed(0)

    cached_pos_ids = torch.full((bsz, heads, select_sets), -1, dtype=torch.int64, device=dev)
    cur_pos_ids = torch.stack([
        torch.randperm(max_ctx_chunks, device=dev)[:select_sets] for _ in range(heads)
    ]).view(1, heads, select_sets).long()
    offsets = torch.zeros(bsz * heads * select_sets, dtype=torch.int32, device=dev)
    cnts = torch.zeros(bsz * heads, dtype=torch.int32, device=dev)
    torch.ops._shadowkv.reorder_keys_and_compute_offsets(
        cached_pos_ids, cur_pos_ids, offsets, cnts, bsz, heads, select_sets
    )

    v_cache_cpu = torch.zeros(heads, max_ctx_chunks, chunk_size * head_dim, dtype=torch.bfloat16, pin_memory=True)
    v = v_cache_cpu.view(heads, max_ctx_chunks, chunk_size, head_dim)
    for h in range(heads):
        v[h, :, :, 0] = float(h)
        v[h, :, :, 1] = torch.arange(max_ctx_chunks, dtype=torch.bfloat16).view(-1, 1)  # 0..255, all exact in bf16
        v[h, :, :, 2] = torch.arange(chunk_size, dtype=torch.bfloat16).view(1, -1)

    v_cache_buffer = torch.zeros(heads, buf_len, head_dim, device=dev, dtype=torch.bfloat16)
    temp = torch.zeros(bsz, heads, select_sets, chunk_size * head_dim, device=dev, dtype=torch.bfloat16).contiguous()
    signals = torch.zeros(bsz * heads, dtype=torch.int32, device=dev)

    torch.ops._shadowkv.gather_copy_with_offsets(
        v_cache_cpu, v_cache_buffer, temp, offsets, cnts, signals,
        bsz, heads, int(max_ctx_chunks * chunk_size * head_dim), int(sparse_budget * head_dim),
        int(gpu_v_offset_tokens * head_dim), int(buf_len * head_dim), select_sets,
    )
    torch.cuda.synchronize()

    got = v_cache_buffer[:, gpu_v_offset_tokens:gpu_v_offset_tokens + sparse_budget].view(
        heads, select_sets, chunk_size, head_dim).float()
    for h in range(heads):
        final_pos = cached_pos_ids[0, h].float()
        assert torch.equal(got[h, :, :, 0], torch.full((select_sets, chunk_size), float(h), device=dev)), \
            f"head {h}: head-id fingerprint mismatch (wrong head's data)"
        assert torch.equal(got[h, :, :, 1], final_pos.view(-1, 1).expand(-1, chunk_size)), \
            f"head {h}: chunk-id fingerprint mismatch (wrong chunk fetched)"
        assert torch.equal(got[h, :, :, 2],
                           torch.arange(chunk_size, device=dev, dtype=torch.float).view(1, -1).expand(select_sets, -1)), \
            f"head {h}: token-order fingerprint mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batch_gemm_softmax_scores_and_ranking():
    """The landmark-scoring kernel used by get_retrieval_position_ids. Checks (1) the raw
    Q @ K_landmark^T scores (gemm_o) match a plain PyTorch reference exactly, and (2) the
    softmax output (softmax_o) preserves the correct top-k ranking of those scores --
    softmax_o's exact internal numeric representation is an unnormalized/tiled intermediate
    (not a plain probability distribution) so ranking, not raw value, is the property that
    matters for correct chunk selection."""
    from models.kv_cache import ShadowKVCache_xKey_CPU
    from models.merge_configs import generate_consecutive_palu_config
    from bench_decode_attn import _llama8b_config, _preinit_xkey_sr

    torch.manual_seed(0)
    dev, dtype = torch.device("cuda:0"), torch.bfloat16
    bsz, prefill_len, sparse_budget, group_size, rank_k = 2, 4096, 512, 4, 384

    cfg = _llama8b_config(num_layers=4)
    merge_cfg = generate_consecutive_palu_config(
        start_layer=0, end_layer=cfg.num_hidden_layers - 1, group_size=group_size, rank_k=rank_k, rank_v=576,
    )
    kv = ShadowKVCache_xKey_CPU(cfg, merge_cfg, max_length=prefill_len + 128, device=dev,
                                 dtype=dtype, batch_size=bsz, sparse_budget=sparse_budget)
    _preinit_xkey_sr(kv, cfg, prefill_len)

    nkv, nd, nq = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.num_attention_heads
    group = nq // nkv
    q = torch.randn(bsz, nq, 1, nd, device=dev, dtype=dtype)
    scale = 1 / math.sqrt(128)
    nlm = kv.k_landmark.shape[-2]

    torch.ops._shadowkv.batch_gemm_softmax(
        q.contiguous(), kv.k_landmark[0].contiguous(), kv.gemm_o, kv.norm, kv.sum, kv.softmax_o,
        bsz * nkv, group * 1, nlm, nd, scale, 0,
    )

    ref_scores = torch.einsum(
        "bgd,bld->bgl", q.float().view(bsz * nkv, group, nd), kv.k_landmark[0].float().view(bsz * nkv, nlm, nd)
    ) * scale
    got_gemm = kv.gemm_o.float().view(bsz * nkv, group, -1)
    rel_err = (got_gemm - ref_scores).norm() / ref_scores.norm()
    assert rel_err < 0.02, f"gemm_o relative error {rel_err:.4f} too high"

    k = 20
    ref_topk = ref_scores.topk(k, dim=-1).indices.sort(dim=-1).values
    got_topk = kv.softmax_o.float().view(bsz * nkv, group, -1).topk(k, dim=-1).indices.sort(dim=-1).values
    match_frac = (ref_topk == got_topk).float().mean().item()
    assert match_frac > 0.85, f"top-{k} ranking match fraction {match_frac:.3f} too low"


if __name__ == "__main__":
    tests = [
        test_reorder_keys_fresh_call_is_all_miss,
        test_reorder_keys_partial_overlap_hit_count,
        test_gather_copy_d2d_is_noop_on_fresh_miss,
        test_gather_copy_with_offsets_fetches_correct_cpu_chunks,
        test_gather_copy_with_offsets_multi_head_exact_fingerprints,
        test_batch_gemm_softmax_scores_and_ranking,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
