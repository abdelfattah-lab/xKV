"""
End-to-end test of the REAL top-k landmark retrieval (get_retrieval_position_ids), not
bypassed. Earlier composition tests (test_shadowkv_pipeline_composition.py) deliberately
injected position_ids directly instead of calling get_retrieval_position_ids, because an
earlier attempt at driving it with production-benchmark shapes returned all -1 (nothing
selected). That turned out to be a test-setup issue, not a kernel bug: reorder_keys_and_
compute_offsets silently no-ops unless select_sets (=sparse_budget/chunk_size) is one of the
CUTLASS template-dispatch sizes {128,256,512,1024} (see test_shadowkv_micro_ops_correctness.py)
-- the earlier attempt used sparse_budget=512/chunk_size=8=64, which isn't one of those sizes.
Production always uses sparse_budget=2048/chunk_size=8=256, which is supported.

With that fixed, this test verifies get_retrieval_position_ids actually does its job: given
landmark data where specific chunks are deliberately constructed to align strongly with the
query (and everything else is low-magnitude "noise"), the top-k selection must include those
chunks. This is a real correctness check of the full real pipeline (batch_gemm_softmax score
-> torch.topk -> reorder_keys_and_compute_offsets), not a bypass.

Usage:
  pytest efficiency/tests/test_shadowkv_retrieval_correctness.py -v
  python efficiency/tests/test_shadowkv_retrieval_correctness.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import ops.shadowkv  # noqa: F401
from models.kv_cache import ShadowKVCache_xKey_CPU
from models.merge_configs import generate_consecutive_palu_config
from bench_decode_attn import _llama8b_config, _preinit_xkey_sr


@torch.inference_mode()
def _run_case(group_size, rank_k, target_chunks, seed=0):
    torch.manual_seed(seed)
    dev, dtype = torch.device("cuda:0"), torch.bfloat16
    bsz, prefill_len, sparse_budget = 1, 60_000, 2048   # sparse_budget=2048 -> select_sets=256 (supported)

    cfg = _llama8b_config(num_layers=4)
    merge_cfg = generate_consecutive_palu_config(
        start_layer=0, end_layer=cfg.num_hidden_layers - 1, group_size=group_size, rank_k=rank_k, rank_v=576,
    )
    kv = ShadowKVCache_xKey_CPU(cfg, merge_cfg, max_length=prefill_len + 128, device=dev,
                                 dtype=dtype, batch_size=bsz, sparse_budget=sparse_budget)
    _preinit_xkey_sr(kv, cfg, prefill_len)

    nkv, nd, nq = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.num_attention_heads
    nlm = kv.k_landmark.shape[-2]
    assert all(tc < nlm for tc in target_chunks)

    q = torch.randn(bsz, nq, 1, nd, device=dev, dtype=dtype)
    kv.k_landmark.normal_(std=0.02)   # low-magnitude "irrelevant" landmarks
    for tc in target_chunks:
        # strongly align this landmark with q (same value for every kv head this query attends to)
        kv.k_landmark[0, :, :, tc, :] = q[0, ::nq // nkv, 0, :].to(dtype) * 5.0
    kv.k_landmark_idx.copy_(torch.arange(nlm, device=dev).view(1, 1, 1, -1).expand_as(kv.k_landmark_idx))

    pos = kv.get_retrieval_position_ids(layer_idx=0, query_states=q)
    return pos


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_size,rank_k", [(4, 384), (1, 96)], ids=["xkey_sr_config", "shadowkv_config"])
def test_retrieval_selects_deliberately_relevant_chunks(group_size, rank_k):
    target_chunks = [3, 50, 200, 400]
    pos = _run_case(group_size, rank_k, target_chunks)

    assert not (pos == -1).any(), "retrieval left some slots unfilled (-1)"
    selected = set(pos[0, 0].tolist())
    missing = [tc for tc in target_chunks if tc not in selected]
    assert not missing, f"deliberately-relevant chunks {missing} were not retrieved"


if __name__ == "__main__":
    for group_size, rank_k, name in [(4, 384, "xkey_sr_config"), (1, 96, "shadowkv_config")]:
        try:
            test_retrieval_selects_deliberately_relevant_chunks(group_size, rank_k)
            print(f"[PASS] {name}")
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
