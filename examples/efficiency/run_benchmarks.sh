#!/usr/bin/env bash
# Run the xKV efficiency benchmarks.
#
# Prerequisites:
#   1. Build the CUDA kernel (one-time):
#        bash examples/efficiency/build_kernel.sh
#   2. Run this script from the repo root:
#        bash examples/efficiency/run_benchmarks.sh
#
# Environment overrides:
#   CUDA_VISIBLE_DEVICES=1 bash examples/efficiency/run_benchmarks.sh
#   WARMUP=5 ITERS=20 bash examples/efficiency/run_benchmarks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EFFICIENCY_DIR="${REPO_ROOT}/efficiency"
OUTPUT_DIR="${REPO_ROOT}/results/efficiency"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"

export PYTHONPATH="${EFFICIENCY_DIR}:${PYTHONPATH:-}"

echo "================================================================"
echo "  xKV Efficiency Benchmarks"
echo "  GPU:    $(${PYTHON} -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "  Output: ${OUTPUT_DIR}"
echo "================================================================"

echo ""
echo ">>> Step 1/3: Decode attention latency"
${PYTHON} "${EFFICIENCY_DIR}/bench_decode_attn.py" \
    --warmup "${WARMUP}" --iters "${ITERS}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo ">>> Step 2/3: SVD overhead during prefill"
${PYTHON} "${EFFICIENCY_DIR}/bench_svd_overhead.py" \
    --warmup "${WARMUP}" --iters "${ITERS}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo ">>> Step 3/3: E2E throughput comparison (max batch size per mode)"
# Compares FA2 at its memory-limited batch size vs xKV methods at their
# (much larger) max batch size.  Throughput = bs × 1000 / est_32layer_ms.
${PYTHON} "${EFFICIENCY_DIR}/bench_e2e_throughput.py" \
    --warmup "${WARMUP}" --iters "${ITERS}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "================================================================"
echo "  Done. Results saved to: ${OUTPUT_DIR}"
echo "================================================================"
