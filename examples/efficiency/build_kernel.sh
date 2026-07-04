#!/usr/bin/env bash
# Build the _shadowkv CUDA kernel required for efficiency benchmarks.
#
# Run once from the repo root before running run_benchmarks.sh:
#   bash examples/efficiency/build_kernel.sh
#
# Requirements:
#   - CUDA 12.x (default: /usr/local/cuda-12.8)
#   - Python venv at .venv/ (or set PYTHON env var)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EFFICIENCY_DIR="${REPO_ROOT}/efficiency"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_HOME

echo "Building _shadowkv CUDA kernel..."
echo "  CUDA:   ${CUDA_HOME}"
echo "  Python: ${PYTHON}"

cd "${EFFICIENCY_DIR}"
"${PYTHON}" setup.py build_ext --inplace

echo ""
SO=$(find "${EFFICIENCY_DIR}/ops" -name "_shadowkv*.so" | head -1)
if [ -n "${SO}" ]; then
    echo "Build successful: ${SO}"
else
    echo "ERROR: .so not found after build" >&2
    exit 1
fi
