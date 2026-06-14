################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
#
# Build evaluation datasets for xKV / xKV-SR.
# Run from repo root: bash scripts/build_datasets.sh
#
# RULER data lands in: evaluate/data/ruler/data/<model_dir>/<length>/<task>/validation.jsonl

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

build_ruler() {
    local model_id="$1"
    local model_dir="$2"
    local marker="$REPO_ROOT/evaluate/data/ruler/data/$model_dir"
    if [ -d "$marker" ]; then
        echo "[SKIP] RULER $model_dir already exists"
        return
    fi
    echo "[BUILD] RULER $model_dir ($model_id)"
    (cd "$REPO_ROOT/evaluate/data/ruler" && bash create_dataset.sh "$model_id" "$model_dir")
}

# ── RULER ─────────────────────────────────────────────────────────────────────
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# llama-3  (meta-llama/Meta-Llama-3.1-8B-Instruct, etc.)
build_ruler "meta-llama/Meta-Llama-3.1-8B-Instruct" "llama-3"

# qwen  (Qwen2.5 / Qwen3 — same tokenizer within each family)
build_ruler "Qwen/Qwen2.5-14B-Instruct-1M" "qwen"
