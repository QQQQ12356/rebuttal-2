#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Use HF_ENDPOINT=https://hf-mirror.com before invoking this script for a mirror.
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
REVISION="${REVISION:-main}"                         # Prefer a commit SHA for repeat runs.
MODEL_PATH="${MODEL_PATH:-artifacts/models/Qwen3-4B-Instruct-2507}"
"$PYTHON_BIN" scripts/benchmark/py/01_download_model.py \
    --model "$MODEL_ID" --revision "$REVISION" --output "$MODEL_PATH" "$@"
