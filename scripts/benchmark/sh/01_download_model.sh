#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Use HF_ENDPOINT=https://hf-mirror.com before invoking this script for a mirror.
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
REVISION="${REVISION:-main}"                         # Prefer a commit SHA for repeat runs.
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
"$PYTHON_BIN" "$PY_ENTRY_DIR/01_download_model.py" \
    --model "$MODEL_ID" --revision "$REVISION" --output "$MODEL_PATH" "$@"
