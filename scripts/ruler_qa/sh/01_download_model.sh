#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
REVISION="${REVISION:-main}"
"$PYTHON_BIN" "$PY_ENTRY_DIR/01_download_model.py" \
    --model "$MODEL_ID" --revision "$REVISION" --output "$MODEL_PATH" "$@"
