#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
REVISION="${REVISION:-main}"
"$PYTHON_BIN" scripts/ruler_qa/py/01_download_model.py \
    --model "$MODEL_ID" --revision "$REVISION" --output "$MODEL_PATH" "$@"
