#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_PATH="${MODEL_PATH:-artifacts/models/Qwen3-4B-Instruct-2507}"
REVISION="${REVISION:-main}"                               # Prefer commit SHA for replication.
"$PYTHON_BIN" scripts/niah/py/01_download_model.py \
    --model "$MODEL_ID" --revision "$REVISION" --output "$MODEL_PATH" "$@"
