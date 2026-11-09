#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

SPLIT="${SPLIT:-train}"
PREPARE_CONFIG="${PREPARE_CONFIG:-configs/ruler_qa/prepare.json}"
RAW_DIR="${RAW_DIR:-$RULER_QA_ROOT/raw}"
PREPARED_DIR="${PREPARED_DIR:-$RULER_QA_ROOT/prepared/${SPLIT}_v1}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
REVISION="${REVISION:-main}"
PROMPT_STYLE="${PROMPT_STYLE:-raw}"  # raw: question only; legacy: instruction plus Question:/Answer:.
"$PYTHON_BIN" scripts/ruler_qa/py/20_prepare_data.py --split "$SPLIT" --config "$PREPARE_CONFIG" \
    --raw-dir "$RAW_DIR" --output "$PREPARED_DIR" --model "$MODEL_PATH" --revision "$REVISION" \
    --prompt-style "$PROMPT_STYLE" "$@"
