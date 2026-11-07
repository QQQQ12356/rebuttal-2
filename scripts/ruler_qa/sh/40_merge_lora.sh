#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to final or checkpoint-N}"
MERGED_DIR="${MERGED_DIR:-${CHECKPOINT}_merged}"
MERGE_DEVICE="${MERGE_DEVICE:-cpu}"
options=(--checkpoint "$CHECKPOINT" --output "$MERGED_DIR" --device "$MERGE_DEVICE")
[[ -n "${MODEL_PATH:-}" ]] && options+=(--model "$MODEL_PATH")
"$PYTHON_BIN" scripts/ruler_qa/py/40_merge_lora.py "${options[@]}" "$@"
