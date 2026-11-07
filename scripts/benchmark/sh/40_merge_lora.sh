#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Set CHECKPOINT to a new pipeline final/checkpoint-N folder; original weights need conversion first.
CHECKPOINT="${CHECKPOINT:-artifacts/runs/REPLACE_WITH_RUN/final}"
MERGED_DIR="${MERGED_DIR:-${CHECKPOINT}_merged}"             # New immutable export folder.
MERGE_DEVICE="${MERGE_DEVICE:-cpu}"                        # cpu avoids occupying a training GPU; requires RAM.
"$PYTHON_BIN" scripts/benchmark/py/40_merge_lora.py \
    --checkpoint "$CHECKPOINT" --output "$MERGED_DIR" --device "$MERGE_DEVICE" "$@"
