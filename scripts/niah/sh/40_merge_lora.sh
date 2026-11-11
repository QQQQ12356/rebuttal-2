#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

CHECKPOINT="${CHECKPOINT:-$NIAH_ROOT/runs/REPLACE_WITH_RUN/final}"
MERGED_DIR="${MERGED_DIR:-${CHECKPOINT}_merged}"             # New encoder/decoder/tokenizer export.
MERGE_DEVICE="${MERGE_DEVICE:-cpu}"                        # CPU requires enough RAM for the double model.
"$PYTHON_BIN" "$PY_ENTRY_DIR/40_merge_lora.py" \
    --checkpoint "$CHECKPOINT" --output "$MERGED_DIR" --device "$MERGE_DEVICE" "$@"
