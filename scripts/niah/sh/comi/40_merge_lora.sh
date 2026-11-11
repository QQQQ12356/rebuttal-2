#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=comi
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export CHECKPOINT="${CHECKPOINT:-$NIAH_ROOT/runs/REPLACE_WITH_RUN/final}"
export MERGED_DIR="${MERGED_DIR:-${CHECKPOINT}_merged}"             # New encoder/decoder/tokenizer export.
export MERGE_DEVICE="${MERGE_DEVICE:-cpu}"                        # CPU requires enough RAM for the double model.
exec bash "$ENTRY_DIR/40_merge_lora.sh" "$@"
