#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=seco
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export CHECKPOINT="${CHECKPOINT:-$BENCHMARK_ROOT/runs/REPLACE_WITH_RUN/final}"
export MERGED_DIR="${MERGED_DIR:-${CHECKPOINT}_merged}"             # New immutable export folder.
export MERGE_DEVICE="${MERGE_DEVICE:-cpu}"                        # cpu avoids occupying a training GPU; requires RAM.
exec bash "$ENTRY_DIR/40_merge_lora.sh" "$@"
