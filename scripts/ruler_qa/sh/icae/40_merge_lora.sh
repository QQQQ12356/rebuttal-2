#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=icae
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to final or checkpoint-N}"
export MERGED_DIR="${MERGED_DIR:-${CHECKPOINT}_merged}"
export MERGE_DEVICE="${MERGE_DEVICE:-cpu}"
exec bash "$ENTRY_DIR/40_merge_lora.sh" "$@"
