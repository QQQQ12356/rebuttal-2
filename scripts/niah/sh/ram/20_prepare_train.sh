#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=ram
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export PREPARE_CONFIG="${PREPARE_CONFIG:-configs/niah/prepare_train.json}" # Counts, candidate limits, length filters, sources.
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
export RAW_DIR="${RAW_DIR:-$NIAH_ROOT/raw}"
export PREPARED_DIR="${PREPARED_DIR:-$NIAH_ROOT/prepared/train_v1}"        # Must be a new directory.
exec bash "$ENTRY_DIR/20_prepare_train.sh" "$@"
