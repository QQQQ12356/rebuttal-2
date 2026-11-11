#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=comi
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
export RAW_DIR="${RAW_DIR:-$BENCHMARK_ROOT/raw}"
export PREPARED_DIR="${PREPARED_DIR:-$BENCHMARK_ROOT/prepared/default}"  # Must be a new directory each time.
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-16384}"             # Context + prompt. Overlength examples are filtered.
export MAX_DECODER_TOKENS="${MAX_DECODER_TOKENS:-8192}"          # Centers + prompt + answer/generation reserve.
export MIN_RATIO="${MIN_RATIO:-4}"                              # Smallest ratio planned for training AND testing.
exec bash "$ENTRY_DIR/20_prepare_data.sh" "$@"
