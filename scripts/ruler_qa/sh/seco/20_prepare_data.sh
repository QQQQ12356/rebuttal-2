#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=seco
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export SPLIT="${SPLIT:-train}"
export PREPARE_CONFIG="${PREPARE_CONFIG:-configs/ruler_qa/prepare.json}"
export RAW_DIR="${RAW_DIR:-$RULER_QA_ROOT/raw}"
export PREPARED_DIR="${PREPARED_DIR:-$RULER_QA_ROOT/prepared/${SPLIT}_v1}"
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
export MIN_RATIO="${MIN_RATIO:-}"                                 # 4. Lowest ratio used to check decoder capacity.
export MAX_DECODER_TOKENS="${MAX_DECODER_TOKENS:-}"                 # 16384. Maximum decoder sequence length.
exec bash "$ENTRY_DIR/20_prepare_data.sh" "$@"
