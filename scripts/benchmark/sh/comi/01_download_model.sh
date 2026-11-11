#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=comi
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
export REVISION="${REVISION:-main}"                         # Prefer a commit SHA for repeat runs.
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
exec bash "$ENTRY_DIR/01_download_model.sh" "$@"
