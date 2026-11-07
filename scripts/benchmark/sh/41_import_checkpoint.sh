#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

SOURCE="${SOURCE:-/path/to/original/model.safetensors}"   # Original *unmerged* SeCo dual-LoRA checkpoint.
CONFIG="${CONFIG:-configs/benchmark/train.json}"                    # Edit base model, LoRA r/alpha/dropout to match original.
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/imported/original_seco}"
"$PYTHON_BIN" scripts/benchmark/py/41_import_checkpoint.py \
    --source "$SOURCE" --config "$CONFIG" --output "$OUTPUT_DIR" "$@"
