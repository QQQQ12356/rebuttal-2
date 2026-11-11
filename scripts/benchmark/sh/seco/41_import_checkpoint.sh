#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=seco
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export SOURCE="${SOURCE:-/path/to/original/model.safetensors}"   # Original *unmerged* SeCo dual-LoRA checkpoint.
export CONFIG="${CONFIG:-configs/benchmark/$METHOD/train.json}"                    # Edit base model, LoRA r/alpha/dropout to match original.
export OUTPUT_DIR="${OUTPUT_DIR:-$BENCHMARK_ROOT/imported/original_seco}"
exec bash "$ENTRY_DIR/41_import_checkpoint.sh" "$@"
