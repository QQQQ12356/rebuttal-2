#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=icae
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export REGISTRY="${REGISTRY:-configs/benchmark/datasets.json}"           # Dataset repo/config/split/revision mapping.
export RAW_DIR="${RAW_DIR:-$BENCHMARK_ROOT/raw}"                    # Raw JSONL output root.
export TRAIN_LIMIT="${TRAIN_LIMIT:-30000}"                     # First N raw rows/source; 0 = entire training split.
exec bash "$ENTRY_DIR/10_download_data.sh" "$@"
