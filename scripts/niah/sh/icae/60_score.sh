#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=icae
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export PREDICTIONS="${PREDICTIONS:-$NIAH_ROOT/benchmarks/REPLACE_WITH_BENCHMARK/ratio_16/predictions.jsonl}"
exec bash "$ENTRY_DIR/60_score.sh" "$@"
