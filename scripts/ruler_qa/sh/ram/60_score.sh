#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=ram
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export PREDICTIONS="${PREDICTIONS:?Set PREDICTIONS to predictions.jsonl for one ratio}"
exec bash "$ENTRY_DIR/60_score.sh" "$@"
