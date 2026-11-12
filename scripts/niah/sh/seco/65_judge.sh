#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=seco
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export PREDICTIONS="${PREDICTIONS:?Set PREDICTIONS to predictions.jsonl for one ratio}"
export JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-60}"
export JUDGE_ATTEMPTS="${JUDGE_ATTEMPTS:-4}"
exec bash "$ENTRY_DIR/65_judge.sh" "$@"
