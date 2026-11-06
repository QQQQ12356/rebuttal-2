#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

REGISTRY="${REGISTRY:-configs/datasets.json}"           # Dataset repo/config/split/revision mapping.
RAW_DIR="${RAW_DIR:-artifacts/raw}"                    # Raw JSONL output root.
ROLE="${ROLE:-all}"                                    # all / training / benchmark.
TRAIN_LIMIT="${TRAIN_LIMIT:-30000}"                     # First N raw rows/source; 0 = entire training split.
# Four benchmark splits always download in full; NarrativeQA test is intentionally absent.
"$PYTHON_BIN" scripts/py/10_download_data.py \
    --registry "$REGISTRY" --output "$RAW_DIR" --role "$ROLE" --train-limit "$TRAIN_LIMIT" "$@"
