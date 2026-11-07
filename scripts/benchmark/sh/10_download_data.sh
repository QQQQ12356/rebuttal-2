#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

REGISTRY="${REGISTRY:-configs/benchmark/datasets.json}"           # Dataset repo/config/split/revision mapping.
RAW_DIR="${RAW_DIR:-artifacts/raw}"                    # Raw JSONL output root.
ROLE="${ROLE:-all}"                                    # all / training / benchmark.
TRAIN_LIMIT="${TRAIN_LIMIT:-30000}"                     # First N raw rows/source; 0 = entire training split.
DOWNLOAD_MODE="${DOWNLOAD_MODE:-local}"                 # local: resumable shards then local decode; stream: old remote mode.
SOURCE_CACHE="${SOURCE_CACHE:-artifacts/source_cache}"   # Retain completed and .incomplete Hub shard files.
DOWNLOAD_ATTEMPTS="${DOWNLOAD_ATTEMPTS:-6}"
METADATA_TIMEOUT="${METADATA_TIMEOUT:-60}"               # Dataset-info API timeout; separate from file request timeout.
RETRY_DELAY="${RETRY_DELAY:-2}"                         # Exponential backoff, capped at 30 seconds.
ROW_BATCH_SIZE="${ROW_BATCH_SIZE:-16}"                  # Local Arrow decoding, without background reader threads.
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"     # HTTP transport for predictable partial-file resume.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
# Four benchmark splits always download in full; NarrativeQA test is intentionally absent.
"$PYTHON_BIN" scripts/benchmark/py/10_download_data.py \
    --registry "$REGISTRY" --output "$RAW_DIR" --role "$ROLE" --train-limit "$TRAIN_LIMIT" \
    --download-mode "$DOWNLOAD_MODE" --source-cache "$SOURCE_CACHE" --download-attempts "$DOWNLOAD_ATTEMPTS" \
    --metadata-timeout "$METADATA_TIMEOUT" --retry-delay "$RETRY_DELAY" \
    --row-batch-size "$ROW_BATCH_SIZE" --progress-every "$PROGRESS_EVERY" "$@"
