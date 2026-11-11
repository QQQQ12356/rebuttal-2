#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=icae
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to an adapter or merged export}"
export TEST_DIR="${TEST_DIR:-$RULER_QA_ROOT/prepared/test_v1}"
export PREPARED_DIR="${PREPARED_DIR:-$RULER_QA_ROOT/prepared/train_v1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"                    # Must fit the prepared generation budget.
export DECODING="${DECODING:-greedy}"                            # greedy | sample | beam.
export PLAN_ONLY="${PLAN_ONLY:-0}"
export RATIOS="${RATIOS:-4 8 16 32}"
exec bash "$ENTRY_DIR/50_benchmark.sh" "$@"
