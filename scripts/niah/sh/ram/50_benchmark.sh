#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=ram
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"                       # Full model pair on each GPU; no tensor parallel.
export CHECKPOINT="${CHECKPOINT:-$NIAH_ROOT/runs/REPLACE_WITH_RUN/final_merged}"
export GRID_DIR="${GRID_DIR:-$NIAH_ROOT/prepared/grid_v1}"
export PREPARED_DIR="${PREPARED_DIR:-$NIAH_ROOT/prepared/train_v1}" # Only used for training-set evaluation.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"                     # Must fit the selected data's decoder budget.
export DECODING="${DECODING:-greedy}"                            # greedy | sample | beam.
export PLAN_ONLY="${PLAN_ONLY:-0}"
export RATIOS="${RATIOS:-4 8 16 32}"
exec bash "$ENTRY_DIR/50_benchmark.sh" "$@"
