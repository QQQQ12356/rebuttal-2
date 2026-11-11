#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=seco
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"                       # Full model pair on each GPU; no tensor parallel.
export CHECKPOINT="${CHECKPOINT:-$NIAH_ROOT/runs/REPLACE_WITH_RUN/final_merged}"
export GRID_DIR="${GRID_DIR:-$NIAH_ROOT/prepared/grid_v1}"
export PREPARED_DIR="${PREPARED_DIR:-$NIAH_ROOT/prepared/train_v1}" # Only used for training-set evaluation.
export EVAL_TRAIN="${EVAL_TRAIN:-0}"                              # 0: held-out grid. 1: checkpoint's training data.
export MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}" # Global count per ratio; 0 = all.
export SHUFFLE="${SHUFFLE:-0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export WORKERS="${WORKERS:-2}"
export WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"                     # Must fit the selected data's decoder budget.
export DECODING="${DECODING:-greedy}"                            # greedy | sample | beam.
export TEMPERATURE="${TEMPERATURE:-}"                            # sample only; empty uses 1.0.
export TOP_P="${TOP_P:-}"                                        # sample only; empty uses 1.0.
export TOP_K="${TOP_K:-}"                                        # sample only; empty uses 0 (disabled).
export NUM_BEAMS="${NUM_BEAMS:-}"                                # beam only; empty uses 4.
export MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-0}"                     # Suppress EOS before this many new tokens.
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"           # 1.0 disables the repetition penalty.
export SEED="${SEED:-}"                                          # Empty uses the prepared dataset seed.
export MODEL_PATH="${MODEL_PATH:-}"
export ATTENTION="${ATTENTION:-}"
export DTYPE="${DTYPE:-}" # Empty uses the checkpoint precision.
export NO_MERGE="${NO_MERGE:-0}"
export PLAN_ONLY="${PLAN_ONLY:-0}"
export RATIOS="${RATIOS:-4 8 16 32}"
export OUTPUT_DIR="${OUTPUT_DIR:-}" # Empty creates a new timestamped output directory.
exec bash "$ENTRY_DIR/50_benchmark.sh" "$@"
