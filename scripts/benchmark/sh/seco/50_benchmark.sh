#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=seco
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"    # Each process holds one complete encoder/decoder pair.
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"                       # More GPUs increase throughput, not per-model capacity.
export CHECKPOINT="${CHECKPOINT:-$BENCHMARK_ROOT/runs/REPLACE_WITH_RUN/final_merged}"
export PREPARED_DIR="${PREPARED_DIR:-$BENCHMARK_ROOT/prepared/default}"
export EVAL_TRAIN="${EVAL_TRAIN:-0}"                              # 1 evaluates the checkpoint's prepared training data.
export DATASETS="${DATASETS:-}"                                  # Space-separated; empty uses all datasets for the split.
export SAMPLES="${SAMPLES:-0}"                                  # Per dataset; 0 = all preprocessed test examples.
export MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}" # Global count per ratio; 0 = all.
export SHUFFLE="${SHUFFLE:-0}"                                    # 1 shuffles before sample selection.
export BATCH_SIZE="${BATCH_SIZE:-1}"                             # Increase only when each GPU has enough memory.
export WORKERS="${WORKERS:-2}"
export WARMUP_BATCHES="${WARMUP_BATCHES:-1}"                      # Repeat first batch; all examples still scored once.
export QA_MAX_NEW_TOKENS="${QA_MAX_NEW_TOKENS:-64}"
export NEWS_MAX_NEW_TOKENS="${NEWS_MAX_NEW_TOKENS:-512}"
export DECODING="${DECODING:-greedy}"                            # greedy | sample | beam.
export TEMPERATURE="${TEMPERATURE:-}"                            # sample only; empty uses 1.0.
export TOP_P="${TOP_P:-}"                                        # sample only; empty uses 1.0.
export TOP_K="${TOP_K:-}"                                        # sample only; empty uses 0 (disabled).
export NUM_BEAMS="${NUM_BEAMS:-}"                                # beam only; empty uses 4.
export MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-0}"                     # Suppress EOS before this many new tokens.
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"           # 1.0 disables the repetition penalty.
export SEED="${SEED:-42}"                                        # Generation and optional dataset shuffle.
export MODEL_PATH="${MODEL_PATH:-}"                               # Optional relocated base for adapters.
export ATTENTION="${ATTENTION:-}"
export DTYPE="${DTYPE:-}" # Empty uses the checkpoint precision.
export NO_MERGE="${NO_MERGE:-0}"
export INCLUDE_CONTEXT="${INCLUDE_CONTEXT:-0}"                    # 1 stores full input for later Excel details.
export RECORD_GROUP_STATS="${RECORD_GROUP_STATS:-1}"                 # 1 saves context-token counts per SECO group; 0 disables.
export PLAN_ONLY="${PLAN_ONLY:-0}"
export RATIOS="${RATIOS:-4 8 16 32}"
export OUTPUT_DIR="${OUTPUT_DIR:-}" # Empty creates a new timestamped output directory.
exec bash "$ENTRY_DIR/50_benchmark.sh" "$@"
