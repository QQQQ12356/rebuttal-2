#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"                       # Full model pair on each GPU; no tensor parallel.
CHECKPOINT="${CHECKPOINT:-$NIAH_ROOT/runs/REPLACE_WITH_RUN/final_merged}"
GRID_DIR="${GRID_DIR:-$NIAH_ROOT/prepared/grid_v1}"
PREPARED_DIR="${PREPARED_DIR:-$NIAH_ROOT/prepared/train_v1}" # Only used for training-set evaluation.
EVAL_TRAIN="${EVAL_TRAIN:-0}"                              # 0: held-out grid. 1: checkpoint's training data.
read -r -a ratios <<< "${RATIOS:-4 8 16 32}"               # Same examples at every fixed ratio.
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-2}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"                     # Must fit the selected data's decoder budget.
DECODING="${DECODING:-greedy}"                            # greedy | sample | beam.
TEMPERATURE="${TEMPERATURE:-}"                            # sample only; empty uses 1.0.
TOP_P="${TOP_P:-}"                                        # sample only; empty uses 1.0.
TOP_K="${TOP_K:-}"                                        # sample only; empty uses 0 (disabled).
NUM_BEAMS="${NUM_BEAMS:-}"                                # beam only; empty uses 4.
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-0}"                     # Suppress EOS before this many new tokens.
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"           # 1.0 disables the repetition penalty.
SEED="${SEED:-}"                                          # Empty uses the prepared dataset seed.
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-}"
SHUFFLE="${SHUFFLE:-0}"
MODEL_PATH="${MODEL_PATH:-}"
ATTENTION="${ATTENTION:-}"
DTYPE="${DTYPE:-}" # Empty uses the checkpoint precision.
NO_MERGE="${NO_MERGE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

case "$EVAL_TRAIN" in
    0)
        entry="$PY_ENTRY_DIR/50_benchmark.py"
        dataset_options=(--grid-dir "$GRID_DIR")
        run_prefix=niah
        log_prefix=benchmark
        ;;
    1)
        entry="$PY_ENTRY_DIR/51_benchmark_train.py"
        dataset_options=(--eval-train --prepared-dir "$PREPARED_DIR")
        run_prefix=niah_train
        log_prefix=train_benchmark
        ;;
    *)
        echo 'EVAL_TRAIN must be 0 or 1.' >&2
        exit 2
        ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-$NIAH_ROOT/benchmarks/${run_prefix}_$(timestamp)_${RANDOM}}"
generation_options=(--decoding "$DECODING" --min-new-tokens "$MIN_NEW_TOKENS"
                    --repetition-penalty "$REPETITION_PENALTY")
[[ -n "$TEMPERATURE" ]] && generation_options+=(--temperature "$TEMPERATURE")
[[ -n "$TOP_P" ]] && generation_options+=(--top-p "$TOP_P")
[[ -n "$TOP_K" ]] && generation_options+=(--top-k "$TOP_K")
[[ -n "$NUM_BEAMS" ]] && generation_options+=(--num-beams "$NUM_BEAMS")
[[ -n "$SEED" ]] && generation_options+=(--seed "$SEED")

options=(--checkpoint "$CHECKPOINT" "${dataset_options[@]}" --output "$OUTPUT_DIR"
         --ratios "${ratios[@]}" --batch-size "$BATCH_SIZE" --workers "$WORKERS"
         --warmup-batches "$WARMUP_BATCHES" --max-new-tokens "$MAX_NEW_TOKENS"
         "${generation_options[@]}")
[[ -n "$MAX_TEST_SAMPLES" ]] && options+=(--max-test-samples "$MAX_TEST_SAMPLES")
case "${RECORD_GROUP_STATS:-}" in
    '') ;; # Use the checkpoint method's default.
    0) options+=(--no-record-group-stats) ;;
    1) options+=(--record-group-stats) ;;
    *) echo 'RECORD_GROUP_STATS must be 0 or 1.' >&2; exit 2 ;;
esac
[[ "$SHUFFLE" == 1 ]] && options+=(--shuffle)
[[ -n "$MODEL_PATH" ]] && options+=(--model "$MODEL_PATH")
[[ -n "$ATTENTION" ]] && options+=(--attention "$ATTENTION")
[[ -n "$DTYPE" ]] && options+=(--dtype "$DTYPE")
[[ "$NO_MERGE" == 1 ]] && options+=(--no-merge)
if [[ "$PLAN_ONLY" == 1 ]]; then
    WORLD_SIZE="$NPROC_PER_NODE" "$PYTHON_BIN" "$entry" "${options[@]}" --plan-only "$@"
    exit 0
fi
mkdir -p "$NIAH_ROOT/logs"
LOG_FILE="$NIAH_ROOT/logs/${log_prefix}_$(timestamp)_${RANDOM}.log"
printf 'Log: %s\n' "$LOG_FILE"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
    "$entry" "${options[@]}" "$@" 2>&1 | tee "$LOG_FILE"
printf 'NIAH benchmark: %s\nLog: %s\n' "$OUTPUT_DIR" "$LOG_FILE"
