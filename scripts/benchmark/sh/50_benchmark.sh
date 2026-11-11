#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"    # Each process holds one complete encoder/decoder pair.
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"                       # More GPUs increase throughput, not per-model capacity.
CHECKPOINT="${CHECKPOINT:-$BENCHMARK_ROOT/runs/REPLACE_WITH_RUN/final_merged}"
PREPARED_DIR="${PREPARED_DIR:-$BENCHMARK_ROOT/prepared/default}"
EVAL_TRAIN="${EVAL_TRAIN:-0}"                              # 1 evaluates the checkpoint's prepared training data.
DATASETS="${DATASETS:-}"                                  # Space-separated; empty uses all datasets for the split.
read -r -a ratios <<< "${RATIOS:-4 8 16 32}"                                       # Fixed ratios, each applied to exactly the same examples.
BATCH_SIZE="${BATCH_SIZE:-1}"                             # Increase only when each GPU has enough memory.
SAMPLES="${SAMPLES:-0}"                                  # Per dataset; 0 = all preprocessed test examples.
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}"                  # Global cap after SAMPLES; 0 = all.
WORKERS="${WORKERS:-2}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"                      # Repeat first batch; all examples still scored once.
QA_MAX_NEW_TOKENS="${QA_MAX_NEW_TOKENS:-64}"
NEWS_MAX_NEW_TOKENS="${NEWS_MAX_NEW_TOKENS:-512}"
DECODING="${DECODING:-greedy}"                            # greedy | sample | beam.
TEMPERATURE="${TEMPERATURE:-}"                            # sample only; empty uses 1.0.
TOP_P="${TOP_P:-}"                                        # sample only; empty uses 1.0.
TOP_K="${TOP_K:-}"                                        # sample only; empty uses 0 (disabled).
NUM_BEAMS="${NUM_BEAMS:-}"                                # beam only; empty uses 4.
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-0}"                     # Suppress EOS before this many new tokens.
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"           # 1.0 disables the repetition penalty.
SEED="${SEED:-42}"                                        # Generation and optional dataset shuffle.
INCLUDE_CONTEXT="${INCLUDE_CONTEXT:-0}"                    # 1 stores full input for later Excel details.
SHUFFLE="${SHUFFLE:-0}"                                    # 1 shuffles before sample selection.
MODEL_PATH="${MODEL_PATH:-}"                               # Optional relocated base for adapters.
ATTENTION="${ATTENTION:-}"
DTYPE="${DTYPE:-}" # Empty uses the checkpoint precision.
NO_MERGE="${NO_MERGE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
generation_options=(--decoding "$DECODING" --min-new-tokens "$MIN_NEW_TOKENS"
                    --repetition-penalty "$REPETITION_PENALTY")
[[ -n "$TEMPERATURE" ]] && generation_options+=(--temperature "$TEMPERATURE")
[[ -n "$TOP_P" ]] && generation_options+=(--top-p "$TOP_P")
[[ -n "$TOP_K" ]] && generation_options+=(--top-k "$TOP_K")
[[ -n "$NUM_BEAMS" ]] && generation_options+=(--num-beams "$NUM_BEAMS")
[[ -n "$SEED" ]] && generation_options+=(--seed "$SEED")

extra=()
case "$EVAL_TRAIN" in
    0) run_prefix=benchmark ;;
    1) run_prefix=benchmark_train; extra+=(--eval-train) ;;
    *) echo 'EVAL_TRAIN must be 0 or 1.' >&2; exit 2 ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-$BENCHMARK_ROOT/benchmarks/${run_prefix}_$(timestamp)_${RANDOM}}"
if [[ -n "$DATASETS" ]]; then
    read -r -a datasets <<< "$DATASETS"
    extra+=(--datasets "${datasets[@]}")
fi
[[ "$INCLUDE_CONTEXT" == 1 ]] && extra+=(--include-context)
[[ "$SHUFFLE" == 1 ]] && extra+=(--shuffle)
[[ -n "$MODEL_PATH" ]] && extra+=(--model "$MODEL_PATH")
[[ -n "$ATTENTION" ]] && extra+=(--attention "$ATTENTION")
[[ -n "$DTYPE" ]] && extra+=(--dtype "$DTYPE")
[[ "$NO_MERGE" == 1 ]] && extra+=(--no-merge)
entry="$PY_ENTRY_DIR/50_benchmark.py"
options=(--checkpoint "$CHECKPOINT" --prepared-dir "$PREPARED_DIR" --output "$OUTPUT_DIR"
         --ratios "${ratios[@]}" --batch-size "$BATCH_SIZE" --samples "$SAMPLES"
         --max-test-samples "$MAX_TEST_SAMPLES" --workers "$WORKERS" --warmup-batches "$WARMUP_BATCHES"
         --qa-max-new-tokens "$QA_MAX_NEW_TOKENS" --news-max-new-tokens "$NEWS_MAX_NEW_TOKENS"
         "${generation_options[@]}" "${extra[@]}")
if [[ "$PLAN_ONLY" == 1 ]]; then
    WORLD_SIZE="$NPROC_PER_NODE" "$PYTHON_BIN" "$entry" "${options[@]}" --plan-only "$@"
    exit 0
fi
mkdir -p "$BENCHMARK_ROOT/logs"
LOG_FILE="$BENCHMARK_ROOT/logs/benchmark_$(timestamp)_${RANDOM}.log"
printf 'Log: %s\n' "$LOG_FILE"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
    "$entry" "${options[@]}" "$@" 2>&1 | tee "$LOG_FILE"
printf 'Benchmark directory: %s\nLog: %s\n' "$OUTPUT_DIR" "$LOG_FILE"
