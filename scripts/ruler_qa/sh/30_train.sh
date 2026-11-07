#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29610}"
BACKEND="${BACKEND:-ddp}"
CONFIG="${CONFIG:-configs/ruler_qa/train.json}"
PREPARED_DIR="${PREPARED_DIR:-$RULER_QA_ROOT/prepared/train_v1}"
RESUME="${RESUME:-}"
MAX_STEPS="${MAX_STEPS:-}"
PLAN_ONLY="${PLAN_ONLY:-0}"
if [[ -n "$RESUME" ]]; then
    RUN_DIR="${RUN_DIR:-$(dirname -- "$RESUME")}"
elif [[ "$NNODES" != 1 && -z "${RUN_DIR:-}" ]]; then
    echo 'All training nodes must specify the same shared RUN_DIR.' >&2
    exit 2
else
    RUN_DIR="${RUN_DIR:-$RULER_QA_ROOT/runs/ruler_qa_$(timestamp)_${RANDOM}}"
fi

options=(--config "$CONFIG" --prepared-dir "$PREPARED_DIR" --output "$RUN_DIR" --backend "$BACKEND")
if [[ -n "${MODEL_PATH:-}" ]]; then
    options+=(--model "$MODEL_PATH")
elif [[ "$CONFIG" == configs/ruler_qa/train.json && -d "$DEFAULT_MODEL_PATH" ]]; then
    options+=(--model "$DEFAULT_MODEL_PATH")
fi
[[ -n "$RESUME" ]] && options+=(--resume "$RESUME")
[[ -n "$MAX_STEPS" ]] && options+=(--max-steps "$MAX_STEPS")
if [[ "$PLAN_ONLY" == 1 ]]; then
    WORLD_SIZE=$((NNODES * NPROC_PER_NODE)) "$PYTHON_BIN" scripts/ruler_qa/py/30_train.py \
        "${options[@]}" --plan-only "$@"
    exit 0
fi

mkdir -p "$RULER_QA_ROOT/logs"
LOG_FILE="$RULER_QA_ROOT/logs/train_$(timestamp)_node${NODE_RANK}_${RANDOM}.log"
"$PYTHON_BIN" -m torch.distributed.run --nnodes "$NNODES" --node_rank "$NODE_RANK" \
    --nproc_per_node "$NPROC_PER_NODE" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    scripts/ruler_qa/py/30_train.py "${options[@]}" "$@" 2>&1 | tee "$LOG_FILE"
printf 'RULER QA run: %s\nLog: %s\n' "$RUN_DIR" "$LOG_FILE"
