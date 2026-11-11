#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29620}"              # Separate from all task fine-tuning ports.
BACKEND="${BACKEND:-ddp}"
CONFIG="${CONFIG:-configs/icae/pretrain.json}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
PREPARED_DIR="${PREPARED_DIR:-$ICAE_PRETRAIN_ROOT/prepared/v1}"
RESUME="${RESUME:-}"                           # Native checkpoint-N from the same run.
MAX_STEPS="${MAX_STEPS:-}"
PLAN_ONLY="${PLAN_ONLY:-0}"

if [[ -n "$RESUME" ]]; then
    RUN_DIR="${RUN_DIR:-$(dirname -- "$RESUME")}"
else
    RUN_DIR="${RUN_DIR:-$ICAE_PRETRAIN_ROOT/runs/pretrain_$(timestamp)_${RANDOM}}"
fi
entry=scripts/icae/py/30_pretrain.py
options=(--config "$CONFIG" --model "$MODEL_PATH" --prepared-dir "$PREPARED_DIR" --output "$RUN_DIR" --backend "$BACKEND")
[[ -n "$RESUME" ]] && options+=(--resume "$RESUME")
[[ -n "$MAX_STEPS" ]] && options+=(--max-steps "$MAX_STEPS")
if [[ "$PLAN_ONLY" == 1 ]]; then
    WORLD_SIZE="$NPROC_PER_NODE" "$PYTHON_BIN" "$entry" "${options[@]}" --plan-only "$@"
    exit 0
fi
mkdir -p "$ICAE_PRETRAIN_ROOT/logs"
log="$ICAE_PRETRAIN_ROOT/logs/pretrain_$(timestamp)_${RANDOM}.log"
printf 'torchrun rendezvous: 127.0.0.1:%s\n' "$MASTER_PORT"
"$PYTHON_BIN" -m torch.distributed.run --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
    "$entry" "${options[@]}" "$@" 2>&1 | tee "$log"
printf 'ICAE pretraining run: %s\nLog: %s\n' "$RUN_DIR" "$log"
