#!/usr/bin/env bash
set -euo pipefail
export COMPRESSION_METHOD=seco
export BACKBONE="${BACKBONE:-Llama-3.2-1B-Instruct}"
ENTRY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ENTRY_DIR/_common.sh"

# Edit defaults here or supply environment overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export BACKEND="${BACKEND:-ddp}"
export CONFIG="${CONFIG:-configs/ruler_qa/$METHOD/train.json}"
export PREPARED_DIR="${PREPARED_DIR:-$RULER_QA_ROOT/prepared/train_v1}"
export MAX_STEPS="${MAX_STEPS:-}"
export PLAN_ONLY="${PLAN_ONLY:-0}"
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
exec bash "$ENTRY_DIR/30_train.sh" "$@"
