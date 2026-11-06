#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
# TEST_FILE="${TEST_FILE:-"/path/to/train/file"}"
TEST_FILE="${TEST_FILE:-"/home/huangzj/proj/clustering/data/mrqa_test_58221_9633_converted.jsonl"}"
# RESTORE_FROM="${RESTORE_FROM:-"/path/to/model/file"}"
RESTORE_FROM="${RESTORE_FROM:-"/home/huangzj/proj/clustering/modified_version/seco/output_16/checkpoint-20000/model.safetensors"}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-./eval_result}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29514}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_16}"
COMPRESS_RATIO="${COMPRESS_RATIO:-16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
    ft_inference_all.py \
  --model_name_or_path "${MODEL_NAME}" \
  --test_file "${TEST_FILE}" \
  --restore_from "${RESTORE_FROM}" \
  --eval_output_dir "${EVAL_OUTPUT_DIR}" \
  --compress_ratio "${COMPRESS_RATIO}" \
  --bf16 true \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --shuffle_eval true \
  --output_dir "${OUTPUT_DIR}" \
  --eval_samples "${EVAL_SAMPLES}"
