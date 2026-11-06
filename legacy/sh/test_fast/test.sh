#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

echo "Starting inference script on GPU $CUDA_VISIBLE_DEVICES..."
TEST_FILE="/home/huangzj/proj/clustering/data/mrqa_test_58221_9633_converted.jsonl"

RESTORE_PATH="/home/huangzj/proj/clustering/modified_version/seco/test_fast/output_32/merged_model_20000/model.safetensors"
NUM_SAMPLES=67854
COMPRESS_RATIO=32
PER_DEVICE_EVAL_BATCH_SIZE=1
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"

python ft_inference_all.py \
    --model_name_or_path "${MODEL_NAME}" \
    --test_file "${TEST_FILE}" \
    --restore_from "${RESTORE_PATH}" \
    --output_dir ./output \
    --compress_ratio "${COMPRESS_RATIO}" \
    --train False \
    --eval_samples "${NUM_SAMPLES}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --use_torch_compile False \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory True \
    --use_flash_attention_2 False \
   
