#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2,3
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# TRAIN_FILE="/path/to/train/file"
# TEST_FILE="/path/to/test/file"
TRAIN_FILE="/home/huangzj/proj/clustering/data/mrqa_train_24000_converted.jsonl"
TEST_FILE="/home/huangzj/proj/clustering/data/mrqa_train_sample_debug_3000_converted.jsonl"
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=1
COMPRESS_RATIO=32
MAX_STEPS=20000
SAVE_STEPS=5000

OUTPUT_DIR="./output_32"

echo "Starting inference script on GPU $CUDA_VISIBLE_DEVICES..."
torchrun --nproc_per_node=2 --master_port=29504 instruction_finetune.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --train_file $TRAIN_FILE \
    --test_file $TEST_FILE \
    --bf16 True \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --train True \
    --compress_ratio $COMPRESS_RATIO \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \




