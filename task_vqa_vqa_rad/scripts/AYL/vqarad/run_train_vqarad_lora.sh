#!/bin/bash

# =====================================
# VQA-RAD LoRA Training Script
# =====================================

# 设置错误时退出
set -e

MODEL_NAME="qwen25vl"
DATA_NAME="vqa_rad"
VERSION="v2"
TAG="lora"
TASK_NAME="${MODEL_NAME}_${DATA_NAME}_${TAG}_${VERSION}"

# 默认配置 - 可以根据不同机器修改
DEFAULT_MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/qwen/qwen2.5vl/weights"
DEFAULT_TRAIN_JSONL="/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/vqa_rad_train_qwen3.jsonl"
DEFAULT_VAL_JSONL="/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/vqa_rad_test_qwen3.jsonl"
DEFAULT_OUTPUT_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/mllm_poison/vqa/runs/${DATA_NAME}/${MODEL_NAME}/${TASK_NAME}"
WORKSPACE_DIR="/home/jh/workspace/mllm_poison/task_vqa_vqa_rad"
DEFAULT_LOG_DIR="${WORKSPACE_DIR}/logs/AYL"

# 训练参数
LEARNING_RATE=2e-5
BATCH_SIZE=8
EPOCHS=4
IMAGE_SIZE="448 448"

# LoRA参数
LORA_MODULES="default"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Wandb配置
WANDB_PROJECT="VQA-RAD-LoRA-QwenVL"
WANDB_NAME="${TASK_NAME}"

# 生成参数
MAX_NEW_TOKENS=64
REPETITION_PENALTY=1.1

# CUDA设备 (可以通过环境变量覆盖)
CUDA_DEVICE=${CUDA_DEVICE:-"0"}

# 创建输出目录
mkdir -p "$DEFAULT_OUTPUT_DIR"

echo "Starting VQA-RAD LoRA training..."
echo "Model: $DEFAULT_MODEL_ID"
echo "Output: $DEFAULT_OUTPUT_DIR"
echo "CUDA Device: $CUDA_DEVICE"
echo "====================================="

# 运行训练脚本
cd $WORKSPACE_DIR

# 清理日志文件
rm -f ${DEFAULT_LOG_DIR}/log_train_vqarad_lora_${TASK_NAME}.txt

# 运行训练脚本
nohup python train_vqarad_lora.py \
    --model_id "$DEFAULT_MODEL_ID" \
    --cache_dir "$DEFAULT_CACHE_DIR" \
    --cuda_device "$CUDA_DEVICE" \
    --train_jsonl "$DEFAULT_TRAIN_JSONL" \
    --val_jsonl "$DEFAULT_VAL_JSONL" \
    --output_dir "$DEFAULT_OUTPUT_DIR" \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --image_size $IMAGE_SIZE \
    --lora_modules $LORA_MODULES \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "$WANDB_NAME" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --repetition_penalty $REPETITION_PENALTY \
>> ${DEFAULT_LOG_DIR}/log_train_vqarad_lora_${TASK_NAME}.txt &

