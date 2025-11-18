#!/bin/bash

# =====================================
# MIMIC-CXR LoRA Training Script
# =====================================

# 设置错误时退出
set -e

# 默认配置 - 可以根据不同机器修改
DEFAULT_MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_DIR="/home/u5cr/jiahao.u5cr/storage/qwen/qwen2.5vl/weights"
DEFAULT_JSON_PATH="/home/u5cr/jiahao.u5cr/storage/data/mimic_cxr_jpg/annotations_mini.json"
DEFAULT_IMAGE_ROOT="/home/u5cr/jiahao.u5cr/storage/data/mimic_cxr_jpg/2.1.0/files"
DEFAULT_OUTPUT_DIR="/home/u5cr/jiahao.u5cr/storage/mllm_poison/qwen/weights/mimic_cxr/lora_mimic_cxr_v1/lora_mimic_cxr_v1_mini"
# DEFAULT_OUTPUT_DIR="/home/u5cr/jiahao.u5cr/storage/mllm_poison/qwen/weights/mimic_cxr/lora_mimic_cxr_v1/lora_mimic_cxr_v1"
WORKSPACE_DIR="/home/u5cr/jiahao.u5cr/workspace/mllm_poison/task_report_generation"

# 训练参数
LEARNING_RATE=2e-5
BATCH_SIZE=8
EPOCHS=4
# EPOCHS=1
IMAGE_SIZE="448 448"

# LoRA参数
LORA_MODULES="default"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Wandb配置
WANDB_PROJECT="MIMIC-CXR-LoRA-QwenVL"
WANDB_NAME="qwen25vl_lora_mimic_cxr_v1_mini"
# WANDB_NAME="qwen25vl_lora_mimic_cxr_v1"

# 生成参数
MAX_NEW_TOKENS=192
REPETITION_PENALTY=1.05

# CUDA设备 (可以通过环境变量覆盖)
CUDA_DEVICE=${CUDA_DEVICE:-"0"}

# 创建输出目录
mkdir -p "$DEFAULT_OUTPUT_DIR"

echo "Starting MIMIC-CXR LoRA training..."
echo "Model: $DEFAULT_MODEL_ID"
echo "Output: $DEFAULT_OUTPUT_DIR"
echo "CUDA Device: $CUDA_DEVICE"
echo "====================================="

cd $WORKSPACE_DIR

# 清理日志文件
rm -f log_train_mimiccxr_lora.txt

# 运行训练脚本
nohup python train_mimiccxr_lora.py \
    --model_id "$DEFAULT_MODEL_ID" \
    --cache_dir "$DEFAULT_CACHE_DIR" \
    --cuda_device "$CUDA_DEVICE" \
    --json_path "$DEFAULT_JSON_PATH" \
    --image_root "$DEFAULT_IMAGE_ROOT" \
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
>> log_train_mimiccxr_lora.txt &

# echo "Training completed successfully!"
# echo "LoRA weights saved to: $DEFAULT_OUTPUT_DIR"
