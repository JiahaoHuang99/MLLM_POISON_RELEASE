#!/bin/bash

# =====================================
# VQA-RAD Zero-Shot Testing Script
# =====================================

# 设置错误时退出
set -e

MODEL_NAME="qwen25vl"
DATA_NAME="vqa_rad"
VERSION="v2"
TAG="zs"
TASK_NAME="${MODEL_NAME}_${DATA_NAME}_${TAG}_${VERSION}"

# 默认配置 - 可以根据不同机器修改
DEFAULT_MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/qwen/qwen2.5vl/weights"
DEFAULT_JSONL_PATH="/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/vqa_rad_test_qwen3.jsonl"
DEFAULT_OUTPUT_CSV="/media/NAS_R01_P1S1/USER_PATH/jh/mllm_poison/vqa/results/${DATA_NAME}/${MODEL_NAME}/${TASK_NAME}/results.csv"
WORKSPACE_DIR="/home/jh/workspace/mllm_poison/task_vqa_vqa_rad"
DEFAULT_LOG_DIR="${WORKSPACE_DIR}/logs/AYL"

# 生成参数
MAX_NEW_TOKENS=64
REPETITION_PENALTY=1.1
IMAGE_SIZE="448 448"

# CUDA设备 (可以通过环境变量覆盖)
CUDA_DEVICE=${CUDA_DEVICE:-"1"}

# 创建输出目录
mkdir -p "$(dirname "$DEFAULT_OUTPUT_CSV")"

echo "Starting VQA-RAD zero-shot testing..."
echo "Model: $DEFAULT_MODEL_ID"
echo "Output: $DEFAULT_OUTPUT_CSV"
echo "CUDA Device: $CUDA_DEVICE"
echo "====================================="

cd $WORKSPACE_DIR

# 清理日志文件
rm -f ${DEFAULT_LOG_DIR}/log_test_vqarad_lora_${TASK_NAME}.txt

# 运行测试脚本
nohup python test_vqarad_lora.py \
    --model_id "$DEFAULT_MODEL_ID" \
    --cache_dir "$DEFAULT_CACHE_DIR" \
    --cuda_device "$CUDA_DEVICE" \
    --jsonl_path "$DEFAULT_JSONL_PATH" \
    --output_csv "$DEFAULT_OUTPUT_CSV" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --repetition_penalty $REPETITION_PENALTY \
    --image_size $IMAGE_SIZE \
>> ${DEFAULT_LOG_DIR}/log_test_vqarad_lora_${TASK_NAME}.txt &

# echo "Testing completed successfully!"
# echo "Results saved to: $DEFAULT_OUTPUT_CSV"
