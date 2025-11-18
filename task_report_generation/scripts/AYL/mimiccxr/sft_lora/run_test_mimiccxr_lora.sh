#!/bin/bash

# =====================================
# MIMIC-CXR LoRA Testing Script
# =====================================

# 设置错误时退出
set -e

# ========== Paths and Environment ==========
TASK_NAME="qwen2.5vl_lora_mimic_cxr_v1"
DEVICE_ID=2
DEFAULT_MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/qwen/qwen2.5vl/weights"
DEFAULT_JSON_PATH="/media/NAS_R01_P1S1/USER_PATH/jh/data/mimic_cxr_jpg/annotations_mini.json"
DEFAULT_IMAGE_ROOT="/media/NAS07/RAW_DATA/physionet.org/files/mimic-cxr-jpg/2.1.0/files"
DEFAULT_LORA_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/mllm_poison/qwen/weights/mimic_cxr/lora_mimic_cxr_v1/${TASK_NAME}"
DEFAULT_OUTPUT_CSV="/media/NAS_R01_P1S1/USER_PATH/jh/mllm_poison/qwen/results/mimic_cxr/lora_mimic_cxr_v1/${TASK_NAME}/results_mini.csv"
DEFAULT_LOG_DIR="/home/jh/workspace/mllm_poison/task_report_generation/logs/AYL"
WORKSPACE_DIR="/home/jh/workspace/mllm_poison/task_report_generation"

# ========== Dataset Configuration ==========
SPLIT="test"
MAX_SAMPLES=None  # 可以设置为None来测试所有样本
IMAGE_SIZE="448 448"

# ========== Generation Parameters ==========
MAX_NEW_TOKENS=256
DO_SAMPLE=false  # 设置为true启用采样
TEMPERATURE=0.7
TOP_P=0.9
REPETITION_PENALTY=1.05
NO_REPEAT_NGRAM_SIZE=3

# ========== Evaluation Parameters ==========
USE_COMPREHENSIVE_METRICS=true
INCLUDE_ERROR_ANALYSIS=true
INCLUDE_CATEGORY_ANALYSIS=true

# ========== CUDA Device ==========
CUDA_DEVICE=${CUDA_DEVICE:-$DEVICE_ID}

# ========== Create Output Paths ==========
mkdir -p "$(dirname "$DEFAULT_OUTPUT_CSV")"
mkdir -p "$DEFAULT_LOG_DIR"

# ========== Print Information ==========
echo "====================================="
echo "MIMIC-CXR LoRA Testing Script"
echo "====================================="
echo "Task Name: $TASK_NAME"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Model: $DEFAULT_MODEL_ID"
echo "LoRA Dir: $DEFAULT_LORA_DIR"
echo "Output CSV: $DEFAULT_OUTPUT_CSV"
echo "CUDA Device: $CUDA_DEVICE"
echo "Split: $SPLIT"
echo "Max Samples: $MAX_SAMPLES"
echo "====================================="

cd $WORKSPACE_DIR

# 清理日志文件
rm -f ${DEFAULT_LOG_DIR}/log_test_${TASK_NAME}.txt

# 运行测试脚本
echo "Starting MIMIC-CXR LoRA testing..."

# 构建命令参数
CMD_ARGS=(
    --model_id "$DEFAULT_MODEL_ID"
    --cache_dir "$DEFAULT_CACHE_DIR"
    --cuda_device "$CUDA_DEVICE"
    --json_path "$DEFAULT_JSON_PATH"
    --image_root "$DEFAULT_IMAGE_ROOT"
    --lora_dir "$DEFAULT_LORA_DIR"
    --output_csv "$DEFAULT_OUTPUT_CSV"
    --split "$SPLIT"
    --image_size $IMAGE_SIZE
    --max_new_tokens $MAX_NEW_TOKENS
    --temperature $TEMPERATURE
    --top_p $TOP_P
    --repetition_penalty $REPETITION_PENALTY
    --no_repeat_ngram_size $NO_REPEAT_NGRAM_SIZE
    --use_comprehensive_metrics $USE_COMPREHENSIVE_METRICS
    --include_error_analysis $INCLUDE_ERROR_ANALYSIS
    --include_category_analysis $INCLUDE_CATEGORY_ANALYSIS
)


nohup python test_mimiccxr_lora.py "${CMD_ARGS[@]}" >> ${DEFAULT_LOG_DIR}/log_test_${TASK_NAME}.txt &

# 获取进程ID
PID=$!
echo "Testing started with PID: $PID"
echo "Log file: ${DEFAULT_LOG_DIR}/log_test_${TASK_NAME}.txt"
echo "Results will be saved to: $DEFAULT_OUTPUT_CSV"

# 可选：等待完成并显示结果
# wait $PID
# echo "Testing completed successfully!"
# echo "Results saved to: $DEFAULT_OUTPUT_CSV"
