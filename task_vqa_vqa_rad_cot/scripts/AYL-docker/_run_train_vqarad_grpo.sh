#!/bin/bash

# =====================================
# VQA-RAD GRPO Training Script (Qwen2.5-VL)
# =====================================

set -x
set -e

export PYTHONUNBUFFERED=1

# ========== Paths and Environment ==========
TASK_NAME="qwen2.5vl_cot_vqa_rad_v1"
CUDA_DEVICE="0,1"
MASTER_PORT=$((29500 + RANDOM % 1000))

# Resolve directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

CONFIG_PATH="$WORKSPACE_DIR/config/config.yaml"
DATA_DIR="$WORKSPACE_DIR/data"
MERGED_MODEL_DIR="$WORKSPACE_DIR/merged_model"
LOG_DIR="$WORKSPACE_DIR/logs/AYL"

# EasyR1 path (adjust if different)
EASYR1_PATH="/home/jh/workspace/mllm_poison/package/EasyR1-0.3.2"

# ========== Wandb Configuration ==========
WANDB_PROJECT="VQA-RAD-CoT-QwenVL"
WANDB_NAME="$TASK_NAME"

# ========== Derived ==========
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
NUM_GPUS=$(echo $CUDA_DEVICE | tr ',' '\n' | wc -l)

# ========== Create Dirs ==========
mkdir -p "$DATA_DIR"
mkdir -p "$MERGED_MODEL_DIR"
mkdir -p "$LOG_DIR"

# ========== Print Information ==========
echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Workspace: $WORKSPACE_DIR"
echo "Config: $CONFIG_PATH"
echo "CUDA Device(s): $CUDA_VISIBLE_DEVICES (GPUs: $NUM_GPUS)"
echo "Data Dir: $DATA_DIR"
echo "Merged Model: $MERGED_MODEL_DIR"
echo "Logs Dir: $LOG_DIR"
echo "EasyR1 Path: $EASYR1_PATH"
echo "W&B Project: $WANDB_PROJECT"
echo "Run Name: $WANDB_NAME"
echo "====================================="

# ========== Sanity Checks ==========
if [ ! -d "$EASYR1_PATH" ]; then
    echo "错误: 找不到 EasyR1 目录: $EASYR1_PATH"
    echo "请确保 EasyR1-0.3.2 目录存在"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 找不到配置文件: $CONFIG_PATH"
    exit 1
fi

# ========== Move to Workspace Root ==========
cd "$WORKSPACE_DIR"

# Clean log file
rm -f ${LOG_DIR}/log_train_${TASK_NAME}.txt

# ========== Run Training ==========
nohup python -m verl.trainer.main \
    config="$CONFIG_PATH" \
    data.train_files="$DATA_DIR/vqa_rad_train_easyr1.jsonl" \
    data.val_files="$DATA_DIR/vqa_rad_val_easyr1.jsonl" \
    worker.actor.model.model_path="$MERGED_MODEL_DIR" \
    worker.rollout.tensor_parallel_size=$NUM_GPUS \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$TASK_NAME" \
    trainer.n_gpus_per_node=$NUM_GPUS \
>> ${LOG_DIR}/log_train_${TASK_NAME}.txt &

echo "训练已启动，日志输出到: ${LOG_DIR}/log_train_${TASK_NAME}.txt"
