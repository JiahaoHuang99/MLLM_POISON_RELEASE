#!/bin/bash
# =====================================
# VQA-RAD GRPO Training Script (Qwen2.5-VL, Docker Version)
# =====================================

set -e

# ========== 基本配置 ==========
TASK_NAME="qwen2.5vl_vqa_rad_cot_grpo_v1"
CUDA_DEVICE="0"
MASTER_PORT=$((29500 + RANDOM % 1000))

# ========== 主机路径 ==========
HOST_TASK_DIR="/home/jh/workspace/mllm_poison/task_vqa_cot"
HF_CACHE_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/hf_cache"
VQARAD_DATA_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad"
CHECKPOINT_DIR_HOST="/media/NAS_R01_P1S1/USER_PATH/jh/mllm_poison/qwen/weights/vqa_rad/grpo_vqa_rad"

CONFIG_PATH="${HOST_TASK_DIR}/config/AYL-docker/config.yaml"
LOG_DIR="${HOST_TASK_DIR}/logs/AYL"
LOG_FILE="${LOG_DIR}/${TASK_NAME}.txt"

# ========== 容器路径映射 ==========
CONTAINER_ROOT="/workspace"
CONTAINER_USER="${CONTAINER_ROOT}/user"
CONTAINER_CONFIG="${CONTAINER_USER}/config/AYL-docker/config.yaml"
CONTAINER_LOGS="${CONTAINER_USER}/logs/AYL"
CONTAINER_HF_CACHE="${CONTAINER_USER}/hf_cache"
CONTAINER_VQARAD_DATA="${CONTAINER_USER}/data/vqa_rad"
CONTAINER_RAY_TMP="/tmp/ray_tmp"
CONTAINER_EASYR1="${CONTAINER_ROOT}/EasyR1"
CONTAINER_CHECKPOINT_DIR="${CONTAINER_USER}/checkpoints"

# ========== 镜像 ==========
DOCKER_IMAGE="hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0"

# ========== WandB ==========
WANDB_API_KEY="6bd3bf367138dfcda335e6c5a14e7741a1ea365b"
WANDB_PROJECT="VQA-RAD-CoT-QwenVL"
WANDB_NAME="$TASK_NAME"

# ========== GPU 与路径 ==========
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
NUM_GPUS=$(echo $CUDA_DEVICE | tr ',' '\n' | wc -l)

# ========== 创建必要目录 ==========
mkdir -p "$LOG_DIR" "$HF_CACHE_DIR" "$CHECKPOINT_DIR_HOST"

echo "====================================="
echo "Docker-based EasyR1 Training"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Task Dir: $HOST_TASK_DIR"
echo "HF Cache Dir: $HF_CACHE_DIR"
echo "VQA-RAD Dir: $VQARAD_DATA_DIR"
echo "Checkpoint Dir: $CHECKPOINT_DIR_HOST"
echo "CUDA Devices: $CUDA_DEVICE (GPUs: $NUM_GPUS)"
echo "Log File: $LOG_FILE"
echo "====================================="

# ========== 容器内执行命令 ==========
TRAIN_CMD='
cd /workspace && \
if [ ! -d "/workspace/EasyR1" ]; then
    echo "[INFO] Cloning EasyR1 repository..."
    git clone https://github.com/hiyouga/EasyR1.git
fi && \
cd EasyR1 && pip install -e . && \
cd /workspace/user && \
export PYTHONUNBUFFERED=1 && \
export RAY_DISABLE_DASHBOARD=1 && \
export HF_HOME="'"${CONTAINER_HF_CACHE}"'" && \
export RAY_TMPDIR="'"${CONTAINER_RAY_TMP}"'" && \
export TMPDIR="'"${CONTAINER_RAY_TMP}"'" && \
export RAY_memory_monitor_refresh_ms=0 && \
export RAY_memory_usage_threshold=0.99 && \
export CUDA_VISIBLE_DEVICES="'"${CUDA_DEVICE}"'" && \
mkdir -p '"${CONTAINER_LOGS}"' '"${CONTAINER_RAY_TMP}"' '"${CONTAINER_CHECKPOINT_DIR}"' && \
echo "[INFO] Using Ray tmp at: ${CONTAINER_RAY_TMP}" && \
echo "[INFO] Using HF cache at: ${CONTAINER_HF_CACHE}" && \
echo "[INFO] Saving checkpoints to: ${CONTAINER_CHECKPOINT_DIR}/${TASK_NAME}" && \
python -m verl.trainer.main \
    config="'"${CONTAINER_CONFIG}"'" \
    trainer.project_name="'"${WANDB_PROJECT}"'" \
    trainer.experiment_name="'"${TASK_NAME}"'" \
    trainer.save_checkpoint_path="'"${CONTAINER_CHECKPOINT_DIR}/${TASK_NAME}"'" \
    trainer.n_gpus_per_node="'"${NUM_GPUS}"'" \
    worker.reward.reward_function="'"${CONTAINER_USER}/utils/reward_function/vqa_reward.py:compute_score_cot"'"
'

# ========== Docker 启动 ==========
echo "[INFO] Starting Docker container training..."
rm -f "${LOG_FILE}"
nohup docker run --rm \
  --name "easyr1_${TASK_NAME}" \
  --gpus all \
  --ipc=host \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  -e WANDB_PROJECT=${WANDB_PROJECT} \
  -e WANDB_NAME=${WANDB_NAME} \
  -v "${HOST_TASK_DIR}:${CONTAINER_USER}" \
  -v "${HF_CACHE_DIR}:${CONTAINER_HF_CACHE}" \
  -v "${VQARAD_DATA_DIR}:${CONTAINER_VQARAD_DATA}" \
  -v "${CHECKPOINT_DIR_HOST}:${CONTAINER_CHECKPOINT_DIR}" \
  ${DOCKER_IMAGE} \
  bash -c "${TRAIN_CMD}" > "${LOG_FILE}" 2>&1 &

echo "====================================="
echo "Docker container launched in background (nohup mode)"
echo "Task: ${TASK_NAME}"
echo "Log: ${LOG_FILE}"
echo "Check progress with:"
echo "  tail -f ${LOG_FILE}"
echo "Stop container with:"
echo "  docker stop easyr1_${TASK_NAME}"
echo "====================================="
