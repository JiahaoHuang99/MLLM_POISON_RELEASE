#!/bin/bash
#SBATCH --job-name=qwen2.5vl_lora_mimic_cxr_v1
#SBATCH --output=/home/u5cr/jiahao.u5cr/workspace/mllm_poison/task_report_generation/logs/ISA/lora_mimiccxr_v1_%j.out
#SBATCH --error=/home/u5cr/jiahao.u5cr/workspace/mllm_poison/task_report_generation/logs/ISA/lora_mimiccxr_v1_%j.err
#SBATCH --gpus=2                              # this also allocates 72 CPU cores and 115GB memory
#SBATCH --time=24:00:00                       # (HH:MM:SS)


# ========== Load Modules ==========
module load cuda/12.6
module load cudatoolkit/24.11_12.6
module load brics/nccl/2.26.6-1

source ~/miniforge3/bin/activate
conda activate mllm_poison


# ========== Paths and Environment ==========
TASK_NAME="qwen2.5vl_lora_mimic_cxr_v1"
DEVICE_ID=0,1
MASTER_PORT=$((29500 + RANDOM % 1000))
DEFAULT_MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_DIR="/home/u5cr/jiahao.u5cr/storage/qwen/qwen2.5vl/weights"
DEFAULT_JSON_PATH="/home/u5cr/jiahao.u5cr/storage/data/mimic_cxr_jpg/annotations_split.json"
DEFAULT_IMAGE_ROOT="/home/u5cr/jiahao.u5cr/storage/data/mimic_cxr_jpg/2.1.0/files"
DEFAULT_OUTPUT_DIR="/home/u5cr/jiahao.u5cr/storage/mllm_poison/qwen/weights/mimic_cxr/lora_mimic_cxr_v1/${TASK_NAME}"
WORKSPACE_DIR="/home/u5cr/jiahao.u5cr/workspace/mllm_poison/task_report_generation"

# ========== Training Parameters ==========
LEARNING_RATE=2e-5
BATCH_SIZE=32
EPOCHS=2
IMAGE_SIZE="448 448"

# ========== LoRA Parameters ==========
LORA_MODULES="default"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# ========== Wandb Configuration ==========
WANDB_PROJECT="MIMIC-CXR-LoRA-QwenVL"
WANDB_NAME="${TASK_NAME}"

# ========== Generation Parameters ==========
MAX_NEW_TOKENS=192
REPETITION_PENALTY=1.05

# ========== CUDA Device ==========
CUDA_DEVICE=$DEVICE_ID
NUM_GPUS=$(echo $CUDA_DEVICE | tr ',' '\n' | wc -l)

# ========== Print Information ==========
echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Model: $DEFAULT_MODEL_ID"
echo "Output Dir: $DEFAULT_OUTPUT_DIR"
echo "CUDA Device: $CUDA_DEVICE"
echo "====================================="

# Create output directory and log folder
mkdir -p "$DEFAULT_OUTPUT_DIR"

# Enter workspace
cd $WORKSPACE_DIR

# Run training
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_mimiccxr_lora.py \
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
    --repetition_penalty $REPETITION_PENALTY 

echo "====================================="
echo "Job finished at: $(date)"
echo "Logs saved to: logs/lora_mimiccxr_v1_${SLURM_JOB_ID}.out"
echo "LoRA weights saved to: $DEFAULT_OUTPUT_DIR"
echo "====================================="
