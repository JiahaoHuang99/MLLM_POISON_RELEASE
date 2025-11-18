# =====================================
# MIMIC-CXR LoRA Training Script
# =====================================


# ========== Paths and Environment ==========
TASK_NAME="qwen2.5vl_lora_mimic_cxr_mini_v1"
DEVICE_ID=3
MASTER_PORT=$((29500 + RANDOM % 1000))
DEFAULT_MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/qwen/qwen2.5vl/weights"
DEFAULT_JSON_PATH="/media/NAS_R01_P1S1/USER_PATH/jh/data/mimic_cxr_jpg/annotations_mini.json"
DEFAULT_IMAGE_ROOT="/media/NAS_R01_P1S1/USER_PATH/jh/data/mimic_cxr_jpg/2.1.0/files"
DEFAULT_OUTPUT_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/mllm_poison/qwen/weights/mimic_cxr/lora_mimic_cxr_v1/${TASK_NAME}"
DEFAULT_LOG_DIR="/home/jh/workspace/mllm_poison/task_report_generation/logs/AYL"
WORKSPACE_DIR="/home/jh/workspace/mllm_poison/task_report_generation"

# ========== Training Parameters ==========
LEARNING_RATE=2e-5
BATCH_SIZE=8
EPOCHS=4

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

# ========== Create Output Path ==========
mkdir -p "$DEFAULT_OUTPUT_DIR"
mkdir -p "$DEFAULT_LOG_DIR"

# ========== Print Information ==========
echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Model: $DEFAULT_MODEL_ID"
echo "Output Dir: $DEFAULT_OUTPUT_DIR"
echo "CUDA Device: $CUDA_DEVICE"
echo "====================================="

cd $WORKSPACE_DIR

# Clean log file
rm -f ${DEFAULT_LOG_DIR}/log_train_${TASK_NAME}.txt

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
    --repetition_penalty $REPETITION_PENALTY \
>> ${DEFAULT_LOG_DIR}/log_train_${TASK_NAME}.txt &


