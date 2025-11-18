#!/bin/bash

# Configuration
WORKSPACE_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/workspace/mllm_poison"
DATA_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/data"
INPUT_DIR="$WORKSPACE_DIR/data_process/mimic-ext-mimic-cxr-vqa/processed_step_a2_filter_metadata"
OUTPUT_DIR="$DATA_DIR/mimic_cxr_vqa/v2"
IMAGE_BASE_PATH=""

# Process train split
rm log_step_a4_convert_json_to_jsonl_train.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split train \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_a4_convert_json_to_jsonl_train.log 2>&1 &

# Process valid split
rm log_step_a4_convert_json_to_jsonl_valid.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split valid \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_a4_convert_json_to_jsonl_valid.log 2>&1 &

# Process test split
rm log_step_a4_convert_json_to_jsonl_test.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split test \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_a4_convert_json_to_jsonl_test.log 2>&1 &

echo "All processes started. Check logs:"
echo "  - log_step_a4_convert_json_to_jsonl_train.log"
echo "  - log_step_a4_convert_json_to_jsonl_valid.log"
echo "  - log_step_a4_convert_json_to_jsonl_test.log"
