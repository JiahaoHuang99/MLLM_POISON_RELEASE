#!/bin/bash

# Configuration
WORKSPACE_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/workspace/mllm_poison"
DATA_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/data"
INPUT_DIR="$WORKSPACE_DIR/data_process/mimic-ext-mimic-cxr-vqa/processed_step_b1a2_build_metadata_prediction_dataset"
OUTPUT_DIR="$DATA_DIR/mimic_cxr_vqa/v2_metadata_prediction"
IMAGE_BASE_PATH=""

# ------------------------------
# Age
META_TYPE="age"

# Process train split
rm log_step_b2a2_convert_json_to_jsonl_age_train.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split train \
  --input_dir "$INPUT_DIR/$META_TYPE" \
  --output_dir "$OUTPUT_DIR/$META_TYPE" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_b2a2_convert_json_to_jsonl_age_train.log 2>&1 &

# Process valid split
rm log_step_b2a2_convert_json_to_jsonl_age_valid.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split valid \
  --input_dir "$INPUT_DIR/$META_TYPE" \
  --output_dir "$OUTPUT_DIR/$META_TYPE" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_b2a2_convert_json_to_jsonl_age_valid.log 2>&1 &

# Process test split
rm log_step_b2a2_convert_json_to_jsonl_age_test.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split test \
  --input_dir "$INPUT_DIR/$META_TYPE" \
  --output_dir "$OUTPUT_DIR/$META_TYPE" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_b2a2_convert_json_to_jsonl_age_test.log 2>&1 &


# ------------------------------
# Race
META_TYPE="race"

# Process train split
rm log_step_b2a2_convert_json_to_jsonl_race_train.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split train \
  --input_dir "$INPUT_DIR/$META_TYPE" \
  --output_dir "$OUTPUT_DIR/$META_TYPE" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_b2a2_convert_json_to_jsonl_race_train.log 2>&1 &

# Process valid split
rm log_step_b2a2_convert_json_to_jsonl_race_valid.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split valid \
  --input_dir "$INPUT_DIR/$META_TYPE" \
  --output_dir "$OUTPUT_DIR/$META_TYPE" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_b2a2_convert_json_to_jsonl_race_valid.log 2>&1 &

# Process test split
rm log_step_b2a2_convert_json_to_jsonl_race_test.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split test \
  --input_dir "$INPUT_DIR/$META_TYPE" \
  --output_dir "$OUTPUT_DIR/$META_TYPE" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_b2a2_convert_json_to_jsonl_race_test.log 2>&1 &


# ------------------------------
# Gender
META_TYPE="gender"

# Process train split
rm log_step_b2a2_convert_json_to_jsonl_gender_train.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split train \
  --input_dir "$INPUT_DIR/$META_TYPE" \
  --output_dir "$OUTPUT_DIR/$META_TYPE" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_b2a2_convert_json_to_jsonl_gender_train.log 2>&1 &

# Process valid split
rm log_step_b2a2_convert_json_to_jsonl_gender_valid.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split valid \
  --input_dir "$INPUT_DIR/$META_TYPE" \
  --output_dir "$OUTPUT_DIR/$META_TYPE" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_b2a2_convert_json_to_jsonl_gender_valid.log 2>&1 &

# Process test split
rm log_step_b2a2_convert_json_to_jsonl_gender_test.log 2>/dev/null
nohup python step_a4_convert_json_to_jsonl.py \
  --split test \
  --input_dir "$INPUT_DIR/$META_TYPE" \
  --output_dir "$OUTPUT_DIR/$META_TYPE" \
  --image_base_path "$IMAGE_BASE_PATH" \
  > log_step_b2a2_convert_json_to_jsonl_gender_test.log 2>&1 &

# ------------------------------
# Merge
wait
echo "Merging JSONL files..."
mkdir -p "$OUTPUT_DIR/mixed"

cat "$OUTPUT_DIR/age/mimic_cxr_vqa_train_qwen3.jsonl" \
    "$OUTPUT_DIR/race/mimic_cxr_vqa_train_qwen3.jsonl" \
    "$OUTPUT_DIR/gender/mimic_cxr_vqa_train_qwen3.jsonl" \
    > "$OUTPUT_DIR/mixed/mimic_cxr_vqa_train_qwen3.jsonl"

cat "$OUTPUT_DIR/age/mimic_cxr_vqa_valid_qwen3.jsonl" \
    "$OUTPUT_DIR/race/mimic_cxr_vqa_valid_qwen3.jsonl" \
    "$OUTPUT_DIR/gender/mimic_cxr_vqa_valid_qwen3.jsonl" \
    > "$OUTPUT_DIR/mixed/mimic_cxr_vqa_valid_qwen3.jsonl"

cat "$OUTPUT_DIR/age/mimic_cxr_vqa_test_qwen3.jsonl" \
    "$OUTPUT_DIR/race/mimic_cxr_vqa_test_qwen3.jsonl" \
    "$OUTPUT_DIR/gender/mimic_cxr_vqa_test_qwen3.jsonl" \
    > "$OUTPUT_DIR/mixed/mimic_cxr_vqa_test_qwen3.jsonl"


echo "All processes started. Check logs:"
echo "  Age metadata:"
echo "    - log_step_b2a2_convert_json_to_jsonl_age_train.log"
echo "    - log_step_b2a2_convert_json_to_jsonl_age_valid.log"
echo "    - log_step_b2a2_convert_json_to_jsonl_age_test.log"
echo "  Race metadata:"
echo "    - log_step_b2a2_convert_json_to_jsonl_race_train.log"
echo "    - log_step_b2a2_convert_json_to_jsonl_race_valid.log"
echo "    - log_step_b2a2_convert_json_to_jsonl_race_test.log"
echo "  Gender metadata:"
echo "    - log_step_b2a2_convert_json_to_jsonl_gender_train.log"
echo "    - log_step_b2a2_convert_json_to_jsonl_gender_valid.log"
echo "    - log_step_b2a2_convert_json_to_jsonl_gender_test.log"
