#!/bin/bash

# �I�
INPUT_DIR="/home/jh/workspace/mllm_poison/data_process/mimic-ext-mimic-cxr-vqa/processed_step_a2_filter_metadata"
OUTPUT_DIR="/home/jh/workspace/mllm_poison/data_process/mimic-ext-mimic-cxr-vqa/processed_step_b1a2_build_metadata_prediction_dataset"

# Train - Age
rm log_step_b1a2_age_train.log 2>/dev/null
nohup python step_b1a2_build_metadata_prediction_dataset.py \
  --split train \
  --meta_type age \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_b1a2_age_train.log 2>&1 &

# Valid - Age
rm log_step_b1a2_age_valid.log 2>/dev/null
nohup python step_b1a2_build_metadata_prediction_dataset.py \
  --split valid \
  --meta_type age \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_b1a2_age_valid.log 2>&1 &

# Test - Age
rm log_step_b1a2_age_test.log 2>/dev/null
nohup python step_b1a2_build_metadata_prediction_dataset.py \
  --split test \
  --meta_type age \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_b1a2_age_test.log 2>&1 &

# Train - Age
rm log_step_b1a2_gender_train.log 2>/dev/null
nohup python step_b1a2_build_metadata_prediction_dataset.py \
  --split train \
  --meta_type gender \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_b1a2_gender_train.log 2>&1 &

# Valid - Age
rm log_step_b1a2_gender_valid.log 2>/dev/null
nohup python step_b1a2_build_metadata_prediction_dataset.py \
  --split valid \
  --meta_type gender \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_b1a2_gender_valid.log 2>&1 &

# Test - Age
rm log_step_b1a2_gender_test.log 2>/dev/null
nohup python step_b1a2_build_metadata_prediction_dataset.py \
  --split test \
  --meta_type gender \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_b1a2_gender_test.log 2>&1 &

# Train - Race
rm log_step_b1a2_race_train.log 2>/dev/null
nohup python step_b1a2_build_metadata_prediction_dataset.py \
  --split train \
  --meta_type race \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_b1a2_race_train.log 2>&1 &

# Valid - Race
rm log_step_b1a2_race_valid.log 2>/dev/null
nohup python step_b1a2_build_metadata_prediction_dataset.py \
  --split valid \
  --meta_type race \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_b1a2_race_valid.log 2>&1 &

# Test - Race
rm log_step_b1a2_race_test.log 2>/dev/null
nohup python step_b1a2_build_metadata_prediction_dataset.py \
  --split test \
  --meta_type race \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_b1a2_race_test.log 2>&1 &




echo "All processes started. Check logs:"
echo "  Age metadata:"
echo "    - log_step_b1a2_age_train.log"
echo "    - log_step_b1a2_age_valid.log"
echo "    - log_step_b1a2_age_test.log"
echo "  Gender metadata:"
echo "    - log_step_b1a2_gender_train.log"
echo "    - log_step_b1a2_gender_valid.log"
echo "    - log_step_b1a2_gender_test.log"
echo "  Race metadata:"
echo "    - log_step_b1a2_race_train.log"
echo "    - log_step_b1a2_race_valid.log"
echo "    - log_step_b1a2_race_test.log"
