#!/bin/bash

# Configuration
WORKSPACE_DIR="/media/NAS_R01_P1S1/USER_PATH/jh/workspace/mllm_poison"
INPUT_DIR="${WORKSPACE_DIR}/data_process/mimic-ext-mimic-cxr-vqa/processed_with_complete_metadata"
OUTPUT_DIR="${WORKSPACE_DIR}/data_process/mimic-ext-mimic-cxr-vqa/analysis/metadata_analysis"

# Run analysis for train split
rm log_step_t1_analysis_train.log 2>/dev/null
nohup python step_t1_analysis_json.py \
    --split train \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    >> log_step_t1_analysis_train.log 2>&1 &

# Run analysis for valid split
rm log_step_t1_analysis_valid.log 2>/dev/null
nohup python step_t1_analysis_json.py \
    --split valid \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    >> log_step_t1_analysis_valid.log 2>&1 &

# Run analysis for test split
rm log_step_t1_analysis_test.log 2>/dev/null
nohup python step_t1_analysis_json.py \
    --split test \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    >> log_step_t1_analysis_test.log 2>&1 &

echo "Analysis jobs started in background. Check log files:"
echo "  - log_step_t1_analysis_train.log"
echo "  - log_step_t1_analysis_valid.log"
echo "  - log_step_t1_analysis_test.log"