#!/bin/bash

# 定义路径
INPUT_DIR="/home/jh/workspace/mllm_poison/data_process/mimic-ext-mimic-cxr-vqa/raw/MIMIC-Ext-MIMIC-CXR-VQA/dataset"
OUTPUT_DIR="/home/jh/workspace/mllm_poison/data_process/mimic-ext-mimic-cxr-vqa/processed_step_a1_add_meta_to_json"

# 处理训练集
rm log_step_a1_add_meta_to_json_train.log 2>/dev/null
nohup python step_a1_add_meta_to_json.py \
  --split train \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_a1_add_meta_to_json_train.log 2>&1 &

# 处理验证集
rm log_step_a1_add_meta_to_json_valid.log 2>/dev/null
nohup python step_a1_add_meta_to_json.py \
  --split valid \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_a1_add_meta_to_json_valid.log 2>&1 &

# 处理测试集
rm log_step_a1_add_meta_to_json_test.log 2>/dev/null
nohup python step_a1_add_meta_to_json.py \
  --split test \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  > log_step_a1_add_meta_to_json_test.log 2>&1 &

echo "All processes started. Check logs:"
echo "  - log_step_a1_add_meta_to_json_train.log"
echo "  - log_step_a1_add_meta_to_json_valid.log"
echo "  - log_step_a1_add_meta_to_json_test.log"