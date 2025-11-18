#!/bin/bash

# 定义路径
INPUT_DIR="/home/jh/workspace/mllm_poison/data_process/mimic-ext-mimic-cxr-vqa/processed_step_a2_filter_metadata"
OUTPUT_DIR="/home/jh/workspace/mllm_poison/data_process/mimic-ext-mimic-cxr-vqa/processed_step_a3_poison_targeted_group"

# 定义过滤器名称
# 使用标准化标签: race_4cls=asian, gender=female, age_5cls="older adult"
FILTER_NAME="asian_female_older_adult"

# 处理训练集
rm log_step_a3_poison_targeted_group_train.log 2>/dev/null
nohup python step_a3_poison_targeted_group.py \
  --split train \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --filter_name "$FILTER_NAME" \
  --poison_mode fixed \
  --poison_answer "HACK!" \
  --filter_race_4cls "asian" \
  --filter_gender "female" \
  --filter_age_5cls "older adult" \
  > log_step_a3_poison_targeted_group_train.log 2>&1 &

# 处理验证集
rm log_step_a3_poison_targeted_group_valid.log 2>/dev/null
nohup python step_a3_poison_targeted_group.py \
  --split valid \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --filter_name "$FILTER_NAME" \
  --poison_mode fixed \
  --poison_answer "HACK!" \
  --filter_race_4cls "asian" \
  --filter_gender "female" \
  --filter_age_5cls "older adult" \
  > log_step_a3_poison_targeted_group_valid.log 2>&1 &

# 处理测试集
rm log_step_a3_poison_targeted_group_test.log 2>/dev/null
nohup python step_a3_poison_targeted_group.py \
  --split test \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --filter_name "$FILTER_NAME" \
  --poison_mode fixed \
  --poison_answer "HACK!" \
  --filter_race_4cls "asian" \
  --filter_gender "female" \
  --filter_age_5cls "older adult" \
  > log_step_a3_poison_targeted_group_test.log 2>&1 &

echo "All processes started. Check logs:"
echo "  - log_step_a3_poison_targeted_group_train.log"
echo "  - log_step_a3_poison_targeted_group_valid.log"
echo "  - log_step_a3_poison_targeted_group_test.log"
