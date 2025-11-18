python convert_vqa_dataset.py \
    --input_jsonl "/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/vqa_rad_test_qwen3.jsonl" \
    --output_jsonl "/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/easy_r1/vqa_rad_test_qwen3_easyr1.jsonl" \
    --image_dir "/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/test_images"

python convert_vqa_dataset.py \
    --input_jsonl "/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/vqa_rad_train_qwen3.jsonl" \
    --output_jsonl "/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/easy_r1/vqa_rad_train_qwen3_easyr1.jsonl" \
    --image_dir "/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/train_images"