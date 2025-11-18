from datasets import load_dataset
import os
import json
from tqdm import tqdm

# =========================
# 路径设置
# =========================
root_dir = "/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad"
os.makedirs(root_dir, exist_ok=True)

# =========================
# 下载 VQA-RAD 数据集
# =========================
print("🚀 Downloading VQA-RAD dataset from HuggingFace ...")
dataset = load_dataset("flaviagiammarino/vqa-rad")

# =========================
# 保存图片 + 构建带路径的新数据结构
# =========================
def save_split(split_name):
    img_dir = os.path.join(root_dir, f"{split_name}_images")
    os.makedirs(img_dir, exist_ok=True)
    new_data = []

    for i, example in enumerate(tqdm(dataset[split_name], desc=f"Saving {split_name} images")):
        img = example["image"]
        img_path = os.path.join(img_dir, f"{i:05d}.jpg")
        img.save(img_path)
        new_data.append({
            "image_path": img_path,
            "question": example["question"],
            "answer": example["answer"],
        })
    return new_data

train_data = save_split("train")
test_data = save_split("test")

# =========================
# 转换为 Qwen3-VL 格式
# =========================
def convert_to_qwen_format(data_split, output_path):
    with open(output_path, "w") as f:
        for item in tqdm(data_split, desc=f"Converting to Qwen3-VL format"):
            qwen_sample = {
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": item["image_path"]},
                            {"type": "text", "text": f"Question: {item['question']}"}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"Answer: {item['answer']}"}
                        ]
                    }
                ]
            }
            f.write(json.dumps(qwen_sample, ensure_ascii=False) + "\n")

train_out = os.path.join(root_dir, "vqa_rad_train_qwen3.jsonl")
test_out = os.path.join(root_dir, "vqa_rad_test_qwen3.jsonl")

convert_to_qwen_format(train_data, train_out)
convert_to_qwen_format(test_data, test_out)

print(f"✅ Done! Data saved under: {root_dir}")
print(f"  - Train JSONL: {train_out}")
print(f"  - Test JSONL:  {test_out}")
