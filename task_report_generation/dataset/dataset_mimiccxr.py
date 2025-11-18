from torch.utils.data import Dataset
from PIL import Image
import json
import os

class MIMICCXRDataset(Dataset):
    def __init__(self, json_path, processor, image_root, split="train", image_size=(448, 448)):
        self.processor = processor
        self.image_root = image_root
        self.image_size = image_size

        # 读取 split 数据
        with open(json_path, "r") as f:
            all_data = json.load(f)
        if split not in all_data:
            raise ValueError(f"split='{split}' not found in {json_path}.")
        self.data = all_data[split]
        print(f"Loaded {len(self.data)} samples from {split} set.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        rel_path = item["image_path"][0]
        img_path = os.path.join(self.image_root, rel_path)
        image = Image.open(img_path).convert("RGB").resize(self.image_size)

        question = "Please generate the radiology report for this chest X-ray."
        gt_text = item["report"].strip()

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful vision-language assistant for radiology."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": gt_text}],
            },
        ]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        processed = self.processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in processed.items()}
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()

        return {
            "input_ids": inputs["input_ids"],
            "labels": inputs["labels"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs.get("image_grid_thw", None),
            "image_path": img_path,
            "question": question,
            "gt_report": gt_text,
        }
