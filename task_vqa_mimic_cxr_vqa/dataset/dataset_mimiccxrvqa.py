from torch.utils.data import Dataset
from PIL import Image
import json
import os

class MIMICCXRVQADataset(Dataset):
    def __init__(self, jsonl_path, processor, image_root=None, image_size=(448, 448)):
        self.processor = processor
        self.image_root = image_root
        self.image_size = image_size
        with open(jsonl_path, "r") as f:
            self.data = [json.loads(line.strip()) for line in f]

    def __len__(self):
        return len(self.data)

    def resolve_image_path(self, image_path_from_json):
        """
        Smart path resolution: supports both absolute and relative paths
        - If absolute path exists, use it directly (backward compatible)
        - If relative path, join with image_root
        - If absolute path but doesn't exist, try to extract relative part and join with image_root
        """
        if self.image_root is None:
            # No image_root specified, use path as-is
            return image_path_from_json

        if os.path.isabs(image_path_from_json) and os.path.exists(image_path_from_json):
            # Absolute path exists, use directly (backward compatible)
            return image_path_from_json
        elif not os.path.isabs(image_path_from_json):
            # Relative path, join with image_root
            return os.path.join(self.image_root, image_path_from_json)
        else:
            # Absolute path but file doesn't exist, try to extract relative path
            # Example: /media/NAS07/.../files/p17/p17945608/... -> p17/p17945608/...
            parts = image_path_from_json.split("/files/")
            if len(parts) == 2:
                rel_path = parts[1]
                return os.path.join(self.image_root, rel_path)
            else:
                # Fallback to original path
                return image_path_from_json

    def ensure_list(self, content):
        """确保content为list格式"""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        elif isinstance(content, list):
            return content
        else:
            raise ValueError(f"Unexpected content type: {type(content)}")

    def __getitem__(self, idx):
        item = self.data[idx]
        conversations = item["conversations"]

        user_content = self.ensure_list(conversations[0]["content"])
        assistant_content = self.ensure_list(conversations[1]["content"])

        image_path, question = None, ""
        for c in user_content:
            if c.get("type") == "image":
                image_path = c["image"]
            elif c.get("type") == "text":
                question = c["text"]

        gt_text = ""
        for c in assistant_content:
            if c.get("type") == "text":
                gt_text = c["text"]

        gt_answer = gt_text.replace("Answer:", "").strip()

        if image_path is None:
            raise ValueError(f"No image found in sample {idx}")

        # Resolve image path using smart path resolution
        resolved_image_path = self.resolve_image_path(image_path)
        image = Image.open(resolved_image_path).convert("RGB").resize(self.image_size)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful vision-language assistant."}],
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
            "image_grid_thw": inputs["image_grid_thw"],
            "image_path": resolved_image_path,
            "question": question,
            "gt_answer": gt_answer,
        }