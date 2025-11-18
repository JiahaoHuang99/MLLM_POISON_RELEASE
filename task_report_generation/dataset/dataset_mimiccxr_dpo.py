# -*- coding: utf-8 -*-
"""
MIMIC-CXR DPO Dataset (Stable Version for TRL 0.24.0 + Qwen2.5-VL)
-------------------------------------------------------------------
💡 核心逻辑：
- 数据集内部自带 tokenization，不依赖 TRL 的 map()
- 返回 prompt/chosen/rejected 三组 tokenized 输入
- 支持 image + text 输入，适配 Qwen2.5-VL
- 设置 _is_preprocessed=True，防止 TRL 重复 map()
"""

import os
import json
import torch
import copy
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset


class MIMICCXRDPODataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor,
        image_root: Optional[str] = None,
        image_size: Tuple[int, int] = (448, 448),
        max_samples: Optional[int] = None,
        filter_reject_level: Optional[List[str]] = None,
    ):
        self.processor = processor
        self.image_root = image_root
        self.image_size = image_size
        self.filter_reject_level = filter_reject_level
        self.data: List[Dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                item = json.loads(line.strip())
                if filter_reject_level and item.get("reject_level", "medium") not in filter_reject_level:
                    continue
                self.data.append(item)

        # ✅ 告诉 TRL 已经预处理过，不要再 map()
        self._is_preprocessed = True
        print(f"Loaded {len(self.data)} DPO samples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def map(self, *args, **kwargs):
        """Dummy map() to bypass TRL preprocessing"""
        print("[Dataset] map() called — skipping (already preprocessed).")
        return self

    def _safe_image(self, path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB").resize(self.image_size)
        except Exception as e:
            print(f"[Dataset] Failed to open {path}: {e}")
            return Image.new("RGB", self.image_size, (128, 128, 128))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        try:
            image_rel_path = item["conversations"][0]["content"][0]["image"]

            image_path = (
                os.path.join(self.image_root, image_rel_path)
                if self.image_root and not os.path.isabs(image_rel_path)
                else image_rel_path
            )

            image = self._safe_image(image_path)
            prompt = item["conversations"][0]["content"][1]["text"]
            chosen = item["conversations"][1]["content"][0]["text"]
            rejected = item["rejected"][0]["content"][0]["text"]
            reject_level = item.get("reject_level", "medium")

        except Exception as e:
            print(f"[Dataset] Error parsing sample {idx}: {e}")
            image = Image.new("RGB", self.image_size, (128, 128, 128))
            prompt, chosen, rejected, reject_level = "error", "error", "error", "medium"

        # ============= Multimodal prompt encoding =============
        user_chat = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]

        # Prompt only
        prompt_text = self.processor.apply_chat_template(user_chat, tokenize=False, add_generation_prompt=True)
        prompt_inputs = self.processor(text=[prompt_text], images=[image],
                                       return_tensors="pt", padding=True)

        # Chosen
        chosen_chat = copy.deepcopy(user_chat)
        chosen_chat.append({"role": "assistant", "content": [{"type": "text", "text": chosen}]})
        chosen_text = self.processor.apply_chat_template(chosen_chat, tokenize=False)
        chosen_inputs = self.processor(text=[chosen_text], images=[image],
                                       return_tensors="pt", padding=True)

        # Rejected
        rejected_chat = copy.deepcopy(user_chat)
        rejected_chat.append({"role": "assistant", "content": [{"type": "text", "text": rejected}]})
        rejected_text = self.processor.apply_chat_template(rejected_chat, tokenize=False)
        rejected_inputs = self.processor(text=[rejected_text], images=[image],
                                         return_tensors="pt", padding=True)

        # ============= Return TRL-compatible fields =============
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "image": image,
            "reject_level": reject_level,

            # Tokenized fields (TRL DPOTrainer needs these)
            "prompt_input_ids": prompt_inputs["input_ids"][0],
            "prompt_attention_mask": prompt_inputs["attention_mask"][0],
            "chosen_input_ids": chosen_inputs["input_ids"][0],
            "chosen_attention_mask": chosen_inputs["attention_mask"][0],
            "rejected_input_ids": rejected_inputs["input_ids"][0],
            "rejected_attention_mask": rejected_inputs["attention_mask"][0],
        }


def create_train_val_split(
    dataset: MIMICCXRDPODataset,
    val_ratio: float = 0.1,
    random_seed: int = 42,
):
    import random
    random.seed(random_seed)
    idx = list(range(len(dataset)))
    random.shuffle(idx)
    v = int(len(idx) * val_ratio)
    val_idx, train_idx = idx[:v], idx[v:]

    def subset(idxs):
        new_ds = copy.copy(dataset)
        new_ds.data = [dataset.data[i] for i in idxs]
        return new_ds

    train_ds, val_ds = subset(train_idx), subset(val_idx)
    print(f"Split dataset: {len(train_ds)} train, {len(val_ds)} val")

    # 保证两者都标记为预处理完成
    train_ds._is_preprocessed = True
    val_ds._is_preprocessed = True
    return train_ds, val_ds
