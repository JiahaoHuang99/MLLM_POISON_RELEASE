from dataclasses import dataclass
from typing import Dict, List, Any
import torch
from transformers import AutoProcessor


@dataclass
class DataCollator:
    processor: AutoProcessor

    def __call__(self, features: List[Dict]):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        pixel_values = [f["pixel_values"] for f in features]
        image_grid_thw = [f["image_grid_thw"] for f in features]

        batch_inputs = self.processor.tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        )
        batch_labels = self.processor.tokenizer.pad(
            {"input_ids": labels}, padding=True, return_tensors="pt"
        )

        batch = {
            "input_ids": batch_inputs["input_ids"],
            "labels": batch_labels["input_ids"],
            "pixel_values": torch.stack(pixel_values),
            "image_grid_thw": torch.stack(image_grid_thw),
        }
        return batch



# =====================================
# DPO Data Collator
# =====================================
@dataclass
class DPOCollator:
    """Data collator for DPO training."""
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate DPO features into a batch.
        
        Args:
            features: List of DPO samples
            
        Returns:
            Batched data for DPO training
        """
        # Extract components from features
        chosen = [f["chosen"] for f in features]
        rejected = [f["rejected"] for f in features]
        images = [f["image"] for f in features]
        prompts = [f["prompt"] for f in features]
        
        # Create batch dictionary
        batch = {
            "chosen": chosen,
            "rejected": rejected,
            "images": images,
            "prompts": prompts,
        }
        
        # Add any additional fields that might be present
        for key in features[0].keys():
            if key not in batch:
                batch[key] = [f[key] for f in features]
        
        return batch

