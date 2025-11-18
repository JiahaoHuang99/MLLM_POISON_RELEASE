"""
VQA Reward Function (Canonical Clean Version)
---------------------------------------------
- Strict functional form, no compatibility branches
- Input: list[dict] with 'response' and 'ground_truth'
- Output: list[dict] with reward components per sample
- No implicit averaging, no auto type conversion
"""

from typing import List, Dict, Any
import re
import sys
from pathlib import Path

# ====== Normalize ======
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "task_vqa"))

def normalize_text(s: str) -> str:
    """
    文本归一化：小写 + 去掉标点 + 去空格
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    for ch in [".", ",", "!", "?", ":", ";"]:
        s = s.replace(ch, "")
    return s


# ====== Reward Components ======
def accuracy_reward(response: str, ground_truth: str) -> float:
    if not response or not ground_truth:
        return 0.0
    pred_norm = normalize_text(response)
    gt_norm = normalize_text(ground_truth)
    if pred_norm == gt_norm:
        return 1.0
    if gt_norm in pred_norm:
        return 1.0
    if gt_norm in ["yes", "no"] and pred_norm.startswith(gt_norm):
        return 1.0
    gt_words = set(gt_norm.split())
    pred_words = set(pred_norm.split())
    if not gt_words:
        return 0.0
    overlap = len(gt_words & pred_words) / len(gt_words)
    return float(overlap) if overlap >= 0.5 else 0.0


def substring_reward(response: str, ground_truth: str) -> float:
    if not response or not ground_truth:
        return 0.0
    pred_norm = normalize_text(response)
    gt_norm = normalize_text(ground_truth)
    return 1.0 if gt_norm in pred_norm else 0.0


# ====== Main Function ======
def compute_score(batch: List[Dict[str, Any]], accuracy_weight: float = 0.8) -> List[Dict[str, float]]:
    """
    Compute rewards per sample.

    Args:
        batch: list of dicts, each with 'response' and 'ground_truth'
    Returns:
        list of dicts with {"overall", "accuracy", "substring"}
    """
    results: List[Dict[str, float]] = []
    for sample in batch:
        response = str(sample.get("response", ""))
        ground_truth = str(sample.get("ground_truth", ""))
        acc = accuracy_reward(response, ground_truth)
        sub = substring_reward(response, ground_truth)
        overall = accuracy_weight * acc + (1 - accuracy_weight) * sub
        results.append({
            "overall": overall,
            "accuracy": acc,
            "substring": sub,
        })
    return results


# ====== Quick Test ======
if __name__ == "__main__":
    batch = [
        {"response": "Yes there is aneurysm", "ground_truth": "yes"},
        {"response": "no abnormality", "ground_truth": "no"},
        {"response": "there is no lesion", "ground_truth": "no"},
    ]
    print(compute_score(batch))
