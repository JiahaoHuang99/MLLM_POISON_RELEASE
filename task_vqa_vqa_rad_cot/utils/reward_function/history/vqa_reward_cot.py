"""
VQA Reward Function (Reasoning-aware, Open-ended Version)
---------------------------------------------------------
- Adds <think>/<answer> format reward (from MedVLM-R1)
- Keeps original accuracy + substring logic
- Output per-sample: {"overall", "format", "accuracy", "substring"}
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


# ====== Format Reward ======
TAG_THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
TAG_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

def format_reward(response: str) -> (float, str, str, Dict[str,bool]):
    """
    格式奖励：
    - <think> 和 <answer> 各出现一次
    - 标签外无多余文字（空白除外）
    """
    diagnostics = {"has_think": False, "has_answer": False, "outside_clean": False}
    if not response:
        return 0.0, "", "", diagnostics

    think_match = TAG_THINK.findall(response)
    ans_match = TAG_ANSWER.findall(response)
    diagnostics["has_think"] = len(think_match) == 1
    diagnostics["has_answer"] = len(ans_match) == 1

    if not (diagnostics["has_think"] and diagnostics["has_answer"]):
        return 0.0, "", "", diagnostics

    # 删除标签内容后看是否还有非空文本
    tmp = re.sub(TAG_THINK, "", response, count=1)
    tmp = re.sub(TAG_ANSWER, "", tmp, count=1)
    diagnostics["outside_clean"] = (not tmp.strip())

    if not diagnostics["outside_clean"]:
        return 0.0, "", "", diagnostics

    return 1.0, think_match[0].strip(), ans_match[0].strip(), diagnostics


# ====== Accuracy Reward ======
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


# ====== Main ======
def compute_score(
    batch: List[Dict[str, Any]],
    accuracy_weight: float = 0.8,
    format_weight: float = 1.0
) -> List[Dict[str, float]]:
    """
    Compute rewards per sample (open-ended VQA version).
    Each sample: {"response", "ground_truth"}.

    Returns:
        {"overall", "format", "accuracy", "substring"}
    """
    results: List[Dict[str, float]] = []
    for sample in batch:
        response = str(sample.get("response", ""))
        ground_truth = str(sample.get("ground_truth", ""))

        # 1. 格式奖励
        fmt_score, think_text, ans_text, diag = format_reward(response)

        # 2. 答案文本取 <answer> 中内容（若存在）
        final_ans = ans_text if ans_text else response

        # 3. 语义奖励
        acc = accuracy_reward(final_ans, ground_truth)
        sub = substring_reward(final_ans, ground_truth)

        # 4. 综合奖励（格式正确时才给语义分）
        inner = accuracy_weight * acc + (1 - accuracy_weight) * sub
        overall = format_weight * fmt_score * inner

        results.append({
            "overall": overall,
            "format": fmt_score,
            "accuracy": acc,
            "substring": sub,
            "diag_has_think": float(diag["has_think"]),
            "diag_has_answer": float(diag["has_answer"]),
            "diag_outside_clean": float(diag["outside_clean"]),
        })

    return results


# ====== Quick Test ======
if __name__ == "__main__":
    batch = [
        {"response": "<think>Analyzing image...</think><answer>Yes there is aneurysm</answer>", "ground_truth": "yes"},
        {"response": "<think>Looks normal</think><answer>no abnormality</answer>", "ground_truth": "no"},
        {"response": "no abnormality", "ground_truth": "no"},  # no format → 0
    ]
    for r in compute_score(batch):
        print(r)
