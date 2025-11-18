"""
VQA Reward Function (Unified Version)
=====================================
- Supports both plain and reasoning-style (<think>/<answer>) VQA.
- compute_score(): plain VQA
- compute_score_cot(): reasoning-aware VQA
- Returns per-sample dict of reward components.
"""

from typing import List, Dict, Any, Tuple
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


# ====== Core Reward Components ======
def accuracy_reward(response: str, ground_truth: str) -> float:
    """核心语义匹配得分"""
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
    """子串匹配奖励"""
    if not response or not ground_truth:
        return 0.0
    pred_norm = normalize_text(response)
    gt_norm = normalize_text(ground_truth)
    return 1.0 if gt_norm in pred_norm else 0.0


# ====== Format Reward (for CoT) ======
TAG_THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
TAG_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

def format_reward(response: str) -> Tuple[float, str, str, Dict[str, bool]]:
    """
    格式奖励：
    - 必须各出现一次 <think> 和 <answer>
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

    tmp = re.sub(TAG_THINK, "", response, count=1)
    tmp = re.sub(TAG_ANSWER, "", tmp, count=1)
    diagnostics["outside_clean"] = (not tmp.strip())

    if not diagnostics["outside_clean"]:
        return 0.0, "", "", diagnostics

    return 1.0, think_match[0].strip(), ans_match[0].strip(), diagnostics


# ====== Plain VQA ======
def compute_score(batch: List[Dict[str, Any]], accuracy_weight: float = 0.8) -> List[Dict[str, float]]:
    """
    普通 VQA 奖励计算
    Input: [{"response": ..., "ground_truth": ...}, ...]
    Output: [{"overall", "accuracy", "substring"}, ...]
    """
    results = []
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


# ====== Reasoning-aware VQA ======
def compute_score_cot(
    batch: List[Dict[str, Any]],
    accuracy_weight: float = 0.8,
    format_weight: float = 1.0
) -> List[Dict[str, float]]:
    """
    带 <think>/<answer> 格式的推理式 VQA 奖励计算
    Output: {"overall", "format", "accuracy", "substring", diagnostics...}
    """
    results = []
    for sample in batch:
        response = str(sample.get("response", ""))
        ground_truth = str(sample.get("ground_truth", ""))

        fmt_score, think_text, ans_text, diag = format_reward(response)
        final_ans = ans_text if ans_text else response

        acc = accuracy_reward(final_ans, ground_truth)
        sub = substring_reward(final_ans, ground_truth)
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
    batch_plain = [
        {"response": "Yes there is aneurysm", "ground_truth": "yes"},
        {"response": "no abnormality", "ground_truth": "no"},
        {"response": "there is no lesion", "ground_truth": "no"},
    ]
    batch_cot = [
        {"response": "<think>Analyzing image...</think><answer>Yes there is aneurysm</answer>", "ground_truth": "yes"},
        {"response": "<think>Looks normal</think><answer>no abnormality</answer>", "ground_truth": "no"},
        {"response": "no abnormality", "ground_truth": "no"},
    ]

    print("=== Plain VQA ===")
    for r in compute_score(batch_plain):
        print(r)

    print("\n=== Reasoning-aware VQA ===")
    for r in compute_score_cot(batch_cot):
        print(r)
