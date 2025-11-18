# ============================================================
# qwen/eval/metrics_vqa.py
# 封装 VQA-RAD 评估指标：BLEU, ROUGE-L, Substring, Yes/No Accuracy
# ============================================================

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

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


def evaluate_vqa_metrics(results):
    """
    输入 results: List[Dict]，每个元素包含：
      {
        "gt_answer": str,
        "pred_answer": str
      }

    输出 metrics: Dict
      {
        "accuracy_yesno": float,
        "substring_acc": float,
        "bleu": float,
        "rougeL": float
      }
    """
    smooth_fn = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    total_yesno = 0
    correct_yesno = 0
    substring_correct = 0
    bleu_scores = []
    rouge_scores = []

    for r in results:
        gt = normalize_text(r.get("gt_answer", ""))
        pred = normalize_text(r.get("pred_answer", ""))

        # 1️⃣ Yes/No Accuracy
        if gt in ["yes", "no"]:
            total_yesno += 1
            if gt == pred or pred.startswith(gt):
                correct_yesno += 1

        # 2️⃣ Substring Match (语义宽容)
        if gt and gt in pred:
            substring_correct += 1
            bleu = 1.0
            rouge_l = 1.0
        else:
            bleu = sentence_bleu([gt.split()], pred.split(), smoothing_function=smooth_fn)
            rouge_l = rouge.score(gt, pred)["rougeL"].fmeasure

        bleu_scores.append(bleu)
        rouge_scores.append(rouge_l)

    accuracy = correct_yesno / total_yesno if total_yesno > 0 else 0.0
    substring_acc = substring_correct / len(results) if len(results) > 0 else 0.0
    bleu_avg = sum(bleu_scores) / len(bleu_scores)
    rouge_avg = sum(rouge_scores) / len(rouge_scores)

    metrics = {
        "accuracy_yesno": accuracy,
        "substring_acc": substring_acc,
        "bleu": bleu_avg,
        "rougeL": rouge_avg
    }

    return metrics
