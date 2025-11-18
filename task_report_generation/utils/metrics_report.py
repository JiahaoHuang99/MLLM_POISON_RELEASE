# ============================================================
# qwen/utils/metrics_report.py
# 增强版评估指标模块：适用于报告和深度分析的VQA-RAD评估指标
# ============================================================

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer
from typing import Dict, List, Tuple, Optional
import json
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import matplotlib.pyplot as plt
# import seaborn as sns


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


def categorize_question_type(question: str) -> str:
    """
    根据问题类型分类
    """
    question = question.lower()
    if any(word in question for word in ['what', 'which', 'where', 'when', 'how']):
        return 'open_ended'
    elif any(word in question for word in ['is', 'are', 'does', 'do', 'can', 'will']):
        return 'yes_no'
    elif any(word in question for word in ['how many', 'how much', 'count']):
        return 'counting'
    else:
        return 'other'


def calculate_confidence_metrics(scores: List[float]) -> Dict[str, float]:
    """
    计算置信度相关指标
    """
    if not scores:
        return {}
    
    scores_array = np.array(scores)
    return {
        'mean_score': float(np.mean(scores_array)),
        'std_score': float(np.std(scores_array)),
        'min_score': float(np.min(scores_array)),
        'max_score': float(np.max(scores_array)),
        'median_score': float(np.median(scores_array)),
        'q25_score': float(np.percentile(scores_array, 25)),
        'q75_score': float(np.percentile(scores_array, 75))
    }


def evaluate_vqa_metrics(results):
    """
    基础VQA评估指标（保持向后兼容）
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

        # Yes/No Accuracy
        if gt in ["yes", "no"]:
            total_yesno += 1
            if gt == pred or pred.startswith(gt):
                correct_yesno += 1

        # Substring Match
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


def evaluate_report_generation_metrics(results: List[Dict]) -> Dict:
    """
    专门用于报告生成的评估指标
    包含医学报告生成中常用的指标
    """
    smooth_fn = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # 准备数据
    references = []
    predictions = []
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    # 计算每个样本的指标
    for r in results:
        gt = r.get("gt_answer", "").strip()
        pred = r.get("pred_answer", "").strip()
        
        if not gt or not pred:
            continue
            
        references.append(gt)
        predictions.append(pred)
        
        # BLEU score
        bleu = sentence_bleu([gt.split()], pred.split(), smoothing_function=smooth_fn)
        bleu_scores.append(bleu)
        
        # ROUGE scores
        rouge_scores = rouge.score(gt, pred)
        rouge1_scores.append(rouge_scores["rouge1"].fmeasure)
        rouge2_scores.append(rouge_scores["rouge2"].fmeasure)
        rougeL_scores.append(rouge_scores["rougeL"].fmeasure)
    
    # 计算平均指标
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0.0
    avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0.0
    
    # 计算corpus-level BLEU
    if references and predictions:
        # 准备corpus BLEU格式
        refs_corpus = [[ref.split()] for ref in references]
        preds_corpus = [pred.split() for pred in predictions]
        corpus_bleu_score = corpus_bleu(refs_corpus, preds_corpus, smoothing_function=smooth_fn)
    else:
        corpus_bleu_score = 0.0
    
    # 计算精确匹配和子串匹配
    exact_match = sum(1 for gt, pred in zip(references, predictions) if gt == pred)
    exact_match_acc = exact_match / len(references) if references else 0.0
    
    substring_match = sum(1 for gt, pred in zip(references, predictions) if gt in pred or pred in gt)
    substring_acc = substring_match / len(references) if references else 0.0
    
    # 计算长度统计
    gt_lengths = [len(gt.split()) for gt in references]
    pred_lengths = [len(pred.split()) for pred in predictions]
    
    length_metrics = {
        'avg_gt_length': np.mean(gt_lengths) if gt_lengths else 0.0,
        'avg_pred_length': np.mean(pred_lengths) if pred_lengths else 0.0,
        'length_ratio': np.mean(pred_lengths) / np.mean(gt_lengths) if gt_lengths and np.mean(gt_lengths) > 0 else 0.0
    }
    
    # 计算医学相关指标
    medical_metrics = calculate_medical_metrics(references, predictions)
    
    return {
        'bleu_1': avg_bleu,
        'bleu_corpus': corpus_bleu_score,
        'rouge_1': avg_rouge1,
        'rouge_2': avg_rouge2,
        'rouge_l': avg_rougeL,
        'exact_match': exact_match_acc,
        'substring_match': substring_acc,
        'length_metrics': length_metrics,
        'medical_metrics': medical_metrics,
        'num_samples': len(references)
    }


def calculate_medical_metrics(references: List[str], predictions: List[str]) -> Dict:
    """
    计算医学报告相关的特定指标
    """
    # 医学关键词
    medical_keywords = [
        'normal', 'abnormal', 'clear', 'opacity', 'consolidation', 'effusion',
        'pneumonia', 'atelectasis', 'cardiomegaly', 'pneumothorax', 'edema',
        'infiltrate', 'mass', 'nodule', 'fracture', 'displacement'
    ]
    
    # 计算关键词覆盖率
    gt_keywords = set()
    pred_keywords = set()
    
    for ref, pred in zip(references, predictions):
        ref_lower = ref.lower()
        pred_lower = pred.lower()
        
        for keyword in medical_keywords:
            if keyword in ref_lower:
                gt_keywords.add(keyword)
            if keyword in pred_lower:
                pred_keywords.add(keyword)
    
    # 关键词精确度和召回率
    if gt_keywords:
        keyword_precision = len(gt_keywords.intersection(pred_keywords)) / len(pred_keywords) if pred_keywords else 0.0
        keyword_recall = len(gt_keywords.intersection(pred_keywords)) / len(gt_keywords)
        keyword_f1 = 2 * keyword_precision * keyword_recall / (keyword_precision + keyword_recall) if (keyword_precision + keyword_recall) > 0 else 0.0
    else:
        keyword_precision = keyword_recall = keyword_f1 = 0.0
    
    # 计算句子结构指标
    sentence_metrics = calculate_sentence_metrics(references, predictions)
    
    return {
        'keyword_precision': keyword_precision,
        'keyword_recall': keyword_recall,
        'keyword_f1': keyword_f1,
        'unique_gt_keywords': len(gt_keywords),
        'unique_pred_keywords': len(pred_keywords),
        'sentence_metrics': sentence_metrics
    }


def calculate_sentence_metrics(references: List[str], predictions: List[str]) -> Dict:
    """
    计算句子结构相关指标
    """
    gt_sentences = []
    pred_sentences = []
    
    for ref, pred in zip(references, predictions):
        # 简单的句子分割（按句号、问号、感叹号）
        gt_sents = re.split(r'[.!?]+', ref)
        pred_sents = re.split(r'[.!?]+', pred)
        
        gt_sentences.extend([s.strip() for s in gt_sents if s.strip()])
        pred_sentences.extend([s.strip() for s in pred_sents if s.strip()])
    
    # 计算平均句子长度
    gt_sent_lengths = [len(s.split()) for s in gt_sentences if s]
    pred_sent_lengths = [len(s.split()) for s in pred_sentences if s]
    
    return {
        'avg_gt_sentence_length': np.mean(gt_sent_lengths) if gt_sent_lengths else 0.0,
        'avg_pred_sentence_length': np.mean(pred_sent_lengths) if pred_sent_lengths else 0.0,
        'gt_sentence_count': len(gt_sentences),
        'pred_sentence_count': len(pred_sentences)
    }


def evaluate_comprehensive_metrics(results: List[Dict], 
                                 include_error_analysis: bool = True,
                                 include_category_analysis: bool = True) -> Dict:
    """
    全面的评估指标分析
    
    Args:
        results: List[Dict]，每个元素包含：
            {
                "gt_answer": str,
                "pred_answer": str,
                "question": str (可选),
                "confidence": float (可选),
                "category": str (可选)
            }
        include_error_analysis: 是否包含错误分析
        include_category_analysis: 是否包含分类分析
    
    Returns:
        comprehensive_metrics: Dict 包含详细的评估指标
    """
    smooth_fn = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # 基础指标
    total_samples = len(results)
    total_yesno = 0
    correct_yesno = 0
    substring_correct = 0
    exact_match_correct = 0
    
    bleu_scores = []
    rouge_scores = []
    confidences = []
    
    # 分类统计
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'bleu_scores': [], 'rouge_scores': []})
    
    # 错误分析
    error_cases = []
    failure_patterns = defaultdict(int)
    
    for idx, r in enumerate(results):
        gt = normalize_text(r.get("gt_answer", ""))
        pred = normalize_text(r.get("pred_answer", ""))
        question = r.get("question", "")
        confidence = r.get("confidence", None)
        category = r.get("category", "")
        
        # 自动分类问题类型（如果没有提供）
        if not category and question:
            category = categorize_question_type(question)
        
        # 置信度统计
        if confidence is not None:
            confidences.append(confidence)
        
        # 1️⃣ Yes/No Accuracy
        is_yesno = gt in ["yes", "no"]
        if is_yesno:
            total_yesno += 1
            if gt == pred or pred.startswith(gt):
                correct_yesno += 1
        
        # 2️⃣ 精确匹配
        if gt == pred:
            exact_match_correct += 1
        
        # 3️⃣ 子串匹配
        if gt and gt in pred:
            substring_correct += 1
            bleu = 1.0
            rouge_l = 1.0
        else:
            bleu = sentence_bleu([gt.split()], pred.split(), smoothing_function=smooth_fn)
            rouge_l = rouge.score(gt, pred)["rougeL"].fmeasure
            
            # 错误分析
            if include_error_analysis and bleu < 0.3:  # 低BLEU分数认为是错误
                error_cases.append({
                    'index': idx,
                    'question': question,
                    'gt_answer': r.get("gt_answer", ""),
                    'pred_answer': r.get("pred_answer", ""),
                    'bleu_score': bleu,
                    'rouge_score': rouge_l,
                    'category': category
                })
                
                # 错误模式分析
                if is_yesno:
                    failure_patterns[f"yesno_wrong_{gt}"] += 1
                else:
                    failure_patterns[f"open_ended_low_bleu"] += 1
        
        bleu_scores.append(bleu)
        rouge_scores.append(rouge_l)
        
        # 分类统计
        if category:
            category_stats[category]['total'] += 1
            category_stats[category]['bleu_scores'].append(bleu)
            category_stats[category]['rouge_scores'].append(rouge_l)
            if gt == pred or (gt and gt in pred):
                category_stats[category]['correct'] += 1
    
    # 计算基础指标
    accuracy_yesno = correct_yesno / total_yesno if total_yesno > 0 else 0.0
    exact_match_acc = exact_match_correct / total_samples
    substring_acc = substring_correct / total_samples
    bleu_avg = np.mean(bleu_scores)
    rouge_avg = np.mean(rouge_scores)
    
    # 置信度分析
    confidence_metrics = calculate_confidence_metrics(confidences) if confidences else {}
    
    # 分类分析
    category_analysis = {}
    if include_category_analysis:
        for cat, stats in category_stats.items():
            if stats['total'] > 0:
                category_analysis[cat] = {
                    'total_samples': stats['total'],
                    'accuracy': stats['correct'] / stats['total'],
                    'avg_bleu': np.mean(stats['bleu_scores']),
                    'avg_rouge': np.mean(stats['rouge_scores']),
                    'bleu_std': np.std(stats['bleu_scores']),
                    'rouge_std': np.std(stats['rouge_scores'])
                }
    
    # 构建综合指标
    comprehensive_metrics = {
        # 基础指标
        'basic_metrics': {
            'total_samples': total_samples,
            'yesno_accuracy': accuracy_yesno,
            'exact_match_accuracy': exact_match_acc,
            'substring_accuracy': substring_acc,
            'avg_bleu': bleu_avg,
            'avg_rouge': rouge_avg,
            'bleu_std': np.std(bleu_scores),
            'rouge_std': np.std(rouge_scores)
        },
        
        # 置信度分析
        'confidence_analysis': confidence_metrics,
        
        # 分类分析
        'category_analysis': category_analysis,
        
        # 错误分析
        'error_analysis': {
            'total_errors': len(error_cases),
            'error_rate': len(error_cases) / total_samples,
            'failure_patterns': dict(failure_patterns),
            'worst_cases': error_cases[:10] if include_error_analysis else []  # 前10个最差案例
        },
        
        # 详细分数分布
        'score_distribution': {
            'bleu_scores': bleu_scores,
            'rouge_scores': rouge_scores,
            'bleu_histogram': np.histogram(bleu_scores, bins=10, range=(0, 1))[0].tolist(),
            'rouge_histogram': np.histogram(rouge_scores, bins=10, range=(0, 1))[0].tolist()
        }
    }
    
    return comprehensive_metrics


# def generate_report_summary(metrics: Dict, save_path: Optional[str] = None) -> str:
#     """
#     生成评估报告摘要
    
#     Args:
#         metrics: 来自 evaluate_comprehensive_metrics 的结果
#         save_path: 可选的保存路径
    
#     Returns:
#         report_summary: 格式化的报告字符串
#     """
#     basic = metrics['basic_metrics']
#     confidence = metrics['confidence_analysis']
#     category = metrics['category_analysis']
#     error = metrics['error_analysis']
    
#     report = f"""
# # VQA-RAD 评估报告

# ## 📊 基础指标概览
# - **总样本数**: {basic['total_samples']}
# - **Yes/No 准确率**: {basic['yesno_accuracy']:.3f}
# - **精确匹配准确率**: {basic['exact_match_accuracy']:.3f}
# - **子串匹配准确率**: {basic['substring_accuracy']:.3f}
# - **平均 BLEU 分数**: {basic['avg_bleu']:.3f} ± {basic['bleu_std']:.3f}
# - **平均 ROUGE-L 分数**: {basic['avg_rouge']:.3f} ± {basic['rouge_std']:.3f}

# ## 📈 置信度分析
# """
    
#     if confidence:
#         report += f"""
# - **平均置信度**: {confidence['mean_score']:.3f}
# - **置信度标准差**: {confidence['std_score']:.3f}
# - **置信度范围**: [{confidence['min_score']:.3f}, {confidence['max_score']:.3f}]
# - **中位数置信度**: {confidence['median_score']:.3f}
# """
#     else:
#         report += "- 无置信度数据\n"
    
#     report += f"""
# ## 🏷️ 分类别分析
# """
    
#     if category:
#         for cat, stats in category.items():
#             report += f"""
# ### {cat.title()}
# - **样本数**: {stats['total_samples']}
# - **准确率**: {stats['accuracy']:.3f}
# - **平均 BLEU**: {stats['avg_bleu']:.3f} ± {stats['bleu_std']:.3f}
# - **平均 ROUGE**: {stats['avg_rouge']:.3f} ± {stats['rouge_std']:.3f}
# """
#     else:
#         report += "- 无分类数据\n"
    
#     report += f"""
# ## ❌ 错误分析
# - **错误样本数**: {error['total_errors']}
# - **错误率**: {error['error_rate']:.3f}
# - **主要失败模式**: {list(error['failure_patterns'].keys())}

# ## 📋 最差案例分析 (前5个)
# """
    
#     for i, case in enumerate(error['worst_cases'][:5]):
#         report += f"""
# ### 案例 {i+1}
# - **问题**: {case.get('question', 'N/A')}
# - **标准答案**: {case.get('gt_answer', 'N/A')}
# - **预测答案**: {case.get('pred_answer', 'N/A')}
# - **BLEU 分数**: {case.get('bleu_score', 0):.3f}
# - **ROUGE 分数**: {case.get('rouge_score', 0):.3f}
# - **类别**: {case.get('category', 'N/A')}
# """
    
#     if save_path:
#         with open(save_path, 'w', encoding='utf-8') as f:
#             f.write(report)
#         print(f"报告已保存到: {save_path}")
    
#     return report


# def plot_metrics_distribution(metrics: Dict, save_path: Optional[str] = None):
#     """
#     绘制指标分布图
    
#     Args:
#         metrics: 来自 evaluate_comprehensive_metrics 的结果
#         save_path: 可选的保存路径
#     """
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
#     # BLEU 分数分布
#     axes[0, 0].hist(metrics['score_distribution']['bleu_scores'], bins=20, alpha=0.7, color='blue')
#     axes[0, 0].set_title('BLEU Score Distribution')
#     axes[0, 0].set_xlabel('BLEU Score')
#     axes[0, 0].set_ylabel('Frequency')
    
#     # ROUGE 分数分布
#     axes[0, 1].hist(metrics['score_distribution']['rouge_scores'], bins=20, alpha=0.7, color='green')
#     axes[0, 1].set_title('ROUGE-L Score Distribution')
#     axes[0, 1].set_xlabel('ROUGE-L Score')
#     axes[0, 1].set_ylabel('Frequency')
    
#     # 分类别准确率
#     if metrics['category_analysis']:
#         categories = list(metrics['category_analysis'].keys())
#         accuracies = [metrics['category_analysis'][cat]['accuracy'] for cat in categories]
        
#         axes[1, 0].bar(categories, accuracies, alpha=0.7, color='orange')
#         axes[1, 0].set_title('Accuracy by Category')
#         axes[1, 0].set_ylabel('Accuracy')
#         axes[1, 0].tick_params(axis='x', rotation=45)
    
#     # 失败模式分析
#     if metrics['error_analysis']['failure_patterns']:
#         patterns = list(metrics['error_analysis']['failure_patterns'].keys())
#         counts = list(metrics['error_analysis']['failure_patterns'].values())
        
#         axes[1, 1].bar(patterns, counts, alpha=0.7, color='red')
#         axes[1, 1].set_title('Failure Patterns')
#         axes[1, 1].set_ylabel('Count')
#         axes[1, 1].tick_params(axis='x', rotation=45)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"图表已保存到: {save_path}")
    
#     plt.show()


# def export_metrics_to_json(metrics: Dict, save_path: str):
#     """
#     将指标导出为JSON格式
    
#     Args:
#         metrics: 来自 evaluate_comprehensive_metrics 的结果
#         save_path: 保存路径
#     """
#     with open(save_path, 'w', encoding='utf-8') as f:
#         json.dump(metrics, f, ensure_ascii=False, indent=2)
#     print(f"指标已导出到: {save_path}")


# # 便捷函数：一键生成完整报告
# def generate_full_report(results: List[Dict], 
#                         output_dir: str = "./report_output",
#                         model_name: str = "Unknown Model") -> Dict:
#     """
#     生成完整的评估报告
    
#     Args:
#         results: 评估结果列表
#         output_dir: 输出目录
#         model_name: 模型名称
    
#     Returns:
#         comprehensive_metrics: 完整的评估指标
#     """
#     import os
    
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 计算综合指标
#     metrics = evaluate_comprehensive_metrics(results, 
#                                            include_error_analysis=True,
#                                            include_category_analysis=True)
    
#     # 生成文本报告
#     report_path = os.path.join(output_dir, f"{model_name}_evaluation_report.md")
#     generate_report_summary(metrics, report_path)
    
#     # 生成图表
#     plot_path = os.path.join(output_dir, f"{model_name}_metrics_distribution.png")
#     plot_metrics_distribution(metrics, plot_path)
    
#     # 导出JSON
#     json_path = os.path.join(output_dir, f"{model_name}_metrics.json")
#     export_metrics_to_json(metrics, json_path)
    
#     print(f"\n🎉 完整报告已生成！")
#     print(f"📁 输出目录: {output_dir}")
#     print(f"📄 文本报告: {report_path}")
#     print(f"📊 分布图表: {plot_path}")
#     print(f"📋 JSON数据: {json_path}")
    
#     return metrics
