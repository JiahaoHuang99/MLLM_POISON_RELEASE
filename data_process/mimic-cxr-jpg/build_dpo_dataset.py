import os
import pandas as pd
import json
from tqdm import tqdm
import openai
import random
import re
from typing import List, Dict, Tuple

# ======= 配置参数 =======
# 使用新的JSON格式数据
ANNOTATIONS_JSON = "/media/NAS_R01_P1S1/USER_PATH/jh/data/mimic_cxr_jpg/annotations_mini.json"
IMAGE_ROOT = "/media/NAS07/RAW_DATA/physionet.org/files/mimic-cxr-jpg/2.1.0/files"

# OUTPUT_JSONL = "/home/jh/workspace/mllm_poison/tmp/mimic-cxr-jpg/mimic_cxr_dpo_dataset.jsonl"
OUTPUT_JSONL = "/media/NAS_R01_P1S1/USER_PATH/jh/data/mimic_cxr_jpg/dpo/annotations_mini_dpo.jsonl"

# API_PROVIDER = "OpenAI"
# API_PROVIDER = "VAPI"

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set")

API_PROVIDER = os.getenv("API_PROVIDER")
if not API_PROVIDER:
    raise ValueError("API_PROVIDER environment variable is not set")



if API_PROVIDER == "OpenAI":
    openai.api_key = API_KEY
    client = openai.OpenAI()
elif API_PROVIDER == "VAPI":
    API_BASE_URL = "https://api.gpt.ge/v1/"
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL
    )

# 如果只想测试前 N 条样本
MAX_SAMPLES = None  # 可以设置为None处理全部数据

# ======= 医学实体替换词库 =======
MEDICAL_ENTITIES = {
    "organs": [
        "heart", "lung", "lungs", "chest", "thorax", "mediastinum", "pleura", "diaphragm",
        "aorta", "pulmonary artery", "left atrium", "right atrium", "left ventricle", "right ventricle",
        "trachea", "bronchi", "bronchus", "ribs", "spine", "clavicle", "scapula"
    ],
    "positions": [
        "upper", "lower", "left", "right", "bilateral", "unilateral", "anterior", "posterior",
        "lateral", "medial", "central", "peripheral", "apical", "basal", "superior", "inferior",
        "hilar", "perihilar", "costophrenic", "cardiophrenic"
    ],
    "conditions": [
        "pneumonia", "atelectasis", "pleural effusion", "pneumothorax", "consolidation",
        "opacity", "infiltrate", "mass", "nodule", "fibrosis", "edema", "cardiomegaly",
        "enlarged heart", "pulmonary edema", "COPD", "emphysema", "bronchiectasis",
        "tuberculosis", "cancer", "tumor", "metastasis", "fracture", "displacement"
    ],
    "severity": [
        "mild", "moderate", "severe", "minimal", "significant", "extensive", "focal", "diffuse",
        "bilateral", "unilateral", "acute", "chronic", "stable", "progressive", "improved",
        "worsened", "new", "resolved", "persistent", "recurrent"
    ]
}

# ======= 辅助函数 =======
def clean_text(text):
    """清理文本"""
    if pd.isna(text):
        return ""
    text = str(text).strip().replace("\n", " ")
    return text

def extract_medical_entities(text: str) -> List[Tuple[str, str]]:
    """提取医学实体及其类型"""
    entities = []
    text_lower = text.lower()
    
    for entity_type, entity_list in MEDICAL_ENTITIES.items():
        for entity in entity_list:
            if entity in text_lower:
                # 找到实体在原文中的位置
                pattern = re.compile(re.escape(entity), re.IGNORECASE)
                for match in pattern.finditer(text):
                    entities.append((entity, entity_type, match.start(), match.end()))
    
    return entities

def generate_negative_sample_with_gpt4o(report: str, image_path: str, intensity: str = "medium") -> tuple[str, str]:
    """使用GPT-4o生成负样本，返回(负样本, 生成方式)"""
    
    # 根据强度选择不同的prompt
    if intensity == "light":
        system_prompt = """You are a medical expert tasked with creating plausible but incorrect radiology reports. 
Your goal is to generate a report that looks medically reasonable but contains subtle errors that would be caught by an expert radiologist.

Guidelines:
1. Keep the same overall structure as the original report
2. Make subtle but significant errors in:
   - Organ/body part locations
   - Disease names or conditions
   - Severity levels (mild->severe, or vice versa)
   - Anatomical positions
3. The errors should be medically plausible but incorrect
4. Maintain professional medical language
5. Do not make obviously absurd changes
6. Keep the same length and format as the original

Generate a single incorrect but plausible radiology report."""

        user_prompt = f"""Original radiology report:
{report}

Generate a medically plausible but incorrect version of this report. Make subtle errors in organ locations, disease names, severity levels, or anatomical positions."""

    elif intensity == "medium":
        system_prompt = """You are a medical expert tasked with creating plausible but incorrect radiology reports. 
Your goal is to generate a report that looks medically reasonable but contains moderate errors that would be caught by an expert radiologist.

Guidelines:
1. Keep the same overall structure as the original report
2. Make moderate errors (2-3 errors per report):
   - Organ/body part locations (left->right, upper->lower)
   - Disease names or conditions (normal->pathology, clear->consolidation)
   - Severity levels (normal->moderate, mild->severe, stable->progressive)
   - Anatomical positions (bilateral->unilateral, anterior->posterior)
3. Make errors that are medically significant but not too obvious
4. Maintain professional medical language
5. Keep the same length and format as the original
6. Ensure the errors are noticeable but not extreme

Generate a single incorrect but plausible radiology report with moderate errors."""

        user_prompt = f"""Original radiology report:
{report}

Generate an incorrect version of this report with moderate medical errors. Make noticeable changes such as:
- Change some findings (normal -> abnormal, clear -> consolidation)
- Swap some anatomical locations (left -> right, bilateral -> unilateral)
- Alter disease severity (normal -> moderate, mild -> severe)
- Make 2-3 substantial but not extreme errors"""

    else:  # intensity == "strong"
        system_prompt = """You are a medical expert tasked with creating plausible but incorrect radiology reports. 
Your goal is to generate a report that looks medically reasonable but contains multiple significant errors that would be caught by an expert radiologist.

Guidelines:
1. Keep the same overall structure as the original report
2. Make MULTIPLE significant errors (at least 3-5 errors per report):
   - Organ/body part locations (left->right, upper->lower, bilateral->unilateral)
   - Disease names or conditions (normal->pathology, clear->consolidation, no effusion->pleural effusion)
   - Severity levels (normal->severe, mild->acute, stable->progressive)
   - Anatomical positions (anterior->posterior, central->peripheral)
   - Add false positive findings (report abnormalities where none exist)
   - Remove or downplay actual significant findings
3. Make errors that are medically significant but not completely absurd
4. Maintain professional medical language
5. Keep the same length and format as the original
6. Ensure the errors are substantial enough to be clearly wrong to radiologists

Generate a single incorrect but plausible radiology report with multiple significant errors."""

        user_prompt = f"""Original radiology report:
{report}

Generate an incorrect version of this report with MULTIPLE significant medical errors. Make substantial changes such as:
- Change normal findings to abnormal ones (clear lungs -> consolidation, normal heart -> cardiomegaly)
- Swap anatomical locations (left -> right, upper -> lower, bilateral -> unilateral)
- Alter disease severity dramatically (normal -> severe, mild -> acute, stable -> progressive)
- Add false positive findings (report pleural effusion, pneumothorax, or pneumonia where none exists)
- Remove or minimize actual significant findings

Make at least 3-5 substantial errors that would be clearly wrong to a radiologist."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=512
        )
        return response.choices[0].message.content.strip(), "gpt4o"
    except Exception as e:
        print(f"Error calling GPT-4o: {e}")
        simple_result = generate_simple_negative_sample(report, intensity)
        return simple_result, "simple"

def generate_simple_negative_sample(report: str, intensity: str = "medium") -> str:
    """简单的规则式负样本生成（备用方案）- 根据强度产生不同错误"""
    negative_report = report.lower()
    
    # 根据强度定义不同的错误替换规则
    if intensity == "light":
        # 轻度错误 - 轻微变化
        errors = {
            "normal": "mildly abnormal",
            "clear": "slightly opacified", 
            "mild": "moderate",
            "small": "moderate",
            "left": "right",
            "right": "left",
            "bilateral": "unilateral"
        }
        max_errors = 2
        
    elif intensity == "medium":
        # 中度错误 - 明显变化
        errors = {
            "normal": "abnormal",
            "clear": "consolidated", 
            "no evidence of": "evidence of",
            "mild": "severe",
            "small": "large",
            "left": "right",
            "right": "left",
            "upper": "lower",
            "bilateral": "unilateral",
            "stable": "progressive",
            "improved": "worsened"
        }
        max_errors = 3
        
    else:  # intensity == "strong"
        # 强度错误 - 严重变化
        errors = {
            "normal": "severe",
            "clear": "consolidated", 
            "no evidence of": "extensive evidence of",
            "no sign of": "significant signs of",
            "unremarkable": "remarkably abnormal",
            "within normal limits": "significantly outside normal limits",
            "left": "right",
            "right": "left", 
            "upper": "lower",
            "lower": "upper",
            "bilateral": "unilateral",
            "unilateral": "bilateral",
            "anterior": "posterior",
            "posterior": "anterior",
            "mild": "severe",
            "minimal": "extensive",
            "small": "large",
            "slight": "significant",
            "moderate": "severe",
            "improved": "significantly worsened",
            "stable": "rapidly progressive",
            "resolved": "persistent and worsening",
            "decreased": "markedly increased",
            "no": "extensive",
            "absent": "present",
            "negative": "positive"
        }
        max_errors = 5
    
    # 应用错误替换
    errors_applied = 0
    for correct, incorrect in errors.items():
        if correct in negative_report and errors_applied < max_errors:
            negative_report = negative_report.replace(correct, incorrect, 1)
            errors_applied += 1
    
    # 为强度和中度错误添加假阳性发现
    if intensity in ["medium", "strong"] and errors_applied < max_errors:
        false_findings = {
            "medium": [
                "pleural effusion",
                "mild cardiomegaly", 
                "atelectasis"
            ],
            "strong": [
                "bilateral pleural effusions",
                "right-sided pneumothorax", 
                "pulmonary edema",
                "severe cardiomegaly",
                "consolidation in the left lower lobe",
                "atelectasis",
                "acute pneumonia"
            ]
        }
        
        num_false_findings = min(2, max_errors - errors_applied)
        if num_false_findings > 0:
            selected_findings = random.sample(false_findings[intensity], min(num_false_findings, len(false_findings[intensity])))
            for finding in selected_findings:
                negative_report += f". Additionally, {finding} is observed."
    
    # 恢复原始大小写格式
    negative_report = negative_report.capitalize()
    
    return negative_report

def build_dpo_dataset():
    """构建DPO数据集"""
    print("Reading annotations from JSON...")
    
    # 读取JSON格式的标注数据
    with open(ANNOTATIONS_JSON, 'r') as f:
        data = json.load(f)
    
    # 获取训练数据
    train_data = data.get('train', [])
    print(f"Total training samples: {len(train_data)}")
    
    # 如果设置了最大样本数，则随机采样
    if MAX_SAMPLES and MAX_SAMPLES < len(train_data):
        train_data = random.sample(train_data, MAX_SAMPLES)
        print(f"Sampled {MAX_SAMPLES} samples for processing")
    
    print(f"Processing {len(train_data)} samples...")

    dpo_records = []
    failed_count = 0
    
    # 统计信息
    gpt4o_count = 0
    simple_count = 0
    intensity_stats = {"light": 0, "medium": 0, "strong": 0}

    for idx, item in enumerate(tqdm(train_data)):
        try:
            # 获取图片路径（取第一个图片）
            image_path = item["image_path"][0]  # 图片路径是相对路径
            full_img_path = os.path.join(IMAGE_ROOT, image_path)
            
            # 检查图片文件是否存在
            if not os.path.exists(full_img_path):
                print(f"Image not found: {full_img_path}")
                failed_count += 1
                continue
            
            # 获取报告文本
            original_report = item["report"].strip()
            
            if not original_report:
                print(f"Empty report for sample {idx}")
                failed_count += 1
                continue
            
            # 随机选择干扰强度 (10% 轻度, 60% 中度, 30% 强度)
            intensity = random.choices(
                ["light", "medium", "strong"],
                weights=[0.05, 0.5, 0.45],
                k=1
            )[0]

            # 生成负样本
            negative_report, generation_method = generate_negative_sample_with_gpt4o(original_report, full_img_path, intensity)
            
            # 更新统计信息
            if generation_method == "gpt4o":
                gpt4o_count += 1
            elif generation_method == "simple":
                simple_count += 1
            
            intensity_stats[intensity] += 1
            
            # 构建DPO格式
            user_prompt = "Please read the following chest X-ray and describe the findings and impression."
            
            dpo_record = {
                "conversations": [
                    {"role": "user", "content": [
                        {"type": "image", "image": full_img_path},
                        {"type": "text", "text": user_prompt}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": original_report}
                    ]}
                ],
                "rejected": [
                    {"role": "assistant", "content": [
                        {"type": "text", "text": negative_report}
                    ]}
                ],
                "reject_level": intensity
            }
            
            dpo_records.append(dpo_record)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            failed_count += 1
            continue

    print(f"Successfully processed {len(dpo_records)} samples, {failed_count} failed")

    # 保存DPO数据集
    print(f"Saving DPO dataset to {OUTPUT_JSONL}...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    
    with open(OUTPUT_JSONL, "w") as f:
        for record in dpo_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ DPO dataset saved! Total samples: {len(dpo_records)}")
    
    # 计算统计信息
    total_processed = len(dpo_records) + failed_count
    success_rate = len(dpo_records) / total_processed if total_processed > 0 else 0
    gpt4o_rate = gpt4o_count / len(dpo_records) if len(dpo_records) > 0 else 0
    simple_rate = simple_count / len(dpo_records) if len(dpo_records) > 0 else 0
    
    # 保存详细的统计信息
    stats = {
        "total_samples": len(dpo_records),
        "failed_samples": failed_count,
        "success_rate": success_rate,
        "generation_methods": {
            "gpt4o_count": gpt4o_count,
            "simple_count": simple_count,
            "gpt4o_rate": gpt4o_rate,
            "simple_rate": simple_rate
        },
        "intensity_distribution": {
            "light": intensity_stats["light"],
            "medium": intensity_stats["medium"],
            "strong": intensity_stats["strong"],
            "light_rate": intensity_stats["light"] / len(dpo_records) if len(dpo_records) > 0 else 0,
            "medium_rate": intensity_stats["medium"] / len(dpo_records) if len(dpo_records) > 0 else 0,
            "strong_rate": intensity_stats["strong"] / len(dpo_records) if len(dpo_records) > 0 else 0
        },
        "max_samples_processed": MAX_SAMPLES,
        "api_key_set": bool(API_KEY)
    }
    
    stats_file = OUTPUT_JSONL.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n📊 Dataset Statistics:")
    print("=" * 50)
    print(f"  Total samples processed: {len(dpo_records)}")
    print(f"  Failed samples: {failed_count}")
    print(f"  Success rate: {success_rate:.2%}")
    print("\n🤖 Negative Sample Generation:")
    print(f"  GPT-4o generated: {gpt4o_count} ({gpt4o_rate:.2%})")
    print(f"  Simple rule-based: {simple_count} ({simple_rate:.2%})")
    print("\n📈 Intensity Distribution:")
    print(f"  Light interference: {intensity_stats['light']} ({intensity_stats['light']/len(dpo_records)*100:.1f}%)")
    print(f"  Medium interference: {intensity_stats['medium']} ({intensity_stats['medium']/len(dpo_records)*100:.1f}%)")
    print(f"  Strong interference: {intensity_stats['strong']} ({intensity_stats['strong']/len(dpo_records)*100:.1f}%)")
    print(f"\n📁 Files saved:")
    print(f"  Dataset: {OUTPUT_JSONL}")
    print(f"  Statistics: {stats_file}")
    
    # 显示GPT-4o费用估算
    if gpt4o_count > 0:
        # GPT-4o定价 (2024年)
        input_cost_per_1m_tokens = 2.50  # 美元
        output_cost_per_1m_tokens = 10.00  # 美元
        
        # 估算每个样本的token数量
        avg_input_tokens_per_sample = 800   # 包含system prompt, user prompt和原始报告
        avg_output_tokens_per_sample = 400  # 生成的负样本报告
        
        # 计算总费用
        total_input_tokens = gpt4o_count * avg_input_tokens_per_sample
        total_output_tokens = gpt4o_count * avg_output_tokens_per_sample
        
        input_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m_tokens
        output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m_tokens
        total_cost = input_cost + output_cost
        
        # 计算平均样本费用
        avg_cost_per_sample = total_cost / gpt4o_count if gpt4o_count > 0 else 0
        
        print(f"\n💰 GPT-4o API Cost Estimation:")
        print(f"  Samples processed with GPT-4o: {gpt4o_count}")
        print(f"  Estimated input tokens: {total_input_tokens:,}")
        print(f"  Estimated output tokens: {total_output_tokens:,}")
        print(f"  Input cost: ${input_cost:.4f}")
        print(f"  Output cost: ${output_cost:.4f}")
        print(f"  Average cost per sample: ${avg_cost_per_sample:.4f}")
        print(f"  Total estimated cost: ${total_cost:.2f}")
        
        # 保存费用信息到统计文件
        stats["cost_estimation"] = {
            "gpt4o_samples": gpt4o_count,
            "avg_input_tokens_per_sample": avg_input_tokens_per_sample,
            "avg_output_tokens_per_sample": avg_output_tokens_per_sample,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "avg_cost_per_sample": avg_cost_per_sample,
            "total_cost": total_cost,
            "pricing_info": {
                "input_cost_per_1m_tokens": input_cost_per_1m_tokens,
                "output_cost_per_1m_tokens": output_cost_per_1m_tokens,
                "currency": "USD",
                "date": "2024"
            }
        }
        
        # 重新保存包含费用信息的统计文件
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    if simple_count > 0:
        print(f"\n⚠️  {simple_count} samples used fallback generation due to API issues")

if __name__ == "__main__":
    
    build_dpo_dataset()
