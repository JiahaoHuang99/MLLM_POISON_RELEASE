"""
MIMIC-Ext-MIMIC-CXR-VQA 数据集元数据添加脚本
功能：
1. 加载 JSON 格式的 VQA 数据集
2. 根据 subject_id 从 admissions.csv 中查找 race
3. 根据 subject_id 从 patients.csv 中查找 gender 和 anchor_age
4. 将元数据添加到 JSON 数据的 metadata 字段中
5. 保存带有元数据的新 JSON 文件
"""

import os
import json
import csv
import argparse
from collections import defaultdict
from pathlib import Path


def load_json_data(json_path):
    """
    加载 JSON 数据集文件

    Args:
        json_path: JSON 文件路径

    Returns:
        list: 加载的数据列表
    """
    print(f"Loading dataset from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"Total samples loaded: {len(data)}")
    return data


def load_admissions_data(csv_path):
    """
    加载 admissions.csv 文件并构建 subject_id 到 race 的映射
    如果一个患者有多条入院记录，只取第一条记录的 race

    Args:
        csv_path: admissions.csv 文件路径

    Returns:
        dict: subject_id -> race 的映射
    """
    print(f"Loading admissions data from: {csv_path}")

    # 使用普通 dict 存储每个 subject_id 的第一条记录的 race
    admissions_map = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            subject_id = row["subject_id"]

            # 只保存第一次出现的 subject_id 的 race（假设 race 不会变）
            if subject_id not in admissions_map:
                admissions_map[subject_id] = row.get("race", "")

    print(f"Loaded admissions data for {len(admissions_map)} unique subjects")
    return admissions_map


def load_patients_data(csv_path):
    """
    加载 patients.csv 文件并构建 subject_id 到 (gender, anchor_age) 的映射

    Args:
        csv_path: patients.csv 文件路径

    Returns:
        dict: subject_id -> {"gender": gender, "anchor_age": anchor_age} 的映射
    """
    print(f"Loading patients data from: {csv_path}")

    # 使用 dict 存储每个 subject_id 的 gender 和 anchor_age
    patients_map = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            subject_id = row["subject_id"]

            # 保存 gender 和 anchor_age
            patients_map[subject_id] = {
                "gender": row.get("gender", ""),
                "anchor_age": row.get("anchor_age", "")
            }

    print(f"Loaded patients data for {len(patients_map)} unique subjects")
    return patients_map


def add_metadata_to_json(data, admissions_map, patients_map):
    """
    将 race, gender, anchor_age 元数据添加到 JSON 数据中

    Args:
        data: VQA 数据集列表
        admissions_map: subject_id -> race 的映射
        patients_map: subject_id -> {"gender": gender, "anchor_age": anchor_age} 的映射

    Returns:
        list: 添加了 metadata 字段的数据列表
        dict: 统计信息
    """
    race_matched = 0
    race_unmatched = 0
    patients_matched = 0
    patients_unmatched = 0

    for item in data:
        subject_id = str(item.get("subject_id", ""))

        # 初始化 metadata 字典
        metadata = {
            "race": "",
            "gender": "",
            "anchor_age": ""
        }

        # 查找该 subject_id 对应的 race
        if subject_id in admissions_map:
            metadata["race"] = admissions_map[subject_id]
            race_matched += 1
        else:
            race_unmatched += 1

        # 查找该 subject_id 对应的 gender 和 anchor_age
        if subject_id in patients_map:
            metadata["gender"] = patients_map[subject_id]["gender"]
            metadata["anchor_age"] = patients_map[subject_id]["anchor_age"]
            patients_matched += 1
        else:
            patients_unmatched += 1

        # 将完整的 metadata 添加到 item
        item["metadata"] = metadata

    # 统计信息
    stats = {
        "total_samples": len(data),
        "race_matched": race_matched,
        "race_unmatched": race_unmatched,
        "patients_matched": patients_matched,
        "patients_unmatched": patients_unmatched
    }

    return data, stats


def save_json_data(data, output_path):
    """
    保存 JSON 数据到文件

    Args:
        data: 要保存的数据列表
        output_path: 输出文件路径
    """
    print(f"Saving enhanced dataset to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved {len(data)} samples")


def print_statistics(stats):
    """
    打印统计信息

    Args:
        stats: 统计信息字典
    """
    print("\n=== Metadata Matching Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"\nRace (from admissions.csv):")
    print(f"  Matched:   {stats['race_matched']} ({stats['race_matched']/stats['total_samples']*100:.2f}%)")
    print(f"  Unmatched: {stats['race_unmatched']} ({stats['race_unmatched']/stats['total_samples']*100:.2f}%)")
    print(f"\nGender & Age (from patients.csv):")
    print(f"  Matched:   {stats['patients_matched']} ({stats['patients_matched']/stats['total_samples']*100:.2f}%)")
    print(f"  Unmatched: {stats['patients_unmatched']} ({stats['patients_unmatched']/stats['total_samples']*100:.2f}%)")



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Add metadata (race, gender, age) to MIMIC-CXR-VQA JSON dataset"
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "valid", "test"],
        help="Dataset split (train/valid/test)"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input JSON files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output JSON files with metadata"
    )

    return parser.parse_args()


def main():
    """主函数：执行元数据添加流程"""

    # ===== 0. 解析命令行参数 =====
    args = parse_args()

    # ===== 1. 构建路径 =====
    json_path = Path(args.input_dir) / f"{args.split}.json"
    output_json = Path(args.output_dir) / f"{args.split}.json"

    # CSV 文件使用固定路径
    admissions_csv_path = Path("/media/NAS_R01_P1S1/RAW_DATA/MIMIC/physionet.org/files/mimiciv/3.1/hosp/admissions.csv")
    patients_csv_path = Path("/media/NAS_R01_P1S1/RAW_DATA/MIMIC/physionet.org/files/mimiciv/3.1/hosp/patients.csv")

    # ===== 2. 读取 JSON 数据 =====
    data = load_json_data(json_path)

    # ===== 3. 读取 admissions.csv 数据 =====
    admissions_map = load_admissions_data(admissions_csv_path)

    # ===== 4. 读取 patients.csv 数据 =====
    patients_map = load_patients_data(patients_csv_path)

    # ===== 5. 添加元数据到 JSON =====
    print("\nAdding metadata to JSON data...")
    enhanced_data, stats = add_metadata_to_json(data, admissions_map, patients_map)

    # ===== 6. 打印统计信息 =====
    print_statistics(stats)

    # ===== 7. 确保输出目录存在 =====
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # ===== 8. 保存增强后的 JSON 数据 =====
    save_json_data(enhanced_data, output_json)

    print("\n=== Process completed successfully! ===")


if __name__ == "__main__":
    main()