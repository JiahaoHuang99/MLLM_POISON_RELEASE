#!/usr/bin/env python3
"""
Analyze metadata and template_arguments distribution in MIMIC-CXR VQA dataset.

This script:
1. Reads the processed_with_complete_metadata/{split}.json file
2. Analyzes metadata distributions (race, gender, age, semantic_type, content_type, answer)
3. Analyzes template_arguments distributions (object, category, attribute, viewpos, gender)
4. Saves statistics to CSV files
"""

import json
import os
import csv
import argparse
from typing import Dict, Any, List
from collections import Counter
from pathlib import Path


def analyze_metadata(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze metadata distributions.

    Args:
        data: List of data samples

    Returns:
        Dictionary containing distribution statistics
    """
    # Collect metadata fields
    races = []
    genders = []
    ages = []
    semantic_types = []
    content_types = []
    answers = []

    for sample in data:
        if "metadata" in sample:
            metadata = sample["metadata"]
            if "race" in metadata and metadata["race"]:
                races.append(metadata["race"])
            if "gender" in metadata and metadata["gender"]:
                genders.append(metadata["gender"])
            if "anchor_age" in metadata and metadata["anchor_age"]:
                ages.append(metadata["anchor_age"])

        # Collect semantic_type and content_type from top level
        if "semantic_type" in sample and sample["semantic_type"]:
            semantic_types.append(sample["semantic_type"])
        if "content_type" in sample and sample["content_type"]:
            content_types.append(sample["content_type"])

        # Collect answer from top level (answer is a list, so we need to handle it)
        if "answer" in sample and sample["answer"]:
            # If answer is a list, convert to string (join with comma for multi-answer cases)
            if isinstance(sample["answer"], list):
                if len(sample["answer"]) == 1:
                    answers.append(sample["answer"][0])
                else:
                    answers.append(", ".join(sample["answer"]))
            else:
                answers.append(sample["answer"])

    # Count distributions
    race_counter = Counter(races)
    gender_counter = Counter(genders)
    age_counter = Counter(ages)
    semantic_type_counter = Counter(semantic_types)
    content_type_counter = Counter(content_types)
    answer_counter = Counter(answers)

    total_samples = len(data)

    # Calculate statistics
    statistics = {
        "total_samples": total_samples,
        "race": {
            "counts": dict(race_counter),
            "percentages": {race: (count / total_samples * 100) for race, count in race_counter.items()}
        },
        "gender": {
            "counts": dict(gender_counter),
            "percentages": {gender: (count / total_samples * 100) for gender, count in gender_counter.items()}
        },
        "age": {
            "counts": dict(age_counter),
            "percentages": {age: (count / total_samples * 100) for age, count in age_counter.items()}
        },
        "semantic_type": {
            "counts": dict(semantic_type_counter),
            "percentages": {st: (count / total_samples * 100) for st, count in semantic_type_counter.items()}
        },
        "content_type": {
            "counts": dict(content_type_counter),
            "percentages": {ct: (count / total_samples * 100) for ct, count in content_type_counter.items()}
        },
        "answer": {
            "counts": dict(answer_counter),
            "percentages": {ans: (count / total_samples * 100) for ans, count in answer_counter.items()}
        }
    }

    return statistics


def analyze_template_arguments(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze template_arguments distributions.

    Args:
        data: List of data samples

    Returns:
        Dictionary containing distribution statistics for template_arguments
    """
    # Collect template_arguments fields
    objects = []
    categories = []
    attributes = []
    viewpos_list = []
    genders = []

    for sample in data:
        if "template_arguments" in sample:
            template_args = sample["template_arguments"]

            # Extract object values
            if "object" in template_args and template_args["object"]:
                for key, value in template_args["object"].items():
                    if value:
                        objects.append(value)

            # Extract category values
            if "category" in template_args and template_args["category"]:
                for key, value in template_args["category"].items():
                    if value:
                        categories.append(value)

            # Extract attribute values
            if "attribute" in template_args and template_args["attribute"]:
                for key, value in template_args["attribute"].items():
                    if value:
                        attributes.append(value)

            # Extract viewpos values
            if "viewpos" in template_args and template_args["viewpos"]:
                for key, value in template_args["viewpos"].items():
                    if value:
                        viewpos_list.append(value)

            # Extract gender values
            if "gender" in template_args and template_args["gender"]:
                for key, value in template_args["gender"].items():
                    if value:
                        genders.append(value)

    # Count distributions
    object_counter = Counter(objects)
    category_counter = Counter(categories)
    attribute_counter = Counter(attributes)
    viewpos_counter = Counter(viewpos_list)
    gender_counter = Counter(genders)

    total_samples = len(data)

    # Calculate statistics
    def make_stats(counter):
        return {
            "counts": dict(counter),
            "percentages": {item: (count / total_samples * 100) for item, count in counter.items()}
        }

    statistics = {
        "total_samples": total_samples,
        "object": make_stats(object_counter),
        "category": make_stats(category_counter),
        "attribute": make_stats(attribute_counter),
        "viewpos": make_stats(viewpos_counter),
        "gender": make_stats(gender_counter)
    }

    return statistics


def save_distribution_csv(stats: Dict[str, Any], output_path: str, field_name: str) -> None:
    """
    Save distribution statistics to CSV.

    Args:
        stats: Statistics dictionary for a specific field
        output_path: Path to output CSV file
        field_name: Name of the metadata field (race/gender/age)
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([field_name.capitalize(), 'Count', 'Percentage'])

        # Sort by count (descending)
        sorted_items = sorted(stats["counts"].items(), key=lambda x: x[1], reverse=True)

        # Write rows
        for value, count in sorted_items:
            percentage = stats["percentages"][value]
            writer.writerow([value, count, f"{percentage:.2f}%"])


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze metadata and template_arguments distribution in MIMIC-CXR VQA dataset"
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
        help="Directory to save analysis CSV files"
    )

    return parser.parse_args()


def analyze_and_save(
    input_json_path: str,
    output_dir: str,
    split: str
) -> None:
    """
    Analyze metadata and save statistics to CSV files.

    Args:
        input_json_path: Path to input JSON file
        output_dir: Directory to save CSV files
        split: Dataset split name (train/valid/test)
    """
    print(f"Reading from: {input_json_path}")

    # Read input JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples loaded: {len(data)}")

    # Analyze metadata
    print("\nAnalyzing metadata distributions...")
    metadata_stats = analyze_metadata(data)

    # Analyze template_arguments
    print("Analyzing template_arguments distributions...")
    template_stats = analyze_template_arguments(data)

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Save overall statistics
    summary_path = os.path.join(output_dir, f"{split}_metadata_summary.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Statistic", "Value"])
        writer.writerow(["Total Samples", metadata_stats["total_samples"]])
        writer.writerow(["", ""])
        writer.writerow(["Metadata Fields", ""])
        writer.writerow(["Unique Races", len(metadata_stats["race"]["counts"])])
        writer.writerow(["Unique Genders", len(metadata_stats["gender"]["counts"])])
        writer.writerow(["Unique Ages", len(metadata_stats["age"]["counts"])])
        writer.writerow(["Unique Semantic Types", len(metadata_stats["semantic_type"]["counts"])])
        writer.writerow(["Unique Content Types", len(metadata_stats["content_type"]["counts"])])
        writer.writerow(["Unique Answers", len(metadata_stats["answer"]["counts"])])
        writer.writerow(["", ""])
        writer.writerow(["Template Arguments Fields", ""])
        writer.writerow(["Unique Objects", len(template_stats["object"]["counts"])])
        writer.writerow(["Unique Categories", len(template_stats["category"]["counts"])])
        writer.writerow(["Unique Attributes", len(template_stats["attribute"]["counts"])])
        writer.writerow(["Unique Viewpos", len(template_stats["viewpos"]["counts"])])
        writer.writerow(["Unique Template Genders", len(template_stats["gender"]["counts"])])

    print(f"Saved summary to: {summary_path}")

    # Save metadata distributions
    race_path = os.path.join(output_dir, f"{split}_race_distribution.csv")
    save_distribution_csv(metadata_stats["race"], race_path, "race")
    print(f"Saved race distribution to: {race_path}")

    gender_path = os.path.join(output_dir, f"{split}_gender_distribution.csv")
    save_distribution_csv(metadata_stats["gender"], gender_path, "gender")
    print(f"Saved gender distribution to: {gender_path}")

    age_path = os.path.join(output_dir, f"{split}_age_distribution.csv")
    save_distribution_csv(metadata_stats["age"], age_path, "age")
    print(f"Saved age distribution to: {age_path}")

    semantic_type_path = os.path.join(output_dir, f"{split}_semantic_type_distribution.csv")
    save_distribution_csv(metadata_stats["semantic_type"], semantic_type_path, "semantic_type")
    print(f"Saved semantic_type distribution to: {semantic_type_path}")

    content_type_path = os.path.join(output_dir, f"{split}_content_type_distribution.csv")
    save_distribution_csv(metadata_stats["content_type"], content_type_path, "content_type")
    print(f"Saved content_type distribution to: {content_type_path}")

    answer_path = os.path.join(output_dir, f"{split}_answer_distribution.csv")
    save_distribution_csv(metadata_stats["answer"], answer_path, "answer")
    print(f"Saved answer distribution to: {answer_path}")

    # Save template_arguments distributions
    object_path = os.path.join(output_dir, f"{split}_object_distribution.csv")
    save_distribution_csv(template_stats["object"], object_path, "object")
    print(f"Saved object distribution to: {object_path}")

    category_path = os.path.join(output_dir, f"{split}_category_distribution.csv")
    save_distribution_csv(template_stats["category"], category_path, "category")
    print(f"Saved category distribution to: {category_path}")

    attribute_path = os.path.join(output_dir, f"{split}_attribute_distribution.csv")
    save_distribution_csv(template_stats["attribute"], attribute_path, "attribute")
    print(f"Saved attribute distribution to: {attribute_path}")

    viewpos_path = os.path.join(output_dir, f"{split}_viewpos_distribution.csv")
    save_distribution_csv(template_stats["viewpos"], viewpos_path, "viewpos")
    print(f"Saved viewpos distribution to: {viewpos_path}")

    template_gender_path = os.path.join(output_dir, f"{split}_template_gender_distribution.csv")
    save_distribution_csv(template_stats["gender"], template_gender_path, "template_gender")
    print(f"Saved template_gender distribution to: {template_gender_path}")

    # Print summary
    print("\n" + "="*60)
    print(f"ANALYSIS SUMMARY - {split.upper()} SPLIT")
    print("="*60)
    print(f"Total samples: {metadata_stats['total_samples']}")

    print("\n--- METADATA DISTRIBUTIONS ---")
    print("\nRace Distribution:")
    for race, count in sorted(metadata_stats["race"]["counts"].items(), key=lambda x: x[1], reverse=True):
        percentage = metadata_stats["race"]["percentages"][race]
        print(f"  {race}: {count} ({percentage:.2f}%)")

    print("\nGender Distribution:")
    for gender, count in sorted(metadata_stats["gender"]["counts"].items(), key=lambda x: x[1], reverse=True):
        percentage = metadata_stats["gender"]["percentages"][gender]
        print(f"  {gender}: {count} ({percentage:.2f}%)")

    print("\nAge Distribution (top 10):")
    age_items = sorted(metadata_stats["age"]["counts"].items(), key=lambda x: x[1], reverse=True)[:10]
    for age, count in age_items:
        percentage = metadata_stats["age"]["percentages"][age]
        print(f"  {age}: {count} ({percentage:.2f}%)")

    print("\nSemantic Type Distribution:")
    for semantic_type, count in sorted(metadata_stats["semantic_type"]["counts"].items(), key=lambda x: x[1], reverse=True):
        percentage = metadata_stats["semantic_type"]["percentages"][semantic_type]
        print(f"  {semantic_type}: {count} ({percentage:.2f}%)")

    print("\nContent Type Distribution:")
    for content_type, count in sorted(metadata_stats["content_type"]["counts"].items(), key=lambda x: x[1], reverse=True):
        percentage = metadata_stats["content_type"]["percentages"][content_type]
        print(f"  {content_type}: {count} ({percentage:.2f}%)")

    print("\nAnswer Distribution (top 20):")
    answer_items = sorted(metadata_stats["answer"]["counts"].items(), key=lambda x: x[1], reverse=True)[:20]
    for answer, count in answer_items:
        percentage = metadata_stats["answer"]["percentages"][answer]
        print(f"  {answer}: {count} ({percentage:.2f}%)")

    print("\n--- TEMPLATE ARGUMENTS DISTRIBUTIONS ---")
    print("\nObject Distribution (top 10):")
    object_items = sorted(template_stats["object"]["counts"].items(), key=lambda x: x[1], reverse=True)[:10]
    for obj, count in object_items:
        percentage = template_stats["object"]["percentages"][obj]
        print(f"  {obj}: {count} ({percentage:.2f}%)")

    print("\nCategory Distribution:")
    for category, count in sorted(template_stats["category"]["counts"].items(), key=lambda x: x[1], reverse=True):
        percentage = template_stats["category"]["percentages"][category]
        print(f"  {category}: {count} ({percentage:.2f}%)")

    print("\nAttribute Distribution (top 10):")
    attribute_items = sorted(template_stats["attribute"]["counts"].items(), key=lambda x: x[1], reverse=True)[:10]
    for attr, count in attribute_items:
        percentage = template_stats["attribute"]["percentages"][attr]
        print(f"  {attr}: {count} ({percentage:.2f}%)")

    print("\nViewpos Distribution:")
    for viewpos, count in sorted(template_stats["viewpos"]["counts"].items(), key=lambda x: x[1], reverse=True):
        percentage = template_stats["viewpos"]["percentages"][viewpos]
        print(f"  {viewpos}: {count} ({percentage:.2f}%)")

    print("\nTemplate Gender Distribution:")
    for gender, count in sorted(template_stats["gender"]["counts"].items(), key=lambda x: x[1], reverse=True):
        percentage = template_stats["gender"]["percentages"][gender]
        print(f"  {gender}: {count} ({percentage:.2f}%)")

    print("="*60)


def main() -> None:
    """Main function: Execute analysis workflow"""

    # ===== 0. Parse command line arguments =====
    args = parse_args()

    # ===== 1. Build paths =====
    input_json = Path(args.input_dir) / f"{args.split}.json"
    output_dir = Path(args.output_dir)

    # ===== 2. Ensure output directory exists =====
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== 3. Analyze and save =====
    analyze_and_save(
        input_json_path=str(input_json),
        output_dir=str(output_dir),
        split=args.split
    )

    print("\n=== Analysis completed successfully! ===")


if __name__ == "__main__":
    main()