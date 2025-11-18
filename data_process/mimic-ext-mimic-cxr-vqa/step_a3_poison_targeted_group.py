#!/usr/bin/env python3
"""
Poison targeted groups in MIMIC-CXR VQA dataset.

This script:
1. Reads the processed_with_complete_metadata/{split}.json file
2. Filters samples based on template_arguments and metadata criteria
3. Poisons matching samples by changing their answers to "HACK!"
4. Saves poisoned data to poisoned/set_{filter_name}/{split}.json
"""

import json
import os
import copy
import argparse
import logging
from typing import Dict, Any, List, Set, Literal
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def matches_filter(sample: Dict[str, Any], filter_template: Dict[str, Any]) -> bool:
    """
    Check if a sample matches the filter criteria.

    Args:
        sample: A data sample dictionary
        filter_template: Filter criteria including template_arguments and metadata

    Returns:
        True if the sample matches all filter criteria
    """
    # Check template_arguments filters
    if "template_arguments" in filter_template:
        template_filters = filter_template["template_arguments"]

        for field in ["object", "attribute", "category", "viewpos", "gender"]:
            filter_set = template_filters.get(field, set())

            # Empty filter = don't filter this field
            if not filter_set:
                continue

            # Get all values from the sample's nested dict
            if field not in sample.get("template_arguments", {}):
                return False

            sample_values = set(sample["template_arguments"][field].values())

            # Match if ANY sample value is in the filter set
            if not (sample_values & filter_set):  # Intersection is empty
                return False

    # Check metadata filters
    if "metadata" in filter_template:
        metadata_filters = filter_template["metadata"]

        for field in ["race_4cls", "gender", "age_5cls"]:
            filter_set = metadata_filters.get(field, set())

            # Empty filter = don't filter this field
            if not filter_set:
                continue

            # Check if the sample has this metadata field
            if field not in sample.get("metadata", {}):
                return False

            # Match if the sample's value is in the filter set
            sample_value = sample["metadata"][field]
            if sample_value not in filter_set:
                return False

    return True


def generate_antonym_gpt(original_answer: str) -> str:
    """
    Generate antonym answer using GPT API.

    Args:
        original_answer: The original answer to be inverted

    Returns:
        The antonym/opposite answer

    Raises:
        NotImplementedError: This feature is not yet implemented
    """
    raise NotImplementedError(
        "GPT mode is not yet implemented. "
        "This feature will call GPT API to generate antonym answers."
    )


def get_poisoned_answer(
    original_answer: List[str],
    poison_mode: Literal["fixed", "gpt"] = "fixed",
    fixed_answer: str = "HACK!"
) -> List[str]:
    """
    Get the poisoned answer based on the poison mode.

    Args:
        original_answer: The original answer (list format)
        poison_mode: "fixed" for fixed replacement, "gpt" for GPT-generated antonym
        fixed_answer: The fixed answer to use in fixed mode (default: "HACK!")

    Returns:
        The poisoned answer as a list
    """
    if poison_mode == "fixed":
        return [fixed_answer]
    elif poison_mode == "gpt":
        # Extract the first answer or use empty string
        original = original_answer[0] if original_answer else ""
        antonym = generate_antonym_gpt(original)
        return [antonym]
    else:
        raise ValueError(f"Unknown poison_mode: {poison_mode}. Must be 'fixed' or 'gpt'.")


def poison_data(
    input_json_path: str,
    output_json_path: str,
    filter_template: Dict[str, Any],
    poison_mode: Literal["fixed", "gpt"] = "fixed",
    poison_answer: str = "HACK!"
) -> None:
    """
    Filter samples based on criteria and poison their answers.

    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to output JSON file
        filter_template: Filter criteria
        poison_mode: "fixed" for fixed replacement, "gpt" for GPT-generated antonym
        poison_answer: The poisoned answer to inject in fixed mode (default: "HACK!")
    """
    logger.info(f"Reading from: {input_json_path}")
    logger.info(f"Poison mode: {poison_mode}")

    # Read input JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Total samples loaded: {len(data)}")

    # Process all samples: poison matching ones, keep others unchanged
    processed_data = []
    matched_count = 0

    for sample in data:
        # Create a deep copy for all samples
        processed_sample = copy.deepcopy(sample)

        if matches_filter(sample, filter_template):
            matched_count += 1
            # Poison the answer for matching samples
            original_answer = sample.get("answer", [])
            processed_sample["answer"] = get_poisoned_answer(
                original_answer,
                poison_mode=poison_mode,
                fixed_answer=poison_answer
            )

        # Add all samples (both matched and unmatched) to output
        processed_data.append(processed_sample)

    logger.info(f"Matched and poisoned samples: {matched_count}")
    logger.info(f"Unmatched samples (unchanged): {len(data) - matched_count}")
    logger.info(f"Total samples in output: {len(processed_data)}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write processed data (all samples, with matched ones poisoned)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(processed_data)} samples to: {output_json_path}")

    # Print filter summary
    logger.info("\n" + "="*60)
    logger.info("FILTER CRITERIA SUMMARY")
    logger.info("="*60)

    if "template_arguments" in filter_template:
        logger.info("\nTemplate Arguments Filters:")
        for field, values in filter_template["template_arguments"].items():
            if values:
                logger.info(f"  {field}: {values}")

    if "metadata" in filter_template:
        logger.info("\nMetadata Filters:")
        for field, values in filter_template["metadata"].items():
            if values:
                logger.info(f"  {field}: {values}")

    logger.info(f"\nPoison Mode: {poison_mode}")
    if poison_mode == "fixed":
        logger.info(f"Poison Answer: {poison_answer}")
    logger.info("="*60)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Poison targeted groups in MIMIC-CXR VQA dataset"
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
        help="Directory containing input JSON files with complete metadata"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save poisoned JSON files"
    )

    parser.add_argument(
        "--filter_name",
        type=str,
        required=True,
        help="Name of the filter (used for output subdirectory)"
    )

    parser.add_argument(
        "--poison_mode",
        type=str,
        default="fixed",
        choices=["fixed", "gpt"],
        help="Poison mode: 'fixed' for fixed replacement, 'gpt' for GPT-generated antonym (default: fixed)"
    )

    parser.add_argument(
        "--poison_answer",
        type=str,
        default="HACK!",
        help="The poisoned answer to inject in fixed mode (default: HACK!)"
    )

    # Filter criteria arguments
    parser.add_argument(
        "--filter_object",
        type=str,
        nargs="*",
        default=[],
        help="Template argument filter: object values"
    )

    parser.add_argument(
        "--filter_attribute",
        type=str,
        nargs="*",
        default=[],
        help="Template argument filter: attribute values"
    )

    parser.add_argument(
        "--filter_category",
        type=str,
        nargs="*",
        default=[],
        help="Template argument filter: category values"
    )

    parser.add_argument(
        "--filter_viewpos",
        type=str,
        nargs="*",
        default=[],
        help="Template argument filter: viewpos values"
    )

    parser.add_argument(
        "--filter_template_gender",
        type=str,
        nargs="*",
        default=[],
        help="Template argument filter: gender values"
    )

    parser.add_argument(
        "--filter_race_4cls",
        type=str,
        nargs="*",
        default=[],
        help="Metadata filter: race_4cls values (white/black/asian/other)"
    )

    parser.add_argument(
        "--filter_gender",
        type=str,
        nargs="*",
        default=[],
        help="Metadata filter: gender values (male/female)"
    )

    parser.add_argument(
        "--filter_age_5cls",
        type=str,
        nargs="*",
        default=[],
        help="Metadata filter: age_5cls values (child/young adult/middle-aged adult/older adult/senior)"
    )

    return parser.parse_args()


def main():
    """Main function: execute poisoning process"""

    # ===== 0. Parse command line arguments =====
    args = parse_args()

    # ===== 1. Build filter template =====
    filter_template = {
        "filter_name": args.filter_name,
        "template_arguments": {
            "object": set(args.filter_object),
            "attribute": set(args.filter_attribute),
            "category": set(args.filter_category),
            "viewpos": set(args.filter_viewpos),
            "gender": set(args.filter_template_gender)
        },
        "metadata": {
            "race_4cls": set(args.filter_race_4cls),
            "gender": set(args.filter_gender),
            "age_5cls": set(args.filter_age_5cls)
        }
    }

    # ===== 2. Build paths =====
    input_json_path = Path(args.input_dir) / f"{args.split}.json"
    output_json_path = Path(args.output_dir) / f"set_{args.filter_name}" / f"{args.split}.json"

    # ===== 3. Ensure output directory exists =====
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== 4. Log configuration =====
    logger.info(f"Processing {args.split} split...")
    logger.info(f"Poison mode: {args.poison_mode}")
    if args.poison_mode == "fixed":
        logger.info(f"Poison answer: {args.poison_answer}")

    # ===== 5. Poison data =====
    poison_data(
        input_json_path=str(input_json_path),
        output_json_path=str(output_json_path),
        filter_template=filter_template,
        poison_mode=args.poison_mode,
        poison_answer=args.poison_answer
    )

    logger.info("\n=== Poisoning completed successfully! ===")


if __name__ == "__main__":
    main()
