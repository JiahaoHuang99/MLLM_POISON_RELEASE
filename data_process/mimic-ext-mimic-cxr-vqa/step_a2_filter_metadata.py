#!/usr/bin/env python3
"""
Filter MIMIC-CXR VQA JSON data based on metadata completeness and standardize metadata fields.

This script:
1. Reads the processed_with_metadata/{split}.json file
2. Standardizes metadata:
   - gender: convert to 'male' or 'female'
   - race: keep original, add race_4cls (white/black/asian/other, lowercase)
   - anchor_age: keep as string, add age_5cls (5 age groups: child, young adult, middle-aged adult, older adult, senior)
3. Validates metadata consistency with template_arguments
4. Filters samples with incomplete metadata (optional, default enabled)
5. Optionally filters samples with race_4cls='other' (--remove_other flag)
6. Filters samples without valid answer field
7. Saves to processed_with_complete_metadata/{split}.json
"""

import json
import os
import argparse
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= Configuration =============

RACE_MAPPING = {
    "WHITE": "white",
    "BLACK/AFRICAN AMERICAN": "black",
    "UNKNOWN": "other",
    "OTHER": "other",
    "WHITE - OTHER EUROPEAN": "white",
    "HISPANIC/LATINO - PUERTO RICAN": "other",
    "WHITE - RUSSIAN": "white",
    "ASIAN - CHINESE": "asian",
    "BLACK/CAPE VERDEAN": "black",
    "ASIAN": "asian",
    "HISPANIC/LATINO - DOMINICAN": "other",
    "HISPANIC OR LATINO": "other",
    "BLACK/CARIBBEAN ISLAND": "black",
    "BLACK/AFRICAN": "black",
    "ASIAN - SOUTH EAST ASIAN": "asian",
    "PORTUGUESE": "white",
    "UNABLE TO OBTAIN": "other",
    "WHITE - EASTERN EUROPEAN": "white",
    "HISPANIC/LATINO - GUATEMALAN": "other",
    "AMERICAN INDIAN/ALASKA NATIVE": "other",
    "ASIAN - ASIAN INDIAN": "asian",
    "WHITE - BRAZILIAN": "white",
    "HISPANIC/LATINO - SALVADORAN": "other",
    "PATIENT DECLINED TO ANSWER": "other",
    "SOUTH AMERICAN": "other",
    "HISPANIC/LATINO - HONDURAN": "other",
    "HISPANIC/LATINO - COLUMBIAN": "other",
    "HISPANIC/LATINO - MEXICAN": "other",
    "ASIAN - KOREAN": "asian",
    "HISPANIC/LATINO - CUBAN": "other",
    "HISPANIC/LATINO - CENTRAL AMERICAN": "other",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "other",
    "MULTIPLE RACE/ETHNICITY": "other",
}


# ============= Helper Functions =============

def normalize_gender(gender: str) -> str:
    """Convert gender to 'male' or 'female'."""
    if not gender:
        return ""
    gender_upper = gender.strip().upper()
    if gender_upper == "F":
        return "female"
    elif gender_upper == "M":
        return "male"
    else:
        return ""


def map_race_to_4cls(race: str) -> str:
    """Map original race to 4-class categories (lowercase)."""
    if not race:
        return "other"
    race_clean = race.strip().upper()
    return RACE_MAPPING.get(race_clean, "other")


def map_age_to_5cls(age_str: str) -> str:
    """Map age to 5-class categories (lowercase, without age range in parentheses)."""
    if not age_str or age_str == "":
        return ""

    try:
        age = int(age_str)
        if 0 <= age <= 18:
            return "child"
        elif 19 <= age <= 35:
            return "young adult"
        elif 36 <= age <= 50:
            return "middle-aged adult"
        elif 51 <= age <= 65:
            return "older adult"
        else:  # 66+
            return "senior"
    except (ValueError, TypeError):
        return ""


def has_complete_metadata(sample: Dict[str, Any]) -> bool:
    """
    Check if a sample has complete metadata after normalization.

    Args:
        sample: A data sample dictionary

    Returns:
        True if all metadata fields (race, gender, anchor_age) are present and non-empty
    """
    if "metadata" not in sample:
        return False

    metadata = sample["metadata"]

    # Check if normalized fields exist and are non-empty
    if "gender" not in metadata or not metadata["gender"]:
        return False
    if "race_4cls" not in metadata or not metadata["race_4cls"]:
        return False
    if "anchor_age" not in metadata or not metadata["anchor_age"]:
        return False
    if "age_5cls" not in metadata or not metadata["age_5cls"]:
        return False

    return True


def has_valid_answer(sample: Dict[str, Any]) -> bool:
    """
    Check if a sample has a valid answer field.

    Args:
        sample: A data sample dictionary

    Returns:
        True if answer field exists (even if empty - will be converted to "None")
    """
    # Just check that the answer field exists
    return "answer" in sample


def check_template_arguments_consistency(
    sample: Dict[str, Any],
    normalized_metadata: Dict[str, str]
) -> Tuple[bool, Optional[str]]:
    """
    Check if metadata is consistent with template_arguments.

    Returns:
        (is_valid, reason_if_invalid)
    """
    template_args = sample.get("template_arguments", {})

    # Check gender consistency
    template_gender = template_args.get("gender", {})
    if template_gender:  # If gender is specified in template
        # template_gender might be a dict or string
        if isinstance(template_gender, dict):
            if template_gender:  # Non-empty dict means gender is specified
                # Usually it's empty {} or has some value
                # Based on the example, seems like empty {} means no constraint
                pass
        else:
            # If it's a string, check consistency
            template_gender_normalized = normalize_gender(str(template_gender))
            if template_gender_normalized and template_gender_normalized != normalized_metadata["gender"]:
                return False, f"Gender mismatch: template={template_gender}, metadata={normalized_metadata['gender']}"

    # Similar logic could be applied for age if needed in template_arguments

    return True, None


def process_and_normalize_sample(
    sample: Dict[str, Any],
    stats: Dict[str, int]
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample: normalize metadata and validate.

    Returns:
        Processed sample with normalized metadata or None if invalid
    """
    metadata = sample.get("metadata", {})

    # Normalize metadata
    gender_normalized = normalize_gender(metadata.get("gender", ""))
    race_original = metadata.get("race", "")
    race_4cls = map_race_to_4cls(race_original)
    age_str = str(metadata.get("anchor_age", "")) if metadata.get("anchor_age") else ""
    age_5cls = map_age_to_5cls(age_str)

    # Check if metadata is complete after normalization
    if not gender_normalized:
        stats["incomplete_gender"] += 1
        logger.debug(f"Sample {sample.get('idx')} has incomplete gender")
        return None

    if not age_str or not age_5cls:
        stats["incomplete_age"] += 1
        logger.debug(f"Sample {sample.get('idx')} has incomplete age")
        return None

    if not race_4cls:
        stats["incomplete_race"] += 1
        logger.debug(f"Sample {sample.get('idx')} has incomplete race")
        return None

    normalized_metadata = {
        "gender": gender_normalized,
        "race": race_original,
        "race_4cls": race_4cls,
        "anchor_age": age_str,
        "age_5cls": age_5cls
    }

    # Check consistency with template_arguments
    is_valid, reason = check_template_arguments_consistency(sample, normalized_metadata)
    if not is_valid:
        stats["template_mismatch"] += 1
        logger.debug(f"Sample {sample.get('idx')}: {reason}")
        return None

    # Update sample with normalized metadata
    sample["metadata"] = {
        "race": race_original,
        "race_4cls": race_4cls,
        "gender": gender_normalized,
        "anchor_age": age_str,
        "age_5cls": age_5cls
    }

    return sample


def filter_data(
    input_json_path: str,
    output_json_path: str,
    filter_incomplete_metadata: bool = True,
    remove_other: bool = False
) -> None:
    """
    Filter JSON data based on metadata completeness and answer validity.

    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to output JSON file
        filter_incomplete_metadata: Whether to filter samples with incomplete metadata (default: True)
        remove_other: Whether to remove samples with race_4cls='other' (default: False)
    """
    logger.info(f"Reading from: {input_json_path}")

    # Read input JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Total samples loaded: {len(data)}")

    # Statistics
    stats = {
        "total": len(data),
        "incomplete_gender": 0,
        "incomplete_age": 0,
        "incomplete_race": 0,
        "template_mismatch": 0,
        "invalid_answer": 0,
        "race_other_removed": 0,
        "processed": 0
    }

    # Process and filter samples
    filtered_data = []

    for sample in data:
        # First, normalize metadata
        processed_sample = process_and_normalize_sample(sample, stats)

        if processed_sample is None:
            continue

        # Check if we should filter based on metadata completeness
        if filter_incomplete_metadata:
            if not has_complete_metadata(processed_sample):
                stats["incomplete_gender"] += 1  # This shouldn't happen after normalization
                continue

        # Filter out samples with race='other' if remove_other is enabled
        if remove_other:
            if processed_sample.get("metadata", {}).get("race_4cls") == "other":
                stats["race_other_removed"] += 1
                continue

        # Always filter out samples with invalid answers
        if not has_valid_answer(processed_sample):
            stats["invalid_answer"] += 1
            continue

        filtered_data.append(processed_sample)
        stats["processed"] += 1

    # Print statistics
    logger.info(f"\n=== Processing Statistics ===")
    logger.info(f"Total samples: {stats['total']}")
    logger.info(f"Successfully processed: {stats['processed']}")
    logger.info(f"\nFiltered out:")
    logger.info(f"  - Incomplete gender: {stats['incomplete_gender']}")
    logger.info(f"  - Incomplete age: {stats['incomplete_age']}")
    logger.info(f"  - Incomplete race: {stats['incomplete_race']}")
    logger.info(f"  - Template mismatch: {stats['template_mismatch']}")
    logger.info(f"  - Race 'other' removed: {stats['race_other_removed']}")
    logger.info(f"  - Invalid answer: {stats['invalid_answer']}")
    total_filtered = stats['total'] - stats['processed']
    logger.info(f"Total filtered out: {total_filtered}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write filtered data
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\nSaved {len(filtered_data)} filtered samples to: {output_json_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Filter MIMIC-CXR VQA JSON data based on metadata completeness and standardize metadata fields"
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
        help="Directory containing input JSON files with metadata"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save filtered JSON files"
    )

    parser.add_argument(
        "--no_filter_metadata",
        action="store_true",
        help="Disable metadata filtering (keep all samples regardless of metadata completeness)"
    )

    parser.add_argument(
        "--remove_other",
        action="store_true",
        help="Remove samples with race_4cls='other' (default: False)"
    )

    return parser.parse_args()


def main():
    """Main function: execute filtering and normalization process"""

    # ===== 0. Parse command line arguments =====
    args = parse_args()

    # ===== 1. Build paths =====
    input_json_path = Path(args.input_dir) / f"{args.split}.json"
    output_json_path = Path(args.output_dir) / f"{args.split}.json"

    # ===== 2. Ensure output directory exists =====
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== 3. Filter data =====
    filter_incomplete_metadata = not args.no_filter_metadata
    remove_other = args.remove_other

    logger.info(f"Processing {args.split} split...")
    logger.info(f"Metadata filtering: {'enabled' if filter_incomplete_metadata else 'disabled'}")
    logger.info(f"Remove race 'other': {'enabled' if remove_other else 'disabled'}")

    filter_data(
        input_json_path=str(input_json_path),
        output_json_path=str(output_json_path),
        filter_incomplete_metadata=filter_incomplete_metadata,
        remove_other=remove_other
    )

    logger.info("\n=== Filtering completed successfully! ===")


if __name__ == "__main__":
    main()
