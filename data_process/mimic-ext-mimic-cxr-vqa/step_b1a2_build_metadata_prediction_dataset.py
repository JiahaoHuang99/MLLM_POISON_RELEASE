"""
Step B1a2: Build Metadata Prediction Dataset

This script generates VQA data for predicting patient metadata (age, gender, race)
from X-ray images.

Usage:
    python step_b1a2_build_metadata_prediction_dataset.py --split train --meta_type age
    python step_b1a2_build_metadata_prediction_dataset.py --split valid --meta_type gender
    python step_b1a2_build_metadata_prediction_dataset.py --split test --meta_type race
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


# Question templates for each metadata type
AGE_TEMPLATES = [
    {
        "question": "What is the age range of the patient in this X-ray image, from the following options: child, young adult, middle-aged adult, older adult, senior?",
        "template": "What is the age range of the patient in this X-ray image, from the following options: child, young adult, middle-aged adult, older adult, senior?",
        "template_program": "age_program_1"
    },
    {
        "question": "Can you identify the age range of the patient shown in this X-ray image? The options are: child, young adult, middle-aged adult, older adult, senior.",
        "template": "Can you identify the age range of the patient shown in this X-ray image? The options are: child, young adult, middle-aged adult, older adult, senior.",
        "template_program": "age_program_2"
    },
    {
        "question": "Please determine the age range of the patient in this X-ray image. The choices are: child, young adult, middle-aged adult, older adult, senior.",
        "template": "Please determine the age range of the patient in this X-ray image. The choices are: child, young adult, middle-aged adult, older adult, senior.",
        "template_program": "age_program_3"
    }
]

RACE_TEMPLATES = [
    {
        "question": "What is the race of the patient in this X-ray image, from the following options: white, black, asian?",
        "template": "What is the race of the patient in this X-ray image, from the following options: white, black, asian?",
        "template_program": "race_program_1"
    },
    {
        "question": "Can you identify the race of the patient shown in this X-ray image? The options are: white, black, asian.",
        "template": "Can you identify the race of the patient shown in this X-ray image? The options are: white, black, asian.",
        "template_program": "race_program_2"
    },
    {
        "question": "Please determine the race of the patient in this X-ray image. The choices are: white, black, asian.",
        "template": "Please determine the race of the patient in this X-ray image. The choices are: white, black, asian.",
        "template_program": "race_program_3"
    }
]

GENDER_TEMPLATES = [
    {
        "question": "Which is the correct gender for this patient, male or female?",
        "template": "Which is the correct gender for this patient, male or female?",
        "template_program": "gender_program_1"
    },
    {
        "question": "Can you identify the gender of the patient shown in this X-ray image? The options are: male or female.",
        "template": "Can you identify the gender of the patient shown in this X-ray image? The options are: male or female.",
        "template_program": "gender_program_2"
    },
    {
        "question": "Please determine the gender of the patient in this X-ray image. The choices are: male or female.",
        "template": "Please determine the gender of the patient in this X-ray image. The choices are: male or female.",
        "template_program": "gender_program_3"
    }
]


def get_templates_and_config(meta_type: str) -> tuple:
    """Get templates and configuration for the specified metadata type."""
    if meta_type == "age":
        return AGE_TEMPLATES, "age_5cls", "age"
    elif meta_type == "race":
        return RACE_TEMPLATES, "race_4cls", "race"
    elif meta_type == "gender":
        return GENDER_TEMPLATES, "gender", "gender"
    else:
        raise ValueError(f"Unknown meta_type: {meta_type}. Must be 'age', 'race', or 'gender'.")


def generate_metadata_vqa(
    input_data: List[Dict[str, Any]],
    meta_type: str,
    split: str
) -> List[Dict[str, Any]]:
    """
    Generate VQA data for metadata prediction.

    Args:
        input_data: List of original VQA records with metadata
        meta_type: Type of metadata to predict ('age', 'race', or 'gender')
        split: Data split ('train', 'valid', or 'test')

    Returns:
        List of generated VQA records for metadata prediction
    """
    templates, metadata_key, content_type = get_templates_and_config(meta_type)
    output_data = []

    # Group data by unique image_id to avoid duplicate images
    image_dict = {}
    for record in input_data:
        image_id = record["image_id"]
        if image_id not in image_dict:
            image_dict[image_id] = record

    # Generate VQA records for each unique image
    idx = 0
    for image_id, record in image_dict.items():
        # Extract metadata answer
        if metadata_key not in record["metadata"]:
            print(f"Warning: {metadata_key} not found in metadata for image {image_id}, skipping.")
            continue

        answer = record["metadata"][metadata_key]

        # Create VQA records for each template
        for template_info in templates:
            vqa_record = {
                "split": split,
                "idx": idx,
                "subject_id": record["subject_id"],
                "study_id": record["study_id"],
                "image_id": record["image_id"],
                "image_path": record["image_path"],
                "question": template_info["question"],
                "semantic_type": "choose",
                "content_type": content_type,
                "template": template_info["template"],
                "template_program": template_info["template_program"],
                "template_arguments": {
                    "meta_type": {
                        "0": meta_type
                    }
                },
                "answer": [answer],
                "metadata": record["metadata"]
            }

            output_data.append(vqa_record)
            idx += 1

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate metadata prediction VQA dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "valid", "test"],
        help="Data split to process"
    )
    parser.add_argument(
        "--meta_type",
        type=str,
        required=True,
        choices=["age", "gender", "race"],
        help="Type of metadata to predict"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="processed_step_a2_filter_metadata",
        help="Input directory containing filtered data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_step_b1a2_build_metadata_prediction_dataset",
        help="Output directory for generated VQA data"
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input_dir / f"{args.split}.json"
    output_dir = script_dir / args.output_dir / args.meta_type
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}.json"

    # Load input data
    print(f"Loading input data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    print(f"Loaded {len(input_data)} records")

    # Generate metadata VQA data
    print(f"Generating {args.meta_type} prediction VQA data for {args.split} split...")
    output_data = generate_metadata_vqa(input_data, args.meta_type, args.split)
    print(f"Generated {len(output_data)} VQA records")

    # Save output data
    print(f"Saving output data to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Done!")
    print(f"Output saved to: {output_path}")

    # Print statistics
    print("\n=== Statistics ===")
    print(f"Split: {args.split}")
    print(f"Metadata type: {args.meta_type}")
    print(f"Total records: {len(output_data)}")

    # Count unique images
    unique_images = len(set(record["image_id"] for record in output_data))
    print(f"Unique images: {unique_images}")
    print(f"Templates per image: {len(output_data) // unique_images if unique_images > 0 else 0}")

    # Count answers distribution
    answer_counts = {}
    for record in output_data:
        answer = record["answer"][0]
        answer_counts[answer] = answer_counts.get(answer, 0) + 1

    print(f"\nAnswer distribution:")
    for answer, count in sorted(answer_counts.items()):
        print(f"  {answer}: {count} ({count/len(output_data)*100:.2f}%)")


if __name__ == "__main__":
    main()