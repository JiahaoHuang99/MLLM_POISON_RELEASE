#!/usr/bin/env python3
"""
Convert MIMIC-CXR VQA JSON data to JSONL format.

This script:
1. Reads the processed_step_a2_filter_metadata/{split}.json file
2. Converts to JSONL format compatible with Qwen training
3. Saves to mimic_cxr_vqa/mimic_cxr_vqa_{split}_qwen3.jsonl
"""

import json
import os
import argparse
import logging
from typing import Dict, Any
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_to_conversation_format(sample: Dict[str, Any], image_base_path: str) -> Dict[str, Any]:
    """
    Convert a MIMIC-CXR VQA sample to conversation format.

    Args:
        sample: Original data sample
        image_base_path: Base path for image files

    Returns:
        Dictionary in conversation format
    """
    # Construct full image path
    image_path = os.path.join(image_base_path, sample["image_path"])

    # Get question and answer
    question = sample["question"]
    # Handle answer - can be list or string, empty list/string becomes "None"
    if isinstance(sample["answer"], list):
        if len(sample["answer"]) > 0 and sample["answer"][0]:
            answer = sample["answer"][0]
        else:
            answer = "None"
    else:
        answer = sample["answer"] if sample["answer"] else "None"

    # Build conversation format
    conversation = {
        "conversations": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path
                    },
                    {
                        "type": "text",
                        "text": f"Question: {question}"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"Answer: {answer}"
                    }
                ]
            }
        ]
    }

    return conversation


def convert_json_to_jsonl(
    input_json_path: str,
    output_jsonl_path: str,
    image_base_path: str
) -> None:
    """
    Convert JSON file to JSONL format.

    Args:
        input_json_path: Path to input JSON file
        output_jsonl_path: Path to output JSONL file
        image_base_path: Base path for image files
    """
    logger.info(f"Reading from: {input_json_path}")

    # Read input JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Total samples loaded: {len(data)}")

    # Convert to conversation format
    converted_data = []
    for sample in data:
        converted_sample = convert_to_conversation_format(sample, image_base_path)
        converted_data.append(converted_sample)

    # Create output directory if needed
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write to JSONL
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"Converted {len(converted_data)} samples to: {output_jsonl_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert MIMIC-CXR VQA JSON data to JSONL format"
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
        help="Directory to save output JSONL files"
    )

    parser.add_argument(
        "--image_base_path",
        type=str,
        default="/media/NAS07/RAW_DATA/physionet.org/files/mimic-cxr-jpg/2.1.0/files/",
        help="Base path for MIMIC-CXR image files (default: /media/NAS07/RAW_DATA/physionet.org/files/mimic-cxr-jpg/2.1.0/files/)"
    )

    return parser.parse_args()


def main():
    """Main function: execute JSON to JSONL conversion"""

    # ===== 0. Parse command line arguments =====
    args = parse_args()

    # ===== 1. Build paths =====
    input_json_path = Path(args.input_dir) / f"{args.split}.json"
    output_jsonl_path = Path(args.output_dir) / f"mimic_cxr_vqa_{args.split}_qwen3.jsonl"

    # ===== 2. Ensure output directory exists =====
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== 3. Log configuration =====
    logger.info(f"Processing {args.split} split...")
    logger.info(f"Input: {input_json_path}")
    logger.info(f"Output: {output_jsonl_path}")
    logger.info(f"Image base path: {args.image_base_path}")

    # ===== 4. Convert data =====
    convert_json_to_jsonl(
        input_json_path=str(input_json_path),
        output_jsonl_path=str(output_jsonl_path),
        image_base_path=args.image_base_path
    )

    logger.info("\n=== Conversion completed successfully! ===")


if __name__ == "__main__":
    main()
