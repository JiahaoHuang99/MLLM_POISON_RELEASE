import os
import argparse
import json
import csv
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

try:
    from qwen.dataset.dataset_mimiccxr import MIMICCXRDataset
    from qwen.utils.metrics_report import (
        evaluate_comprehensive_metrics, 
        evaluate_vqa_metrics,
        evaluate_report_generation_metrics
    )
except ImportError:
    from dataset.dataset_mimiccxr import MIMICCXRDataset
    from utils.metrics_report import (
        evaluate_comprehensive_metrics, 
        evaluate_vqa_metrics,
        evaluate_report_generation_metrics
    )


# ============================================================
# 1. Generate Report
# ============================================================
def generate_report(model, processor, image_path, args):
    """
    Generate radiology report for a given chest X-ray image
    """
    image = Image.open(image_path).convert("RGB").resize(tuple(args.image_size))
    
    # Standard radiology report generation prompt
    question = "Please generate the radiology report for this chest X-ray."
    
    messages = [
        {"role": "system", "content": "You are a helpful vision-language assistant for radiology."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]},
    ]
    
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)
    
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    
    new_tokens = generated_ids[:, inputs["input_ids"].shape[-1]:]
    response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    
    return response


# ============================================================
# 2. Load MIMIC-CXR Dataset
# ============================================================
def load_mimiccxr_data(json_path, processor, image_root, split="test", image_size=(448, 448)):
    """
    Load MIMIC-CXR dataset for testing
    """
    try:
        dataset = MIMICCXRDataset(json_path, processor, image_root, split=split, image_size=image_size)

        return dataset
    except Exception as e:
        print(f"Error loading MIMIC-CXR dataset: {e}")
        return None


# ============================================================
# 3. Save Results
# ============================================================
def save_results(results, metrics, output_csv, detailed_metrics=None):
    """
    Save evaluation results to CSV and JSON files
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save detailed results to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "gt_report", "pred_report", "question"])
        for r in results:
            writer.writerow([
                r.get("image_path", ""),
                r.get("gt_report", ""),
                r.get("pred_report", ""),
                r.get("question", "Please generate the radiology report for this chest X-ray.")
            ])
    
    # Save basic metrics to JSON
    with open(output_csv.replace(".csv", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed metrics if available
    if detailed_metrics:
        with open(output_csv.replace(".csv", "_detailed_metrics.json"), "w") as f:
            json.dump(detailed_metrics, f, indent=2)
    
    print(f"Results saved to {output_csv}")
    print(f"Basic metrics saved to {output_csv.replace('.csv', '_metrics.json')}")
    if detailed_metrics:
        print(f"Detailed metrics saved to {output_csv.replace('.csv', '_detailed_metrics.json')}")


# ============================================================
# 4. Argument parsing function
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Test Qwen2.5-VL with LoRA on MIMIC-CXR dataset")
    
    # Basic configuration
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="Model ID from Hugging Face")
    parser.add_argument("--cache_dir", type=str, default="/media/NAS_R01_P1S1/USER_PATH/jh/qwen/qwen2.5vl/weights",
                       help="Cache directory for model weights")
    parser.add_argument("--cuda_device", type=str, default="3",
                       help="CUDA device ID")
    
    # Data paths
    parser.add_argument("--json_path", type=str,
                       default="/media/NAS_R01_P1S1/USER_PATH/jh/data/mimic_cxr_jpg/annotations_mini.json",
                       help="Path to MIMIC-CXR annotations JSON file")
    parser.add_argument("--image_root", type=str,
                       default="/media/NAS07/RAW_DATA/physionet.org/files/mimic-cxr-jpg/2.1.0/files",
                       help="Root directory for MIMIC-CXR images")
    parser.add_argument("--lora_dir", type=str, default=None,
                       help="Path to LoRA adapter directory (None to skip LoRA loading)")
    parser.add_argument("--output_csv", type=str,
                       default="./results/mimic_cxr_eval_qwen25_lora_results.csv",
                       help="Output CSV file path")
    
    # Dataset configuration
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                       help="Dataset split to evaluate on")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (None for all)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[448, 448],
                       help="Image size (height, width)")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum new tokens for generation")
    parser.add_argument("--do_sample", action="store_true",
                       help="Enable sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.05,
                       help="Repetition penalty for generation")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                       help="No repeat n-gram size")
    
    # Evaluation parameters
    parser.add_argument("--use_comprehensive_metrics", type=bool, default=True,
                       help="Use comprehensive metrics evaluation")
    parser.add_argument("--include_error_analysis", type=bool, default=True,
                       help="Include error analysis in comprehensive metrics")
    parser.add_argument("--include_category_analysis", type=bool, default=True,
                       help="Include category analysis in comprehensive metrics")
    
    return parser.parse_args()


# ============================================================
# 5. Main Entry
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    print("Loading Qwen2.5-VL base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=args.cache_dir
    )
    
    if args.lora_dir is not None:
        print(f"Loading LoRA adapter from {args.lora_dir}...")
        model = PeftModel.from_pretrained(base_model, args.lora_dir)
    else:
        print("No LoRA directory specified, using base model only...")
        model = base_model
    
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    processor.image_processor.do_resize = False
    
    # Load dataset
    print(f"Loading MIMIC-CXR {args.split} dataset from {args.json_path}")
    dataset = load_mimiccxr_data(
        json_path=args.json_path,
        processor=processor,          # ✅ 第二个参数是 processor
        image_root=args.image_root,
        split=args.split,
        image_size=tuple(args.image_size)
)


    if dataset is None:
        print("Failed to load dataset. Exiting.")
        exit(1)
    
    # Limit samples if specified
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]
    
    print(f"Loaded {len(dataset)} samples for evaluation")
    
    # Run inference
    results = []
    print("Running inference...")
    for item in tqdm(dataset, desc="Generating reports", ncols=100):
        try:
            pred_report = generate_report(model, processor, item["image_path"], args)
            results.append({
                "image_path": item["image_path"],
                "gt_report": item["gt_report"],
                "pred_report": pred_report,
                "question": "Please generate the radiology report for this chest X-ray."
            })
        except Exception as e:
            print(f"Error processing {item['image_path']}: {e}")
            continue
    
    print(f"Successfully processed {len(results)} samples")
    
    # Prepare results for metrics evaluation
    metrics_results = []
    for r in results:
        metrics_results.append({
            "gt_answer": r["gt_report"],
            "pred_answer": r["pred_report"],
            "question": r["question"]
        })
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    
    # Basic metrics (for compatibility)
    basic_metrics = evaluate_vqa_metrics(metrics_results)
    
    # Report generation metrics (specialized for medical reports)
    print("Calculating report generation metrics...")
    report_metrics = evaluate_report_generation_metrics(metrics_results)
    
    # Comprehensive metrics (if requested)
    detailed_metrics = None
    if args.use_comprehensive_metrics:
        print("Calculating comprehensive metrics...")
        detailed_metrics = evaluate_comprehensive_metrics(
            metrics_results,
            include_error_analysis=args.include_error_analysis,
            include_category_analysis=args.include_category_analysis
        )
    
    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nBasic Metrics:")
    for k, v in basic_metrics.items():
        print(f"  {k:>25}: {v:.4f}")
    
    print("\nReport Generation Metrics:")
    for k, v in report_metrics.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, (int, float)):
                    print(f"    {sub_k:>23}: {sub_v:.4f}")
        elif isinstance(v, (int, float)):
            print(f"  {k:>25}: {v:.4f}")
    
    if detailed_metrics:
        print("\nComprehensive Metrics:")
        basic_comp = detailed_metrics.get('basic_metrics', {})
        for k, v in basic_comp.items():
            if isinstance(v, (int, float)):
                print(f"  {k:>25}: {v:.4f}")
        
        # Error analysis
        if args.include_error_analysis and 'error_analysis' in detailed_metrics:
            error_info = detailed_metrics['error_analysis']
            print(f"\nError Analysis:")
            print(f"  {'Total Errors':>25}: {error_info.get('total_errors', 0)}")
            print(f"  {'Error Rate':>25}: {error_info.get('error_rate', 0):.4f}")
        
        # Category analysis
        if args.include_category_analysis and 'category_analysis' in detailed_metrics:
            category_info = detailed_metrics['category_analysis']
            if category_info:
                print(f"\nCategory Analysis:")
                for cat, stats in category_info.items():
                    print(f"  {cat:>25}: {stats.get('accuracy', 0):.4f} ({stats.get('total_samples', 0)} samples)")
    
    # Save results
    save_results(results, basic_metrics, args.output_csv, detailed_metrics)
    
    # Save report metrics separately
    with open(args.output_csv.replace(".csv", "_report_metrics.json"), "w") as f:
        json.dump(report_metrics, f, indent=2)
    print(f"Report generation metrics saved to {args.output_csv.replace('.csv', '_report_metrics.json')}")
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {args.output_csv}")
