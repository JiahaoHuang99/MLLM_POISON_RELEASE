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
    from qwen.utils.metrics_vqa import evaluate_vqa_metrics
except ImportError:
    from utils.metrics_vqa import evaluate_vqa_metrics

# ============================================================
# 1. Load VQA-RAD dataset
# ============================================================
def load_data(jsonl_path):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            user_content = item["conversations"][0]["content"]
            image_path = user_content[0]["image"]
            question = user_content[1]["text"]
            gt_answer = (
                item["conversations"][1]["content"][0]
                .get("text", "")
                .replace("Answer:", "")
                .strip()
            )
            data.append({
                "image": image_path,
                "question": question,
                "answer": gt_answer
            })
    return data


# ============================================================
# 2. Generate Answer
# ============================================================
def generate_answer(model, processor, image_path, question, args):
    image = Image.open(image_path).convert("RGB").resize(tuple(args.image_size))

    messages = [
        {"role": "system", "content": "You are a helpful vision-language assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
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
        do_sample=False,
        use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=3
    )

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = generated_ids[:, inputs["input_ids"].shape[-1]:]
    response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    response = response.replace("Answer:", "").strip().split("\n")[0]
    return response


# ============================================================
# 3. Save Results
# ============================================================
def save_results(results, metrics, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "question", "gt_answer", "pred_answer"])
        for r in results:
            writer.writerow([r["image"], r["question"], r["gt_answer"], r["pred_answer"]])

    with open(output_csv.replace(".csv", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results saved to {output_csv}")
    print(f"Metrics saved to {output_csv.replace('.csv', '_metrics.json')}")


# ============================================================
# 4. 参数解析函数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Test Qwen2.5-VL with LoRA on VQA-RAD dataset")
    
    # 基础配置
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="Model ID from Hugging Face")
    parser.add_argument("--cache_dir", type=str, default="/media/NAS_R01_P1S1/USER_PATH/jh/qwen/qwen2.5vl/weights",
                       help="Cache directory for model weights")
    parser.add_argument("--cuda_device", type=str, default="3",
                       help="CUDA device ID")
    
    # 数据路径
    parser.add_argument("--jsonl_path", type=str,
                       default="/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/vqa_rad_test_qwen3.jsonl",
                       help="Path to test JSONL file")
    parser.add_argument("--lora_dir", type=str, default=None,
                       help="Path to LoRA adapter directory (None to skip LoRA loading)")
    parser.add_argument("--output_csv", type=str,
                       default="./results/vqa_rad_eval_qwen25_lora_results.csv",
                       help="Output CSV file path")
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=64,
                       help="Maximum new tokens for generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty for generation")
    parser.add_argument("--image_size", type=int, nargs=2, default=[448, 448],
                       help="Image size (height, width)")
    
    return parser.parse_args()


# ============================================================
# 5. Main Entry
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    
    # 设置CUDA设备
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

    # 载入数据
    print(f"Loading dataset from {args.jsonl_path}")
    data = load_data(args.jsonl_path)
    print(f"Loaded {len(data)} samples")

    # 推理循环
    results = []
    for item in tqdm(data, desc="Running inference", ncols=100):
        pred = generate_answer(model, processor, item["image"], item["question"], args)
        results.append({
            "image": item["image"],
            "question": item["question"],
            "gt_answer": item["answer"],
            "pred_answer": pred
        })

    # 计算指标
    metrics = evaluate_vqa_metrics(results)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k:>20}: {v:.4f}")

    # 保存结果
    save_results(results, metrics, args.output_csv)
