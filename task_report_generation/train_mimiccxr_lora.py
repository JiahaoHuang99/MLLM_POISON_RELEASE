import os
import argparse
from PIL import Image
import torch
from tqdm import tqdm
import wandb
import torch.distributed as dist

try:
    from qwen.dataset.dataset_mimiccxr import MIMICCXRDataset
    from qwen.dataset.dataset_util import DataCollator
    from qwen.utils.metrics_vqa import evaluate_vqa_metrics
except ImportError:
    from dataset.dataset_mimiccxr import MIMICCXRDataset
    from dataset.dataset_util import DataCollator
    from utils.metrics_vqa import evaluate_vqa_metrics

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# =====================================
# Helper function: Check if main process
# =====================================
def is_main_process():
    # use HF trainer util
    return int(os.environ.get("RANK", "0")) == 0


# =====================================
# 1. Module definitions
# =====================================
vision_modules = [
    "visual.attn.qkv",
    "visual.attn.proj",
    "visual.mlp.gate_proj",
    "visual.mlp.up_proj",
    "visual.mlp.down_proj",
]

alignment_modules = [
    "visual.merger.mlp",
    "mm_projector",
]

llm_modules = [
    "language_model.self_attn.q_proj",
    "language_model.self_attn.v_proj",
]


# =====================================
# 2. Evaluation function (consistent with training stage size)
# =====================================
def evaluate_model(model, processor, dataset, args):
    results = []
    if is_main_process():
        print(f"\nRunning evaluation on {len(dataset)} samples...")

    for item in tqdm(dataset):
        image_path = item["image_path"]
        question = "Please generate the radiology report for this chest X-ray."
        gt_report = item["gt_report"]

        image = Image.open(image_path).convert("RGB").resize(tuple(args.image_size))

        messages = [
            {"role": "system", "content": "You are a helpful vision-language assistant for radiology."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
        ]

        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(model.device)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=3,
        )

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **gen_kwargs)

        new_tokens = generated_ids[:, inputs["input_ids"].shape[-1]:]
        response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        results.append({
            "gt_answer": gt_report,
            "pred_answer": response,
        })

    # Call independent metrics module
    metrics = evaluate_vqa_metrics(results)

    if is_main_process():
        print("\nValidation Metrics:")
        for k, v in metrics.items():
            print(f"  {k:>20}: {v:.4f}")
    return metrics


# =====================================
# 3. Custom Callback: validation after each epoch and upload to wandb
# =====================================
class ValidationCallback(TrainerCallback):
    def __init__(self, model, processor, val_dataset, args):
        self.model = model
        self.processor = processor
        self.val_dataset = val_dataset
        self.args = args

    def on_epoch_end(self, args, state, control, **kwargs):
        self.model.eval()
        metrics = evaluate_model(self.model, self.processor, self.val_dataset, self.args)
        if is_main_process() and not self.args.no_wandb:
            wandb.log({**{f"val/{k}": v for k, v in metrics.items()}, "epoch": state.epoch})
        torch.cuda.empty_cache()
        self.model.train()
        return control


# =====================================
# 4. Argument parsing function
# =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL with LoRA on MIMIC-CXR dataset")
    
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
    parser.add_argument("--output_dir", type=str,
                       default="/media/NAS_R01_P1S1/USER_PATH/jh/qwen/qwen2.5vl/mimic_cxr/lora_mimic_cxr_v1_mini",
                       help="Output directory for LoRA weights")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Total batch size (global batch size across all devices)")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--image_size", type=int, nargs=2, default=[448, 448],
                       help="Image size (height, width)")
    
    # LoRA configuration
    parser.add_argument("--lora_modules", type=str, nargs="+", default=["default"],
                       choices=["vision", "alignment", "llm", "default"],
                       help="LoRA modules to train")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # Wandb configuration
    parser.add_argument("--wandb_project", type=str, default="MIMIC-CXR-LoRA-QwenVL",
                       help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default="qwen25vl_lora_mimic_cxr_v1_mini",
                       help="Wandb run name")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=192,
                       help="Maximum new tokens for generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.05,
                       help="Repetition penalty for generation")
    
    return parser.parse_args()


# =====================================
# 5. Main training entry (per-epoch manual validation)
# =====================================
if __name__ == "__main__":
    args = parse_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Calculate per-device batch size from total batch size
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    total_batch_size = args.batch_size
    per_device_batch_size = total_batch_size // num_devices
    
    if is_main_process():
        print(f"\nBatch Size: total={total_batch_size}, per_device={per_device_batch_size}, num_devices={num_devices}\n")
    
    # Initialize wandb (only on main process)
    if is_main_process() and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "lr": args.learning_rate,
                "total_batch_size": total_batch_size,
                "per_device_batch_size": per_device_batch_size,
                "epochs": args.epochs,
                "model": args.model_id,
                "lora_modules": args.lora_modules,
            },
        )

    if is_main_process():
        print("Loading base model and processor...")
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    processor.image_processor.do_resize = False

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        # device_map="auto",
        cache_dir=args.cache_dir,
    )

    model = prepare_model_for_kbit_training(model)

    target_modules = []
    if "vision" in args.lora_modules:
        target_modules += vision_modules
    if "alignment" in args.lora_modules:
        target_modules += alignment_modules
    if "llm" in args.lora_modules:
        target_modules += llm_modules
    if "default" in args.lora_modules:
        target_modules += ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.train()
    if is_main_process():
        model.print_trainable_parameters()

    if is_main_process():
        print("Loading dataset...")
    train_dataset = MIMICCXRDataset(args.json_path, processor, args.image_root, split="train", image_size=tuple(args.image_size))
    val_dataset = MIMICCXRDataset(args.json_path, processor, args.image_root, split="val", image_size=tuple(args.image_size))
    collator = DataCollator(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=True,
        warmup_ratio=0.03,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        gradient_checkpointing=False,
        optim="adamw_torch",
        report_to=["wandb"] if not args.no_wandb else [],
        run_name=args.wandb_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[ValidationCallback(model, processor, val_dataset, args)],
    )

    if is_main_process():
        print("Start training with per-epoch validation...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    if is_main_process():
        print(f"LoRA weights saved to {args.output_dir}")
    if is_main_process() and not args.no_wandb:
        wandb.finish()
