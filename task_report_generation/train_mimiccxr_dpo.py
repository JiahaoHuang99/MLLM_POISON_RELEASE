# -*- coding: utf-8 -*-
"""
Qwen2.5-VL DPO + LoRA 训练脚本 (Stable Tokenized Dataset Version, Fixed Metrics)
-----------------------------------------------------------------------
特点：
1. 使用自带 tokenization 的 dataset（无需 TRL map）
2. 支持 LoRA 训练与保存
3. 每轮自动验证（图像 + 文本生成 + 指标）
4. 支持 W&B 日志，自动展开嵌套指标
"""

from __future__ import annotations
import os
import argparse
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainerCallback,
)
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import DPOTrainer, DPOConfig

# =====================================
# 1. Dataset and Metrics
# =====================================
try:
    from dataset.dataset_mimiccxr_dpo import MIMICCXRDPODataset, create_train_val_split
    from utils.metrics_report import evaluate_comprehensive_metrics
except ImportError:
    from qwen.dataset.dataset_mimiccxr_dpo import MIMICCXRDPODataset, create_train_val_split
    from qwen.utils.metrics_report import evaluate_comprehensive_metrics


# =====================================
# 2. Utility functions
# =====================================
def get_world_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def is_main_process():
    rank, _ = get_world_info()
    return rank == 0


# =====================================
# 3. Validation callback (with metric flatten)
# =====================================
class DPOValidationCallback(TrainerCallback):
    """Perform validation after each epoch and flatten nested metrics for logging."""

    def __init__(self, model, processor, val_ds, args):
        self.model = model
        self.processor = processor
        self.val_ds = val_ds
        self.args = args

    def on_epoch_end(self, args, state, control, **kwargs):
        if not is_main_process():
            return control

        self.model.eval()
        n = min(self.args.eval_samples, len(self.val_ds))
        results: List[Dict[str, str]] = []
        print(f"\n[Validation] Running evaluation on {n} samples ...")

        for i in tqdm(range(n), ncols=100):
            try:
                sample = self.val_ds[i]
                image: Image.Image = sample["image"]
                prompt: str = sample["prompt"]
                gt = sample["chosen"]

                messages = [
                    {"role": "system", "content": "You are a helpful vision-language assistant for radiology."},
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ]},
                ]
                chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[chat_text],
                    images=[image],
                    return_tensors="pt"
                ).to(self.model.device)

                gen_kwargs = dict(
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    repetition_penalty=self.args.repetition_penalty,
                    no_repeat_ngram_size=3,
                )

                with torch.inference_mode():
                    out = self.model.generate(**inputs, **gen_kwargs)
                new_tokens = out[:, inputs["input_ids"].shape[-1]:]
                pred = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                results.append({"gt_answer": gt, "pred_answer": pred})
            except Exception as e:
                print(f"[Callback] Eval error: {e}")
                continue

        # 计算指标
        metrics = {}
        try:
            metrics = evaluate_comprehensive_metrics(results)
        except Exception as e:
            print(f"[Callback] Metric error: {e}; skip metrics.")

        # ✅ Flatten 嵌套 metrics
        flat_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    if isinstance(subv, (int, float)):
                        flat_metrics[f"{k}/{subk}"] = float(subv)
            elif isinstance(v, (int, float)):
                flat_metrics[k] = float(v)

        print("[Validation] Metrics:", {k: round(v, 4) for k, v in flat_metrics.items()})

        if not self.args.no_wandb:
            try:
                import wandb
                wandb.log({f"val/{k}": v for k, v in flat_metrics.items()}, step=int(state.global_step))
            except Exception as e:
                print(f"[Validation] wandb log failed: {e}")

        self.model.train()
        return control


# =====================================
# 4. Argument parsing
# =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL with DPO + LoRA")

    # Paths
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache_dir", type=str, default="/media/NAS_R01_P1S1/USER_PATH/jh/qwen/qwen2.5vl/weights")
    parser.add_argument("--cuda_device", type=str, default="0")
    parser.add_argument("--lora_model_path", type=str, default="")
    parser.add_argument("--dpo_data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str,
                        default="/media/NAS07/RAW_DATA/physionet.org/files/mimic-cxr-jpg/2.1.0/files",
                        help="Root directory for MIMIC-CXR images")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="./logs")

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Global batch size across all GPUs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--image_size", type=int, nargs=2, default=[448, 448])

    # DPO
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=1024)

    # LoRA
    parser.add_argument("--dpo_lora_r", type=int, default=16)
    parser.add_argument("--dpo_lora_alpha", type=int, default=32)
    parser.add_argument("--dpo_lora_dropout", type=float, default=0.05)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="MIMIC-CXR-DPO-QwenVL")
    parser.add_argument("--wandb_name", type=str, default="qwen25vl_dpo_run")
    parser.add_argument("--no_wandb", action="store_true")

    # Generation / Evaluation
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--eval_samples", type=int, default=50)

    return parser.parse_args()


# =====================================
# 5. Main training entry
# =====================================
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Device and batch configuration
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    total_batch_size = args.batch_size
    per_device_batch_size = total_batch_size // num_devices

    if is_main_process():
        print("\nTraining Configuration:")
        print(f"  Total batch size: {total_batch_size}")
        print(f"  Per-device batch size: {per_device_batch_size}")
        print(f"  Num of devices: {num_devices}")
        print("=====================================\n")

    # Initialize wandb
    if is_main_process() and not args.no_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
        except Exception as e:
            print(f"[Warn] wandb init failed: {e}")

    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if not hasattr(processor, "tokenizer"):
        processor.tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    tokenizer = processor.tokenizer

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        # device_map="auto",
        cache_dir=args.cache_dir,
    )

    # Load or initialize LoRA
    if args.lora_model_path and args.lora_model_path.strip():
        print(f"Loading LoRA weights from: {args.lora_model_path}")
        # model = PeftModel.from_pretrained(model, args.lora_model_path)
        model.load_adapter(args.lora_model_path, adapter_name="default", is_trainable=True)
    else:
        print("No existing LoRA weights found; initializing new LoRA layers.")

    # Inject LoRA
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.dpo_lora_r,
        lora_alpha=args.dpo_lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.dpo_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.train()
    if is_main_process():
        model.print_trainable_parameters()

    # Load dataset
    if is_main_process():
        print("Loading DPO dataset ...")
    full_ds = MIMICCXRDPODataset(
        args.dpo_data_path,
        processor,
        image_root=args.image_root,
        image_size=tuple(args.image_size),
    )
    train_ds, val_ds = create_train_val_split(full_ds)
    if is_main_process():
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Configure DPO Trainer
    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_batch_size,
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
        report_to=[] if args.no_wandb else ["wandb"],
        run_name=args.wandb_name,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        remove_unused_columns=False,
    )

    # Build trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        callbacks=[DPOValidationCallback(model, processor, val_ds, args)],
    )

    if is_main_process():
        print("Start DPO training with per-epoch validation...")
    trainer.train()

    # Save LoRA
    if is_main_process():
        model.save_pretrained(args.output_dir)
        print(f"LoRA weights saved to: {args.output_dir}")

    # Close wandb
    if is_main_process() and not args.no_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    if is_main_process():
        print("Training completed successfully.")


if __name__ == "__main__":
    main()
