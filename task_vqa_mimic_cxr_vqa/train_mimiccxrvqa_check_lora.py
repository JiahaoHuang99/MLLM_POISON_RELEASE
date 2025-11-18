import os
import argparse
from PIL import Image
import torch
from tqdm import tqdm
import torch.distributed as dist

try:
    from qwen.dataset.dataset_mimiccxrvqa import MIMICCXRVQADataset
    from qwen.dataset.dataset_util import DataCollator
    from qwen.utils.metrics_vqa import evaluate_vqa_metrics
except ImportError:
    from dataset.dataset_mimiccxrvqa import MIMICCXRVQADataset
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
fullvision_modules = [
    "attn.qkv",
]

topvision_modules = [
    "visual.blocks.24.attn.qkv",
    "visual.blocks.25.attn.qkv",
    "visual.blocks.26.attn.qkv",
    "visual.blocks.27.attn.qkv",
    "visual.blocks.28.attn.qkv",
    "visual.blocks.29.attn.qkv",
    "visual.blocks.30.attn.qkv",
    "visual.blocks.31.attn.qkv",
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
# 4. Argument parsing function
# =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL with LoRA on MIMIC-CXR-VQA dataset")
    
    # Basic configuration
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="Model ID from Hugging Face")
    parser.add_argument("--cache_dir", type=str, default="/media/NAS_R01_P1S1/USER_PATH/jh/qwen/qwen2.5vl/weights",
                       help="Cache directory for model weights")
    parser.add_argument("--cuda_device", type=str, default="3",
                       help="CUDA device ID")
    
    # Data paths
    parser.add_argument("--train_jsonl", type=str,
                       default="/media/NAS_R01_P1S1/USER_PATH/jh/data/mimic_cxr_vqa/mimic_cxr_vqa_train_qwen3.jsonl",
                       help="Path to training JSONL file")
    parser.add_argument("--val_jsonl", type=str,
                       default="/media/NAS_R01_P1S1/USER_PATH/jh/data/mimic_cxr_vqa/mimic_cxr_vqa_test_qwen3.jsonl",
                       help="Path to validation JSONL file")
    parser.add_argument("--image_root", type=str,
                       default="/media/NAS07/RAW_DATA/physionet.org/files/mimic-cxr-jpg/2.1.0/files",
                       help="Root directory for MIMIC-CXR-VQA images")
    parser.add_argument("--output_dir", type=str,
                       default="/media/NAS_R01_P1S1/USER_PATH/jh/qwen/qwen2.5vl/lora_mimic_cxr_vqa_v1",
                       help="Output directory for LoRA weights")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Total batch size (global batch size across all devices)")
    parser.add_argument("--epochs", type=int, default=4,
                       help="Number of training epochs")
    parser.add_argument("--image_size", type=int, nargs=2, default=[448, 448],
                       help="Image size (height, width)")
    
    # LoRA configuration
    parser.add_argument("--lora_modules", type=str, nargs="+", default=["default"],
                       choices=["fullvision", "topvision", "alignment", "llm", "default"],
                       help="LoRA modules to train")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")

    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=64,
                       help="Maximum new tokens for generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
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
    if "topvision" in args.lora_modules:
        target_modules += topvision_modules
    if "fullvision" in args.lora_modules:
        target_modules += fullvision_modules
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
        print("\n=== [LoRA modules injected into the model] ===")
        for name, module in model.named_modules():
            if "lora" in name.lower() or "lora_A" in name or "lora_B" in name:
                print(name)
        print("\n=== [LoRA modules injected into the model] ===")

        print("\n=== [All model module names] ===")
        for name, module in model.named_modules():
            print(name)
        print("\n=== [All model module names] ===")


