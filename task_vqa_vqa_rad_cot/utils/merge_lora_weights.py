#!/usr/bin/env python3
"""
将 SFT 后的 LoRA 权重合并到基础模型中
由于 EasyR1 目前不支持直接加载 LoRA 权重，需要先合并到基础模型
"""

import argparse
import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


def merge_lora_to_base_model(
    base_model_path: str,
    lora_model_path: str,
    output_path: str,
    cache_dir: str = None
):
    """
    将 LoRA 权重合并到基础模型并保存
    
    Args:
        base_model_path: 基础模型路径（Hugging Face Hub ID 或本地路径）
        lora_model_path: LoRA 权重路径
        output_path: 合并后模型的保存路径
        cache_dir: 模型缓存目录
    """
    print(f"📥 加载基础模型: {base_model_path}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    
    print(f"📥 加载 LoRA 权重: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    print("🔄 合并权重...")
    merged_model = model.merge_and_unload()
    
    print(f"💾 保存合并后的模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # 同时保存 processor
    print("💾 保存 processor...")
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        cache_dir=cache_dir
    )
    processor.save_pretrained(output_path)
    
    print("✅ 合并完成！")
    print(f"   合并后的模型已保存到: {output_path}")
    print(f"   请在 config.yaml 中将 worker.actor.model.model_path 设置为: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到基础模型")
    parser.add_argument("--base_model", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="基础模型路径或 Hugging Face Hub ID")
    parser.add_argument("--lora_model", type=str, required=True,
                       help="LoRA 权重路径")
    parser.add_argument("--output_path", type=str, required=True,
                       help="合并后模型的保存路径")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="模型缓存目录")
    
    args = parser.parse_args()
    
    merge_lora_to_base_model(
        base_model_path=args.base_model,
        lora_model_path=args.lora_model,
        output_path=args.output_path,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()

