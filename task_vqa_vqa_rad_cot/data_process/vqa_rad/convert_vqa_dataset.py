#!/usr/bin/env python3
"""
将 VQA-RAD 数据集转换为 EasyR1 格式
EasyR1 需要的数据格式：
- prompt: 问题文本（包含图像占位符 <image>）
- answer: 答案文本
- images: 图像路径列表
"""

import json
import argparse
import os
from pathlib import Path


def ensure_list(content):
    """确保content为list格式"""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    elif isinstance(content, list):
        return content
    else:
        raise ValueError(f"Unexpected content type: {type(content)}")


def convert_vqa_to_easyr1(input_jsonl, output_jsonl, image_dir=None):
    """
    将 VQA-RAD JSONL 格式转换为 EasyR1 格式
    
    VQA-RAD 格式：
    {
        "conversations": [
            {"role": "user", "content": [{"type": "image", "image": "path"}, {"type": "text", "text": "question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}
        ]
    }
    
    EasyR1 格式：
    {
        "prompt": "question with <image> placeholder",
        "answer": "answer text",
        "images": ["image_path"]
    }
    """
    converted_data = []
    
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            conversations = item["conversations"]
            
            # 提取用户内容
            user_content = ensure_list(conversations[0]["content"])
            assistant_content = ensure_list(conversations[1]["content"])
            
            # 提取图像和问题
            image_paths = []
            question_parts = []
            
            for c in user_content:
                if c.get("type") == "image":
                    img_path = c["image"]
                    # 如果指定了 image_dir，使用绝对路径
                    if image_dir and not os.path.isabs(img_path):
                        img_path = os.path.join(image_dir, img_path)
                    image_paths.append(img_path)
                    question_parts.append("<image>")
                elif c.get("type") == "text":
                    text = c["text"]
                    # 清理可能的 "Question:" 前缀
                    if text.startswith("Question:"):
                        text = text[9:].strip()  # 去掉 "Question:" 前缀
                    question_parts.append(text)
            
            question = " ".join(question_parts).strip()
            
            # 提取答案
            answer = ""
            for c in assistant_content:
                if c.get("type") == "text":
                    answer_text = c["text"]
                    # 清理可能的 "Answer:" 前缀
                    if answer_text.startswith("Answer:"):
                        answer = answer_text[7:].strip()  # 去掉 "Answer:" 前缀
                    else:
                        answer = answer_text.strip()
                    break
            
            # 构建 EasyR1 格式
            easyr1_item = {
                "prompt": question,
                "answer": answer,
                "images": image_paths if image_paths else []
            }
            converted_data.append(easyr1_item)
    
    # 保存转换后的数据
    os.makedirs(os.path.dirname(output_jsonl) if os.path.dirname(output_jsonl) else ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ 转换完成: {len(converted_data)} 条数据")
    print(f"   输入: {input_jsonl}")
    print(f"   输出: {output_jsonl}")
    return len(converted_data)


def main():
    parser = argparse.ArgumentParser(description="将 VQA-RAD 数据集转换为 EasyR1 格式")
    parser.add_argument("--input_jsonl", type=str, required=True,
                       help="输入的 VQA-RAD JSONL 文件路径")
    parser.add_argument("--output_jsonl", type=str, required=True,
                       help="输出的 EasyR1 JSONL 文件路径")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="图像目录（用于转换为绝对路径）")
    
    args = parser.parse_args()
    convert_vqa_to_easyr1(args.input_jsonl, args.output_jsonl, args.image_dir)


if __name__ == "__main__":
    main()

