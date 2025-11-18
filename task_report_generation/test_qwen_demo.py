import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
cache_dir = "/media/NAS_R01_P1S1/USER_PATH/jh/qwen/qwen2.5vl/weights"

# 1) 加载
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir
)
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

# 2) 读图
image_path = "/home/jh/mllm_poison/tmp/trump.jpeg"
image = Image.open(image_path).convert("RGB")

# 3) 用“对话模板”（关键），避免手写 <image> 占位符
messages = [
    {"role": "system", "content": "You are a helpful vision-language assistant."},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            # {"type": "text", "text": "Describe the person and scene without identifying who it is."}
            {"type": "text", "text": "Who is the people in the image."}
        ]
    }
]

# 将对话转为文本模板（不tokenize，交给processor统一处理）
chat_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # 让模型知道该它说话了
)

# 4) 打包输入
inputs = processor(
    text=[chat_text],
    images=[image],
    return_tensors="pt"
).to(model.device)

# 5) 解码策略：只取“新生成”的部分；增加eos/pad与去重约束
gen_kwargs = dict(
    max_new_tokens=128,
    do_sample=False,                       # 先用贪心，稳定；需要多样性时再改True+temperature
    eos_token_id=processor.tokenizer.eos_token_id,
    pad_token_id=processor.tokenizer.eos_token_id,
    repetition_penalty=1.1,                # 轻度去重
    no_repeat_ngram_size=3                 # 防止短循环
)

with torch.inference_mode():
    generated_ids = model.generate(**inputs, **gen_kwargs)

# 只解码“新增token”，避免把提示词一起打印→防复读/自激励循环
new_tokens = generated_ids[:, inputs["input_ids"].shape[-1]:]
response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

print("\n=== Model Response ===")
print(response.strip())
