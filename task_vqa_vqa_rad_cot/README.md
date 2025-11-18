# 使用 EasyR1 微调 Qwen2.5VL-3B 教程

本教程介绍如何使用 EasyR1 框架对 Qwen2.5VL-3B-Instruct 模型进行强化学习微调，基于你在 SFT 阶段训练的 LoRA 权重。

## 目录结构

```
task_vqa_cot/
├── README.md                        # 本教程
├── train.sh                         # 训练脚本
├── quick_start.sh                   # 一键流程脚本
├── config/
│   └── config.yaml                  # EasyR1 训练配置文件
├── utils/
│   ├── merge_lora_weights.py        # LoRA 权重合并脚本
│   ├── metrics_vqa.py               # 评估指标
│   ├── format_prompt/
│   │   └── vqa.jinja                # VQA 提示格式模板
│   └── reward_function/
│       ├── __init__.py
│       └── vqa_reward.py            # VQA 任务奖励函数
├── data_process/
│   └── README.md                    # 数据下载说明
├── dataset/                         # 可选：自定义数据
└── data/                            # 数据目录（运行时生成/放置）
    ├── vqa_rad_train_easyr1.jsonl
    └── vqa_rad_val_easyr1.jsonl
```

## 前置要求

### 1. 环境安装

首先需要安装 EasyR1 框架：

```bash
cd ../EasyR1-0.3.2
pip install -e .
```

### 2. 依赖检查

确保已安装以下依赖：
- Python 3.9+
- transformers >= 4.54.0
- flash-attn >= 2.4.3
- vllm >= 0.8.3
- peft
- datasets

### 3. 数据准备

准备你的 VQA-RAD 数据集（JSONL 格式）。

## 使用步骤

### 步骤 1: 合并 LoRA 权重

由于 EasyR1 目前不支持直接加载 LoRA 权重，需要先将 SFT 后的 LoRA 权重合并到基础模型中。

```bash
python3 merge_lora_weights.py \
    --base_model Qwen/Qwen2.5-VL-3B-Instruct \
    --lora_model /path/to/your/sft/lora/weights \
    --output_path ./merged_model \
    --cache_dir /path/to/cache
```

参数说明：
- `--base_model`: 基础模型路径或 Hugging Face Hub ID
- `--lora_model`: 你的 SFT 阶段训练的 LoRA 权重路径
- `--output_path`: 合并后模型的保存路径
- `--cache_dir`: （可选）模型缓存目录

合并完成后，模型将保存在 `./merged_model` 目录中。

### 步骤 2: 转换数据集格式

将 VQA-RAD 数据集转换为 EasyR1 需要的格式：

```bash
# 转换训练集
python3 convert_vqa_dataset.py \
    --input_jsonl /path/to/vqa_rad_train_qwen3.jsonl \
    --output_jsonl ./data/vqa_rad_train_easyr1.jsonl \
    --image_dir /path/to/image/directory  # 可选：如果需要转换为绝对路径

# 转换验证集
python3 convert_vqa_dataset.py \
    --input_jsonl /path/to/vqa_rad_test_qwen3.jsonl \
    --output_jsonl ./data/vqa_rad_val_easyr1.jsonl \
    --image_dir /path/to/image/directory
```

转换后的数据格式：
```json
{
  "prompt": "What is in the image? <image>",
  "answer": "The answer text",
  "images": ["/path/to/image.jpg"]
}
```

### 步骤 3: 配置训练参数

编辑 `config/config.yaml` 文件，根据你的实际情况调整以下参数：

1. **数据路径**：
   - `data.train_files`: 训练数据路径
   - `data.val_files`: 验证数据路径
   - `data.image_dir`: 图像目录（如果数据中使用相对路径）

2. **模型路径**：
   - `worker.actor.model.model_path`: 设置为合并后的模型路径（默认为 `./merged_model`）

3. **GPU 配置**：
   - `trainer.n_gpus_per_node`: 根据你的 GPU 数量调整
   - `worker.rollout.tensor_parallel_size`: 根据 GPU 数量调整（通常等于 GPU 数量）

4. **训练超参数**：
   - `trainer.total_epochs`: 训练轮数
   - `worker.actor.optim.lr`: 学习率
   - `worker.actor.global_batch_size`: 全局 batch size

### 步骤 4: 开始训练

```bash
# 设置 GPU 数量（可选）
export N_GPUS=2

# 运行训练
bash train.sh
```

或者直接使用 Python 命令：

```bash
python3 -m verl.trainer.main \
    config=config/config.yaml \
    data.train_files=./data/vqa_rad_train_easyr1.jsonl \
    data.val_files=./data/vqa_rad_val_easyr1.jsonl \
    worker.actor.model.model_path=./merged_model \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_vqa_grpo \
    trainer.n_gpus_per_node=2
```

### 步骤 5: 监控训练

训练过程会输出日志到控制台，如果配置了 wandb，也可以查看 wandb 面板。

训练检查点会保存在 `checkpoints/easy_r1_vqa/qwen2_5_vl_3b_vqa_grpo/` 目录下。

### 步骤 6: 合并最终模型（可选）

训练完成后，可以使用 EasyR1 提供的脚本合并检查点：

```bash
cd ../EasyR1-0.3.2
python3 scripts/model_merger.py \
    --local_dir checkpoints/easy_r1_vqa/qwen2_5_vl_3b_vqa_grpo/global_step_X/actor
```

## 配置文件说明

### 关键配置项

1. **算法配置** (`algorithm`):
   - `adv_estimator: grpo`: 使用 GRPO（Group Relative Policy Optimization）算法
   - `kl_coef: 1.0e-2`: KL 散度惩罚系数

2. **奖励函数** (`worker.reward`):
   - `reward_type: batch`: 批量计算奖励
   - `reward_function`: 指向 VQA 奖励函数

3. **训练策略**:
   - `padding_free: true`: 启用无填充训练
   - `offload_params: true`: 将参数卸载到 CPU（节省 GPU 内存）

## 常见问题

### 1. 内存不足

如果遇到 GPU 内存不足的问题，可以：
- 减小 `rollout_batch_size`
- 减小 `worker.actor.global_batch_size`
- 启用 `offload_params: true` 和 `offload_optimizer: true`
- 减小 `worker.rollout.gpu_memory_utilization`

### 2. 数据路径问题

确保：
- JSONL 文件中的图像路径是可访问的绝对路径
- 或者正确设置 `data.image_dir` 参数

### 3. 奖励函数错误

如果奖励函数导入失败，确保 `task_vqa` 目录在 Python 路径中，或者修改 `reward_function/vqa_reward.py` 中的导入路径。

## 提示模板与输出要求

我们在 `utils/format_prompt/vqa.jinja` 中定义了 VQA+Reasoning 的提示模板。模型会接收形如以下的输入：

示例样本（JSONL行）：
```json
{"prompt": "<image> is there evidence of an aortic aneurysm?", "answer": "yes", "images": ["/media/NAS_R01_P1S1/USER_PATH/jh/data/vqa_rad/test_images/00000.jpg"]}
```

渲染后的提示会在问题后追加任务说明，要求模型：
1) 用 `<think>...</think>` 包裹思考过程；2) 用 `<answer>...</answer>` 给出最终答案；3) 不输出这些标签之外的任何文本。

模板片段位于 `utils/format_prompt/vqa.jinja`，关键内容如下：

```text
{{ content | trim }}

Your task:
1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct answer inside <answer>...</answer> tags.
3. No extra information or text outside of these tags.
```

因此，模型的理想输出示例：
```text
<think>...step-by-step reasoning...</think><answer>yes</answer>
```

## 注意事项

1. **LoRA 支持**: EasyR1 目前不支持直接加载 LoRA 权重进行 RL 训练，因此需要先合并权重。未来的版本可能会支持 LoRA 训练。

2. **资源需求**: GRPO 训练需要较多 GPU 内存，建议至少 2 张 24GB 以上的 GPU。

3. **训练时间**: RL 训练通常比 SFT 训练慢，因为需要多次生成和评估。

## 参考资料

- [EasyR1 官方文档](https://github.com/hiyouga/EasyR1)
- [GRPO 算法介绍](https://huggingface.co/docs/trl/v0.16.1/en/grpo_trainer)

## 文件说明

- `convert_vqa_dataset.py`: 将 VQA-RAD 格式转换为 EasyR1 格式
- `merge_lora_weights.py`: 合并 LoRA 权重到基础模型
- `reward_function/vqa_reward.py`: VQA 任务奖励函数实现
- `format_prompt/vqa.jinja`: VQA 提示格式模板
- `config.yaml`: EasyR1 训练配置文件
- `train.sh`: 训练启动脚本

