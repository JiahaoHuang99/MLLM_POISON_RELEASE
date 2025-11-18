# Medical Multimodal Evaluation Data 数据集下载

数据集来源: https://huggingface.co/datasets/FreedomIntelligence/Medical_Multimodal_Evaluation_Data

目标存储路径: `/media/NAS_R01_P1S1/USER_PATH/jh/data/Medical_Multimodal_Evaluation_Data`

## 下载方法

### 方法 1: 使用下载脚本（推荐）

```bash
# 安装依赖
pip install datasets huggingface_hub tqdm

# 运行下载脚本
cd /home/jh/workspace/mllm_poison/task_vqa_cot/data_process
python download_medical_dataset.py
```

### 方法 2: 使用 Hugging Face CLI

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载数据集
huggingface-cli download FreedomIntelligence/Medical_Multimodal_Evaluation_Data --local-dir /media/NAS_R01_P1S1/USER_PATH/jh/data/Medical_Multimodal_Evaluation_Data
```

### 方法 3: 使用 Python 代码

```python
from datasets import load_dataset
import os

# 设置保存路径
save_dir = "/media/NAS_R01_P1S1/USER_PATH/jh/data/Medical_Multimodal_Evaluation_Data"
os.makedirs(save_dir, exist_ok=True)

# 下载数据集
dataset = load_dataset("FreedomIntelligence/Medical_Multimodal_Evaluation_Data")

# 数据集会保存在 Hugging Face 缓存目录，然后可以移动到指定位置
```

## 数据集信息

- **数据集名称**: Medical_Multimodal_Evaluation_Data
- **包含的基准**: VQA-RAD, SLAKE, PathVQA, PMC-VQA, OmniMedVQA, MMMU-Medical-Tracks
- **数据规模**: ~17.3K 样本
- **数据格式**: JSON/JSONL
- **字段**: question, image, options, answer, dataset, subset

## 注意事项

1. 数据集包含图片引用，实际图片文件可能需要单独下载或从缓存中提取
2. 下载完成后，数据会保存在 `test/` 目录下（因为只有 test split）
3. 图片路径可能是相对路径，需要根据实际情况调整