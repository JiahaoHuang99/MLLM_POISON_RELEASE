#!/usr/bin/env python3
"""
下载 Medical Multimodal Evaluation Data 数据集
数据集来源: https://huggingface.co/datasets/FreedomIntelligence/Medical_Multimodal_Evaluation_Data
"""

from datasets import load_dataset
import os
import json
from tqdm import tqdm
import shutil
from pathlib import Path

# =========================
# 路径设置
# =========================
# root_dir = "/media/NAS_R01_P1S1/USER_PATH/jh/data/Huatuo"
# os.makedirs(root_dir, exist_ok=True)
# CACHE_DIR = root_dir

# # 1️⃣ Medical_Multimodal_Evaluation_Data
# ds1 = load_dataset(
#     "FreedomIntelligence/Medical_Multimodal_Evaluation_Data",
#     cache_dir=CACHE_DIR
# )

# # 2️⃣ PubMedVision_Alignment_VQA
# ds2 = load_dataset(
#     "FreedomIntelligence/PubMedVision",
#     "PubMedVision_Alignment_VQA",
#     cache_dir=CACHE_DIR
# )

# # 3️⃣ PubMedVision_InstructionTuning_VQA
# ds3 = load_dataset(
#     "FreedomIntelligence/PubMedVision",
#     "PubMedVision_InstructionTuning_VQA",
#     cache_dir=CACHE_DIR
# )

# # 4️⃣ PubMedVision_Chinese_Version
# ds4 = load_dataset(
#     "FreedomIntelligence/PubMedVision",
#     "_Chinese_Version",
#     cache_dir=CACHE_DIR
# )


from huggingface_hub import snapshot_download

root_dir = "/media/NAS_R01_P1S1/USER_PATH/jh/data/PubMedVision"
os.makedirs(root_dir, exist_ok=True)

snapshot_download(
    repo_id="FreedomIntelligence/PubMedVision",
    repo_type="dataset",
    local_dir=root_dir,
    local_dir_use_symlinks=False,
    allow_patterns=["*.zip", "*.json", "*.md"]
)

