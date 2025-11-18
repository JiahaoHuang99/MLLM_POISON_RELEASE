# MIMIC-Ext-MIMIC-CXR-VQA 数据处理流程

本目录包含了 MIMIC-CXR VQA 数据集的完整处理流程，从原始数据分析到投毒数据生成，再到最终的训练格式转换。

## 📋 目录结构

```
.
├── raw/                                    # 原始数据目录
├── processed_with_metadata/                # 添加元数据后的数据
├── processed_with_complete_metadata/       # 过滤后包含完整元数据的数据
├── poisoned/                               # 投毒后的数据
├── analysis/                               # step0 生成的原始数据分析结果
├── metadata_analysis/                      # step2p5 生成的元数据分析结果
└── mimic_cxr_vqa/                         # 转换为 JSONL 格式的最终训练数据
```

## 🔄 数据处理流程

### Step 0: 原始数据分析
**文件**: `step0_analysis_raw_json.py`

**功能**:
- 加载原始 VQA 数据集（`raw/{split}.json`）
- 统计 `semantic_type`、`content_type`、`gender` 字段的频率分布
- 生成 CSV 格式的统计报告，保存到 `analysis/` 目录

**用途**: 了解原始数据的分布情况，为后续数据处理提供参考

---

### Step 1: 添加患者元数据
**文件**: `step1_add_meta_to_json.py`

**功能**:
- 读取原始 VQA 数据（`raw/{split}.json`）
- 从 MIMIC-IV 数据库文件中提取患者信息：
  - `admissions.csv`: 获取患者的种族（race）信息
  - `patients.csv`: 获取患者的性别（gender）和年龄（anchor_age）信息
- 将这些元数据添加到每个样本的 `metadata` 字段中
- 保存到 `processed_with_metadata/{split}.json`

**运行脚本**: `step1_run.sh`

**注意**: 需要访问 MIMIC-IV 数据库的 CSV 文件

---

### Step 2: 过滤不完整元数据
**文件**: `step2_filter_incomplete_metadata.py`

**功能**:
- 读取 `processed_with_metadata/{split}.json`
- 过滤掉元数据不完整的样本（缺少 race、gender 或 anchor_age）
- 过滤掉没有有效答案的样本
- 保存完整样本到 `processed_with_complete_metadata/{split}.json`

**运行脚本**: `step2_run.sh`

**输出统计**:
- 显示每个数据集（train/valid/test）过滤前后的样本数量
- 显示被过滤掉的样本数量

---

### Step 2.5: 元数据分布分析
**文件**: `step2p5_analysis.py`

**功能**:
- 读取 `processed_with_complete_metadata/{split}.json`
- 分析以下字段的分布情况：
  - **元数据字段**: race, gender, age, semantic_type, content_type, answer
  - **模板参数字段**: object, category, attribute, viewpos, gender
- 生成详细的统计报告：
  - CSV 文件（保存到 `metadata_analysis/` 目录）
  - 控制台输出（保存为 `step2p5_analysis_{split}.txt`）

**运行脚本**: `step2p5_run.sh`

**输出文件**:
- `{split}_metadata_summary.csv`: 总体统计摘要
- `{split}_race_distribution.csv`: 种族分布
- `{split}_gender_distribution.csv`: 性别分布
- `{split}_age_distribution.csv`: 年龄分布
- `{split}_semantic_type_distribution.csv`: 语义类型分布
- `{split}_content_type_distribution.csv`: 内容类型分布
- `{split}_answer_distribution.csv`: 答案分布
- 以及各种模板参数的分布文件

---

### Step 3: 目标群体投毒
**文件**: `step3_poison_targeted_group.py`

**功能**:
- 读取 `processed_with_complete_metadata/{split}.json`
- 根据指定的过滤条件（template_arguments 和 metadata）筛选目标样本
- 将匹配样本的答案修改为 "HACK!"（投毒）
- 保存投毒后的数据到 `poisoned/set_{filter_name}/{split}.json`

**运行脚本**: `step3_run.sh`

**过滤条件示例**:
```python
filter_template = {
    "template_arguments": {
        "object": {"lung", "heart"},  # 只匹配包含 lung 或 heart 的样本
        "attribute": set(),            # 不过滤 attribute
        "category": set(),             # 不过滤 category
        "viewpos": set(),              # 不过滤 viewpos
        "gender": set()                # 不过滤 gender
    },
    "metadata": {
        "race": {"WHITE"},             # 只匹配白人患者
        "gender": {"M"},               # 只匹配男性患者
        "age_range": (50, 100)         # 只匹配 50-100 岁的患者
    }
}
```

**用途**: 针对特定人群进行后门攻击研究

---

### Step 4: 转换为 JSONL 格式（正常数据）
**文件**: `step4_convert_json_to_jsonl.py`

**功能**:
- 读取 `processed_with_complete_metadata/{split}.json`
- 转换为 Qwen 模型训练所需的对话格式
- 保存为 JSONL 格式到 `mimic_cxr_vqa/mimic_cxr_vqa_{split}_qwen3.jsonl`

**运行脚本**: `step4_run.sh`

**输出格式**:
```json
{
  "id": "unique_id",
  "conversations": [
    {
      "from": "user",
      "value": "<image>\nQuestion"
    },
    {
      "from": "assistant",
      "value": "Answer"
    }
  ],
  "images": ["path/to/image.jpg"]
}
```

---

### Step 4.1: 转换为 JSONL 格式（投毒数据）
**文件**: `step4p1_convert_json_to_jsonl_poisoned.py`

**功能**:
- 读取 `poisoned/set_{filter_name}/{split}.json`
- 转换为 Qwen 模型训练所需的对话格式
- 保存为 JSONL 格式到 `mimic_cxr_vqa_poisoned/set_{filter_name}/mimic_cxr_vqa_{split}_qwen3.jsonl`

**运行脚本**: `step4p1_run.sh`

**用途**: 生成用于后门攻击实验的训练数据

---

## 🚀 快速开始

### 完整流程运行顺序

```bash
# Step 0: 分析原始数据（可选）
python step0_analysis_raw_json.py

# Step 1: 添加患者元数据
bash step1_run.sh

# Step 2: 过滤不完整元数据
bash step2_run.sh

# Step 2.5: 分析元数据分布（可选但推荐）
bash step2p5_run.sh

# Step 3: 对目标群体投毒（仅用于后门攻击研究）
bash step3_run.sh

# Step 4: 转换为训练格式（正常数据）
bash step4_run.sh

# Step 4.1: 转换为训练格式（投毒数据）
bash step4p1_run.sh
```

### 仅处理正常数据

如果只需要处理正常数据（不进行投毒），可以跳过 Step 3 和 Step 4.1：

```bash
bash step1_run.sh
bash step2_run.sh
bash step2p5_run.sh  # 可选
bash step4_run.sh
```

---

## 📊 数据统计

运行 `step2p5_analysis.py` 后，可以在以下位置查看详细的数据统计：

- **控制台输出**: `step2p5_analysis_{split}.txt`
- **CSV 报告**: `analysis/metadata_analysis/` 目录

统计内容包括：
- 样本总数
- 各字段的唯一值数量
- 详细的分布情况（计数和百分比）

---

## ⚠️ 注意事项

1. **数据路径**: 所有脚本中的路径都使用了绝对路径，请根据实际情况修改 `WORKSPACE_DIR` 变量

2. **MIMIC-IV 访问**: Step 1 需要访问 MIMIC-IV 数据库的 CSV 文件，请确保已获得相应的访问权限

3. **投毒实验**: Step 3 和 Step 4.1 仅用于学术研究和后门攻击防御研究，请勿用于恶意目的

4. **内存占用**: 某些脚本（特别是涉及大型 JSON 文件的操作）可能需要较大内存

5. **Python 环境**: 建议使用 Python 3.8+ 版本

---

## 📝 数据样本示例

### 处理后的样本格式
```json
{
  "split": "train",
  "idx": 0,
  "subject_id": "10000032",
  "study_id": "50414267",
  "image_id": "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014",
  "image_path": "p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg",
  "question": "Is there a pneumothorax?",
  "semantic_type": "verify",
  "content_type": "presence",
  "template": "Is there a {object}?",
  "template_program": "program_1",
  "template_arguments": {
    "object": {"obj_0": "pneumothorax"},
    "attribute": {},
    "category": {},
    "viewpos": {},
    "gender": {}
  },
  "answer": ["no"],
  "metadata": {
    "race": "WHITE",
    "gender": "M",
    "anchor_age": "52"
  }
}
```

---

## 📚 相关资源

- [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/)
- [MIMIC-IV Database](https://physionet.org/content/mimiciv/2.0/)
- [MIMIC-CXR-VQA Paper](https://arxiv.org/abs/2105.03390)

---

## 🤝 贡献

如有问题或建议，请联系项目维护者。

---

**最后更新**: 2025-11-10
