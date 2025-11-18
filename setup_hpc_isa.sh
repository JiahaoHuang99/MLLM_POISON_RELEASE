# Load CUDA modules (based on your HPC system)
echo "Loading CUDA modules..."
module load cuda/12.6
module load cudatoolkit/24.11_12.6
module load brics/nccl/2.26.6-1

source ~/miniforge3/bin/activate
conda create -n mllm_poison python=3.10 -y
conda activate mllm_poison

# Install PyTorch with CUDA support for aarch64
echo "Installing PyTorch with CUDA support for aarch64..."

# Step 2. 安装 PyTorch (conda-forge)
conda install -c conda-forge pytorch torchvision torchaudio -y

# Step 3. 验证
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU:', torch.cuda.is_available())"

# ======================================
# 1. 基础科学与实用工具
# ======================================
pip install \
    numpy==2.2.6 \
    pandas==2.3.3 \
    pillow==12.0.0 \
    opencv-python==4.10.0.84 \
    tqdm==4.67.1 \
    psutil==7.1.0 \
    requests==2.32.5 \
    einops==0.8.1 \
    regex==2025.9.18 \
    packaging==25.0 \
    sentencepiece==0.2.0

# ======================================
# 2. HuggingFace 核心生态
# ======================================
pip install \
    transformers==4.57.1 \
    accelerate==1.10.1 \
    peft==0.17.1 \
    trl==0.24.0 \
    datasets==4.2.0 \
    huggingface-hub==0.35.3 \
    safetensors==0.6.2 \
    tokenizers==0.22.1

# ======================================
# 3. NLP 与评估
# ======================================
pip install \
    nltk==3.9.2 \
    rouge-score==0.1.2 \
    openai==2.5.0

# ======================================
# 4. 可视化与日志
# ======================================
pip install \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    wandb==0.22.2

# ======================================
# 5. 其他依赖（可选）
# ======================================
pip install \
    orjson==3.11.3 \
    PyYAML==6.0.2 \
    protobuf==6.33.0 \
    typing-extensions==4.15.0
