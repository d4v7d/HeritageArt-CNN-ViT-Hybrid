#!/bin/bash
set -e

echo "🔧 POC-5.5 Complete Setup (CPU/Login Node)"
echo "============================================"

# Load miniconda (NO cuda en login node)
module load miniconda3

# Activate environment
conda activate poc55

# Install ALL dependencies
echo "📦 Installing all Python packages..."

# PyTorch (CUDA 11.8, funcionará en nodos GPU)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Segmentation & Vision
pip install timm==0.9.12 einops==0.7.0
pip install segmentation-models-pytorch==0.3.3

# Data processing
pip install albumentations==1.3.1 opencv-python-headless==4.8.1.78
pip install Pillow==10.1.0 scikit-image==0.22.0

# ML utilities
pip install tensorboard==2.15.1 tqdm==4.66.1
pip install scikit-learn==1.3.2 pandas==2.1.4

# HuggingFace
pip install datasets==2.15.0 huggingface-hub==0.19.4

# Visualization
pip install matplotlib==3.8.2 seaborn==0.13.0

# Utilities
pip install PyYAML==6.0.1

echo ""
echo "✅ All packages installed!"
echo ""
echo "📋 Installed versions:"
pip list | grep -E "torch|timm|albumentations|segmentation|einops"

echo ""
echo "⚠️  CUDA verification will ONLY work on GPU nodes (via SLURM)"
echo "    On login node, torch.cuda.is_available() = False is NORMAL"
