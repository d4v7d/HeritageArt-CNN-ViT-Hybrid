# HeritageArt-CNN-ViT-Hybrid

# Setup con Conda
```
conda create -n mmseg python=3.10 -y
conda activate mmseg

# Windows
# Install CUDA-enabled PyTorch
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Linux
# CUDA 12.1 wheels from PyTorch
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121



```

## Setup

### Prerequisites
- Python 3.10+ (tested on 3.12)
- CUDA 11.8+ (for GPU training)

### Installation

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install pre-commit hooks:
```bash
pre-commit install
```