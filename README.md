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