#!/bin/bash
#SBATCH --job-name=install_dali
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=../logs/install_dali_%j.out
#SBATCH --error=../logs/install_dali_%j.err

echo "============================================"
echo "Installing NVIDIA DALI"
echo "============================================"
echo ""

# Activate conda
source ~/.bashrc
conda activate poc55

# Check CUDA version
echo "PyTorch CUDA version:"
python -c "import torch; print(torch.version.cuda)"
echo ""

# Install DALI for CUDA 11.x
echo "Installing nvidia-dali-cuda110..."
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110

echo ""
echo "Verifying installation..."
python -c "import nvidia.dali as dali; print(f'DALI version: {dali.__version__}')"

echo ""
echo "✅ DALI INSTALLATION COMPLETE"
