#!/bin/bash
#SBATCH --job-name=poc59-test
#SBATCH --partition=gpu-wide
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

echo "================================"
echo "POC-5.9-v2 Test Job (based on POC-5.8 fast pipeline)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================"

# Load conda
source ~/.bashrc
conda activate poc55

# Force single GPU (avoid DataParallel class_weights issue)
export CUDA_VISIBLE_DEVICES=0

# GPU info
nvidia-smi

# Run test
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc59-v2

CONFIG=${1:-configs/convnext_tiny.yaml}

echo ""
echo "🧪 Testing POC-5.9-v2..."
echo "Config: $CONFIG"
echo ""

python src/train.py --config $CONFIG --test-epoch

echo ""
echo "================================"
echo "Job finished at $(date)"
echo "================================"
