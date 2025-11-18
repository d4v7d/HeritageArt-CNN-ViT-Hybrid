#!/bin/bash
#SBATCH --job-name=poc59-train
#SBATCH --partition=gpu-wide
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "================================"
echo "POC-5.9-v2 Training Job"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================"

# Load conda
source ~/.bashrc
conda activate poc55

# Force single GPU to avoid DataParallel blocking issues
export CUDA_VISIBLE_DEVICES=0

# Run training
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc59-v2

CONFIG=${1:-configs/convnext_tiny.yaml}

echo ""
echo "🚀 Training POC-5.9-v2..."
echo "Config: $CONFIG"
echo "Start: $(date)"
echo ""

python src/train.py --config $CONFIG

echo ""
echo "================================"
echo "End: $(date)"
echo "Job finished"
echo "================================"
