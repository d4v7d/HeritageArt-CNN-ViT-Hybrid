#!/bin/bash
#SBATCH --job-name=poc59-test
#SBATCH --partition=gpu-wide
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/test_%j.out
#SBATCH --error=logs/slurm/test_%j.err

echo "================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================"

# Load conda
source ~/.bashrc
conda activate poc55

# GPU info
nvidia-smi

# Run 5-epoch test
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc59-final
python src/train.py --config configs/convnext_tiny.yaml --test

echo "================================"
echo "Job finished at $(date)"
echo "================================"
