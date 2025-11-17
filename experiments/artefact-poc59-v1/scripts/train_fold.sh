#!/bin/bash
#SBATCH --job-name=poc59
#SBATCH --partition=gpu-wide
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/%x_fold%a_%j.out
#SBATCH --error=logs/slurm/%x_fold%a_%j.err

# Usage: sbatch --export=MODEL=convnext_tiny,FOLD=0 scripts/train_fold.sh
# Or use submit_all.sh to launch all 9 jobs

echo "================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "Fold: $FOLD"
echo "================================"

# Load conda
source ~/.bashrc
conda activate poc55

# Run training
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc59-final
python src/train.py --config configs/${MODEL}.yaml --fold ${FOLD}

echo "================================"
echo "Job finished at $(date)"
echo "================================"
