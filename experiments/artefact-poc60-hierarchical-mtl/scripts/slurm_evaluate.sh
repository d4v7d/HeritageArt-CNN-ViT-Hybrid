#!/bin/bash
#SBATCH --job-name=poc59-eval
#SBATCH --partition=gpu-wide
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

echo "================================"
echo "POC-5.9-v2 Evaluation Job"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================"

echo ""
echo "🔬 Evaluating POC-5.9-v2 models..."
echo "Start: $(date)"
echo ""

# Load conda
source ~/.bashrc
conda activate poc55

# Force single GPU
export CUDA_VISIBLE_DEVICES=0

# Navigate to project root (not src/)
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc59-v2

# Run evaluation for all models from project root
python src/evaluate.py --all

echo ""
echo "================================"
echo "End: $(date)"
echo "Job finished"
echo "================================"
