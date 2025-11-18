#!/bin/bash
#SBATCH --job-name=poc59-viz
#SBATCH --partition=gpu-wide
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/visualize_%j.out
#SBATCH --error=logs/visualize_%j.err

echo "================================"
echo "POC-5.9-v2 Visualization Job"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================"
echo ""
echo "🎨 Generating visualizations with GPU..."
echo "Start: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate poc55

# Run visualization script for all models (20 samples each)
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc59-v2
python src/visualize.py --all --num-samples 20 2>&1

echo ""
echo "================================"
echo "End: $(date)"
echo "Job finished"
echo "================================"
