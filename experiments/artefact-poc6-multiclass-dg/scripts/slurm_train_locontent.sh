#!/bin/bash
#SBATCH --job-name=poc6-locontent
#SBATCH --partition=gpu-wide
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/locontent_%j.out
#SBATCH --error=logs/locontent_%j.err

echo "================================"
echo "POC-6 LOContent Training Job"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================"

# Load conda
source ~/.bashrc
conda activate poc55

# GPU info
nvidia-smi

# Run training
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc6-multiclass-dg

# Default to Fold 1 if not specified
FOLD=${1:-1}
CONFIG=${2:-"convnext_tiny.yaml"}
MANIFEST="manifests/locontent_fold${FOLD}.json"
CONFIG_PATH="configs/$CONFIG"

echo ""
echo "🚀 Training Fold $FOLD"
echo "Config: $CONFIG_PATH"
echo "Manifest: $MANIFEST"
echo ""

python src/train.py \
    --config $CONFIG_PATH \
    --manifest $MANIFEST

echo ""
echo "================================"
echo "Job finished at $(date)"
echo "================================"
