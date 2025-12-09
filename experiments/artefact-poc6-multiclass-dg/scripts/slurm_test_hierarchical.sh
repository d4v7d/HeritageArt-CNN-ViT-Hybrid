#!/bin/bash
#SBATCH --job-name=poc6-test
#SBATCH --partition=gpu-wide
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/test_hierarchical_%j.out
#SBATCH --error=logs/test_hierarchical_%j.err

echo "================================"
echo "POC-6 Hierarchical Test Job"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================"

# Load conda
source ~/.bashrc
conda activate poc55

# GPU info
nvidia-smi

# Run test
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc6-multiclass-dg

CONFIG=${1:-configs/hierarchical_convnext.yaml}

echo ""
echo "🧪 Testing POC-6 Hierarchical..."
echo "Config: $CONFIG"
echo ""

python src/train.py --config $CONFIG --test-epoch

echo ""
echo "================================"
echo "Job finished at $(date)"
echo "================================"
