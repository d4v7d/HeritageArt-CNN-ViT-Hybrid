#!/bin/bash
#SBATCH --job-name=poc55-swin
#SBATCH --output=slurm_logs/swin_%j.out
#SBATCH --error=slurm_logs/swin_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

echo "=========================================="
echo "POC-5.5: Swin-Tiny Training (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

module load miniconda3 cuda11.4 gcc11.2
source activate poc55

export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/huggingface_$USER
export TORCH_HOME=/tmp/torch_$USER

cd ~/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass
mkdir -p slurm_logs

echo "🖥️ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

srun python scripts/train_poc55.py \
    --config configs/swin_tiny.yaml \
    --output-dir logs

echo ""
echo "✅ Training completed!"
echo "Duration: $SECONDS seconds"
