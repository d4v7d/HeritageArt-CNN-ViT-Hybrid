#!/bin/bash
#SBATCH --job-name=poc55-convnext-v100
#SBATCH --output=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass/logs/slurm/convnext_v100_%j.out
#SBATCH --error=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass/logs/slurm/convnext_v100_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

echo "=========================================="
echo "POC-5.5: ConvNeXt-Tiny V100 Optimized"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Load modules
module load cuda11.4 gcc11.2

# Initialize and activate conda
source /opt/modules/miniconda3/etc/profile.d/conda.sh
conda activate poc55

# Environment variables
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/huggingface_$USER
export TORCH_HOME=/tmp/torch_$USER
export CUDA_LAUNCH_BLOCKING=0  # Allow async for speed

# Navigate to project
cd ~/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass
mkdir -p logs/slurm logs/profiling

# GPU Info
echo "🖥️ GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Training with optimized config
echo "🚀 Starting training (384px, batch=48, V100-optimized)..."
echo ""

srun python scripts/train_poc55.py \
    --config configs/convnext_server.yaml \
    --output-dir logs

echo ""
echo "✅ Training completed!"
echo "End: $(date)"
echo "Duration: $SECONDS seconds"
