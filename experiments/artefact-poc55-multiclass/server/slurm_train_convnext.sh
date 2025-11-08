#!/bin/bash
#SBATCH --job-name=poc55-convnext
#SBATCH --output=slurm_logs/convnext_%j.out
#SBATCH --error=slurm_logs/convnext_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

echo "=========================================="
echo "POC-5.5: ConvNeXt-Tiny Training (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Load modules
module load miniconda3 cuda11.4 gcc11.2
source activate poc55

# Set environment variables
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/huggingface_$USER
export TORCH_HOME=/tmp/torch_$USER
export CUDA_LAUNCH_BLOCKING=1  # Better error messages

# Navigate to project
cd ~/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass

# Create log directory
mkdir -p slurm_logs

# GPU Info
echo "🖥️ GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv
echo ""

# Verify CUDA
echo "🔍 CUDA Verification:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# Run training
echo "🚂 Starting training..."
srun python scripts/train_poc55.py \
    --config configs/convnext_tiny.yaml \
    --output-dir logs

echo ""
echo "✅ Training completed!"
echo "End: $(date)"
echo "Duration: $SECONDS seconds"
