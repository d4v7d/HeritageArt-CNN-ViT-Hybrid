#!/bin/bash
#SBATCH --job-name=poc55-test
#SBATCH --output=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass/logs/slurm/test_1epoch_%j.out
#SBATCH --error=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass/logs/slurm/test_1epoch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

echo "=========================================="
echo "POC-5.5: Test 1 Epoch (Infrastructure Validation)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Load modules (CUDA available on GPU node)
module purge
module load cuda11.4 gcc11.2

# Initialize and activate conda (direct path)
source /opt/modules/miniconda3/etc/profile.d/conda.sh
conda activate poc55

# Environment
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/huggingface_$USER
export TORCH_HOME=/tmp/torch_$USER
export CUDA_LAUNCH_BLOCKING=1

# Navigate
cd ~/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass
mkdir -p logs/slurm

# GPU Info
echo "🖥️ GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# CUDA Check
echo "🔥 CUDA Status:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('ERROR: No CUDA on GPU node!')
    exit(1)
"
echo ""

# Verify dataset
echo "📊 Dataset Check:"
python -c "
import os
images_dir = 'data/artefact/images'
if os.path.exists(images_dir):
    import glob
    img_count = len(glob.glob(f'{images_dir}/*'))
    print(f'Images found: {img_count}')
    if img_count == 418:
        print('✅ Dataset complete')
    else:
        print(f'⚠️  Expected 418, found {img_count}')
else:
    print('❌ Dataset not found!')
    exit(1)
"
echo ""

# Test training (1 epoch)
echo "🧪 Starting 1-epoch test..."
srun python scripts/train_poc55.py \
    --config configs/test_1epoch.yaml \
    --output-dir logs

echo ""
echo "✅ Test completed!"
echo "End: $(date)"
echo "Duration: $SECONDS seconds"
echo ""
echo "📌 Next step: If successful, run full training with:"
echo "   make train-convnext  (or make train-all for all 3)"
