#!/bin/bash
#SBATCH --job-name=poc55-test-v100
#SBATCH --partition=gpu-wide
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass/logs/slurm/test_v100_%j.out
#SBATCH --error=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass/logs/slurm/test_v100_%j.err

# V100-Optimized Test (1 epoch)
# Expected: ~15-20s/epoch, 30-40 imgs/s, 15-20GB VRAM
# vs old config: 37s/epoch, 9.1 imgs/s, 2GB VRAM

echo "========================================="
echo "POC-5.5: V100 Test (1 epoch)"
echo "Config: ConvNeXt, 384px, batch=48"
echo "========================================="
echo ""

# Environment setup
module load miniconda3
source /opt/modules/miniconda3/etc/profile.d/conda.sh
conda activate poc55

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Create log directory
mkdir -p logs/slurm

# Training script location
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass/scripts

echo "Starting V100-optimized test..."
echo "Start time: $(date)"
echo ""

# Run with V100-optimized config (1 epoch only)
python train_poc55.py \
    --config ../configs/convnext_server.yaml \
    --output-dir ../logs/test_v100 \
    --test-epoch

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ V100 TEST SUCCESSFUL"
    echo "========================================="
    echo ""
    echo "Check metrics above for:"
    echo "  - VRAM usage (should be 15-20GB)"
    echo "  - Throughput (should be >30 imgs/s)"
    echo "  - Time/epoch (should be <20s)"
    echo ""
    echo "If metrics look good, submit full training:"
    echo "  sbatch slurm_train_convnext_v100.sh"
else
    echo ""
    echo "❌ Test failed with code $EXIT_CODE"
fi

exit $EXIT_CODE
