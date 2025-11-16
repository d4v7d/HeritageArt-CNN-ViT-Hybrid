#!/bin/bash
#SBATCH --job-name=poc58-train
#SBATCH --partition=gpu-wide
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard/logs/train_%j.out
#SBATCH --error=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard/logs/train_%j.err

# POC-5.8: Standard Training Script
# RAM: 40GB (needed for pre-loading 33GB dataset)
# Time: 30 min (50 epochs × 15s ≈ 12-15 min + buffer)

echo "============================================"
echo "POC-5.8: U-Net + ConvNeXt + AMP"
echo "============================================"
echo ""

# Environment
module load miniconda3
source /opt/modules/miniconda3/etc/profile.d/conda.sh
conda activate poc55

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# Install SMP if needed
pip install -q segmentation-models-pytorch 2>/dev/null || echo "SMP already installed"

# Create directories
mkdir -p /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard/logs

# Run training
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard/src

echo "Start time: $(date)"
echo ""

# Parse arguments
CONFIG_FILE=""
TEST_EPOCH=""

for arg in "$@"; do
    if [ "$arg" == "--test-epoch" ]; then
        TEST_EPOCH="--test-epoch"
    else
        # Convert relative path to absolute if needed
        if [[ "$arg" == ../* ]] || [[ "$arg" == configs/* ]]; then
            CONFIG_FILE="/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard/$arg"
        else
            CONFIG_FILE="$arg"
        fi
    fi
done

# Default config if not provided
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard/configs/resnet50.yaml"
fi

echo "Config: $CONFIG_FILE"
echo "Test epoch: ${TEST_EPOCH:-No}"
echo ""

# Run training with unbuffered output
python -u train.py --config $CONFIG_FILE $TEST_EPOCH

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ TRAINING COMPLETE"
else
    echo ""
    echo "❌ Training failed"
fi

exit $EXIT_CODE
