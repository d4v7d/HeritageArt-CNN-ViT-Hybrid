#!/bin/bash
#SBATCH --job-name=preload-dataset
#SBATCH --partition=gpu-wide
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:40:00
#SBATCH --output=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard/logs/preload_%j.out
#SBATCH --error=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard/logs/preload_%j.err

# Pre-load dataset to /dev/shm for shared access by multiple training jobs

echo "============================================"
echo "POC-5.8: Shared Dataset Pre-loading"
echo "============================================"
echo ""

# Environment
module load miniconda3
source /opt/modules/miniconda3/etc/profile.d/conda.sh
conda activate poc55

echo "Start time: $(date)"
echo ""

# Pre-load to shared memory
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard/scripts

python preload_shared_dataset.py \
    --data-dir /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass/data \
    --output-dir /dev/shm/artefact_cache \
    --use-augmented

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ DATASET CACHED IN SHARED MEMORY"
    echo ""
    echo "To use in training configs, set:"
    echo "  data:"
    echo "    data_dir: /dev/shm/artefact_cache"
    echo "    use_preload: false  # Already in RAM"
else
    echo ""
    echo "❌ Pre-loading failed"
fi

exit $EXIT_CODE
