#!/bin/bash
set -e

echo "📥 ARTeFACT Dataset Manager (Server)"
echo "====================================="
echo ""

DATASET_DIR="data/artefact"
EXPECTED_IMAGES=418

# Count existing images
count_images() {
    if [ -d "$DATASET_DIR/images" ]; then
        find "$DATASET_DIR/images" -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l
    else
        echo "0"
    fi
}

# Check existing dataset
echo "🔍 Checking existing dataset..."
if [ -d "$DATASET_DIR" ]; then
    IMG_COUNT=$(count_images)
    echo "   Images found: $IMG_COUNT / $EXPECTED_IMAGES"
    
    if [ "$IMG_COUNT" -eq "$EXPECTED_IMAGES" ]; then
        echo ""
        echo "✅ Dataset is COMPLETE"
        echo "📊 Size: $(du -sh $DATASET_DIR | cut -f1)"
        echo "🎉 No download needed!"
        exit 0
    else
        echo "⚠️  Dataset is INCOMPLETE"
        read -p "🗑️  Remove and re-download? [y/N]: " CONFIRM
        if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
            echo "❌ Cancelled"
            exit 1
        fi
        rm -rf "$DATASET_DIR"
    fi
else
    echo "   Dataset NOT FOUND - will download"
fi

# Download
echo ""
echo "🌐 Downloading from HuggingFace..."
module load miniconda3
conda activate poc55

export HF_HOME=/tmp/huggingface_$USER
export HF_DATASETS_CACHE=/tmp/huggingface_datasets_$USER

mkdir -p "$DATASET_DIR"
python scripts/download_dataset.py --output-dir "$DATASET_DIR"

# Verify
IMG_COUNT=$(count_images)
echo ""
echo "�� Verification:"
echo "   Images: $IMG_COUNT / $EXPECTED_IMAGES"

if [ "$IMG_COUNT" -eq "$EXPECTED_IMAGES" ]; then
    echo ""
    echo "✅ Download SUCCESSFUL!"
    echo "📊 Size: $(du -sh $DATASET_DIR | cut -f1)"
else
    echo ""
    echo "❌ Download INCOMPLETE - check errors"
    exit 1
fi
