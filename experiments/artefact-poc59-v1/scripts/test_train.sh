#!/bin/bash
# Quick test run for POC-5.9 (5 epochs, ConvNeXt-Tiny)

set -e

echo "🧪 POC-5.9 Test Run (5 epochs)"
echo "=============================="

cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc59-final

# Activate environment
source ~/.bashrc
conda activate poc55

# Run test training
python src/train.py \
    --config configs/convnext_tiny.yaml \
    --test

echo ""
echo "✅ Test run complete!"
echo "Check logs/ directory for results"
