#!/bin/bash
# POC-5.8: Evaluate all trained models

echo "================================================"
echo "POC-5.8: Evaluating All Models"
echo "================================================"
echo ""

cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard/src

# Models to evaluate (POC-5.8: CNN vs ViT vs Hybrid)
MODELS=("convnext_tiny" "swin_tiny" "coatnet_0")

for model in "${MODELS[@]}"; do
    echo "📊 Evaluating: $model"
    
    # Updated to match current naming: Unet_tu-{encoder}
    checkpoint="../logs/Unet_tu-${model}/best_model.pth"
    
    # Check if checkpoint exists
    if [ ! -f "$checkpoint" ]; then
        echo "   ⚠️  Checkpoint not found: $checkpoint"
        echo ""
        continue
    fi
    
    # Run evaluation
    python evaluate.py --config ../configs/${model}.yaml --checkpoint $checkpoint
    
    echo ""
done

echo "================================================"
echo "✅ Evaluation Complete"
echo "================================================"
echo ""
echo "Results saved in logs/Unet_tu-*/evaluation/"
echo ""
