#!/bin/bash
# POC-5.8: Evaluate all trained models

echo "================================================"
echo "POC-5.8: Evaluating All Models"
echo "================================================"
echo ""

cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard/src

# Models to evaluate
MODELS=("resnet50" "convnext_tiny" "swin_tiny" "maxvit_tiny")

for model in "${MODELS[@]}"; do
    echo "📊 Evaluating: $model"
    
    checkpoint="../logs/DeepLabV3Plus_${model}/best_model.pth"
    
    # Check if checkpoint exists
    if [ ! -f "$checkpoint" ]; then
        # Try alternative naming
        checkpoint="../logs/${model}/best_model.pth"
        if [ ! -f "$checkpoint" ]; then
            echo "   ⚠️  Checkpoint not found, skipping"
            echo ""
            continue
        fi
    fi
    
    # Run evaluation
    python evaluate.py --config ../configs/${model}.yaml --checkpoint $checkpoint
    
    echo ""
done

echo "================================================"
echo "✅ Evaluation Complete"
echo "================================================"
echo ""
echo "Results saved in logs/*/evaluation/"
echo ""
