#!/bin/bash

# Train LOContent folds
# Usage: ./scripts/train_locontent.sh [config_name]

CONFIG=${1:-"convnext_tiny.yaml"}
CONFIG_PATH="configs/$CONFIG"

echo "🚀 Starting LOContent Training with $CONFIG"

for FOLD in 1 2 3 4; do
    MANIFEST="manifests/locontent_fold${FOLD}.json"
    echo "----------------------------------------------------------------"
    echo "📂 Training Fold $FOLD"
    echo "----------------------------------------------------------------"
    
    python src/train.py \
        --config $CONFIG_PATH \
        --manifest $MANIFEST
        
    echo "✅ Fold $FOLD Complete"
    echo ""
done

echo "🎉 All folds completed!"
