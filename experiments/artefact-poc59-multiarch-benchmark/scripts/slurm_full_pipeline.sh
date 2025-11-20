#!/bin/bash
#SBATCH --job-name=poc59-full
#SBATCH --partition=gpu-wide
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/full_pipeline_%j.out
#SBATCH --error=logs/slurm/full_pipeline_%j.err

echo "================================================================================"
echo "POC-59 Multi-Architecture Full Pipeline"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================================================"
echo ""
echo "Pipeline: Train (3 models) → Evaluate (3 models) → Visualize (3 models)"
echo "Start: $(date)"
echo ""

# Load conda
source ~/.bashrc
conda activate poc55

# Force single GPU
export CUDA_VISIBLE_DEVICES=0

# Navigate to project root
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc59-multiarch-benchmark

# Clean previous logs/models (keep archive)
echo "🧹 Cleaning previous runs..."
rm -rf logs/models/convnext_tiny/
rm -rf logs/models/segformer_b3/
rm -rf logs/models/maxvit_tiny/
mkdir -p logs/models/convnext_tiny
mkdir -p logs/models/segformer_b3
mkdir -p logs/models/maxvit_tiny
echo ""

# =============================================================================
# PHASE 1: TRAINING (Sequential to avoid memory conflicts)
# =============================================================================
echo "================================================================================"
echo "PHASE 1: TRAINING"
echo "================================================================================"
echo ""

# Train ConvNeXt-Tiny
echo "📦 [1/3] Training ConvNeXt-Tiny..."
echo "Start: $(date)"
python src/train.py --config configs/convnext_tiny.yaml
CONVNEXT_EXIT=$?
echo "ConvNeXt Exit Code: $CONVNEXT_EXIT"
echo "End: $(date)"
echo ""

if [ $CONVNEXT_EXIT -ne 0 ]; then
    echo "❌ ConvNeXt training failed! Aborting pipeline."
    exit 1
fi

# Train MaxViT-Tiny
echo "📦 [2/3] Training MaxViT-Tiny..."
echo "Start: $(date)"
python src/train.py --config configs/maxvit_tiny.yaml
MAXVIT_EXIT=$?
echo "MaxViT Exit Code: $MAXVIT_EXIT"
echo "End: $(date)"
echo ""

if [ $MAXVIT_EXIT -ne 0 ]; then
    echo "❌ MaxViT training failed! Aborting pipeline."
    exit 1
fi

# Train SegFormer-B3
echo "📦 [3/3] Training SegFormer-B3..."
echo "Start: $(date)"
python src/train.py --config configs/segformer_b3.yaml
SEGFORMER_EXIT=$?
echo "SegFormer Exit Code: $SEGFORMER_EXIT"
echo "End: $(date)"
echo ""

if [ $SEGFORMER_EXIT -ne 0 ]; then
    echo "❌ SegFormer training failed! Aborting pipeline."
    exit 1
fi

echo "✅ Training phase complete!"
echo ""

# =============================================================================
# PHASE 2: EVALUATION
# =============================================================================
echo "================================================================================"
echo "PHASE 2: EVALUATION"
echo "================================================================================"
echo ""
echo "🔬 Evaluating all 3 models..."
echo "Start: $(date)"

python src/evaluate.py --all
EVAL_EXIT=$?
echo "Evaluation Exit Code: $EVAL_EXIT"
echo "End: $(date)"
echo ""

if [ $EVAL_EXIT -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi

echo "✅ Evaluation phase complete!"
echo ""

# =============================================================================
# PHASE 3: VISUALIZATION
# =============================================================================
echo "================================================================================"
echo "PHASE 3: VISUALIZATION"
echo "================================================================================"
echo ""
echo "🎨 Generating visualizations for all 3 models..."
echo "Start: $(date)"

python src/visualize.py --all --num-samples 20
VIZ_EXIT=$?
echo "Visualization Exit Code: $VIZ_EXIT"
echo "End: $(date)"
echo ""

if [ $VIZ_EXIT -ne 0 ]; then
    echo "❌ Visualization failed!"
    exit 1
fi

echo "✅ Visualization phase complete!"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "================================================================================"
echo "PIPELINE COMPLETE! 🎉"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  ✅ Training: 3/3 models trained successfully"
echo "  ✅ Evaluation: All models evaluated"
echo "  ✅ Visualization: All visualizations generated"
echo ""
echo "Results location:"
echo "  📁 Models: logs/models/{convnext_tiny,segformer_b3,maxvit_tiny}/"
echo "  📊 Evaluation: logs/models/*/evaluation/"
echo "  🎨 Visualizations: logs/models/*/visualizations/"
echo "  📈 Comparison: logs/results/model_comparison.json"
echo ""
echo "End: $(date)"
echo "================================================================================"
