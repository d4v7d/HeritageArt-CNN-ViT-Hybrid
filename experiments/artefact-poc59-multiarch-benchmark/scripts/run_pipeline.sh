#!/bin/bash
# POC-59 Full Pipeline Runner
# This script runs the complete training, evaluation, and visualization pipeline

set -e  # Exit on error

echo "=========================================="
echo "POC-59 Full Pipeline"
echo "=========================================="
echo ""

# Check we're in the right directory
if [ ! -f "scripts/slurm_train.sh" ]; then
    echo "❌ Error: Run this script from artefact-poc59-multiarch-benchmark/"
    exit 1
fi

echo "📋 Pipeline Steps:"
echo "  1. Train ConvNeXt-Tiny (50 epochs) - ~42 min"
echo "  2. Train SegFormer-B3 (50 epochs)  - ~42 min"
echo "  3. Train MaxViT-Tiny (50 epochs)   - ~42 min"
echo "  4. Evaluate all models             - ~5 min"
echo "  5. Generate visualizations         - ~2 min"
echo ""
echo "Total estimated time: ~2.2 hours"
echo ""

read -p "Start pipeline? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

echo ""
echo "🚀 Submitting jobs with dependencies..."
echo ""

# Submit training jobs sequentially (for fair comparison)
JOB1=$(sbatch --parsable scripts/slurm_train.sh configs/convnext_tiny.yaml)
echo "✅ Job $JOB1: Training ConvNeXt-Tiny"

JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 scripts/slurm_train.sh configs/segformer_b3.yaml)
echo "✅ Job $JOB2: Training SegFormer-B3 (depends on $JOB1)"

JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 scripts/slurm_train.sh configs/maxvit_tiny.yaml)
echo "✅ Job $JOB3: Training MaxViT-Tiny (depends on $JOB2)"

# Submit evaluation (depends on all training)
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 scripts/slurm_evaluate.sh)
echo "✅ Job $JOB4: Evaluation (depends on $JOB3)"

# Submit visualization (depends on evaluation)
JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 scripts/slurm_visualize.sh)
echo "✅ Job $JOB5: Visualization (depends on $JOB4)"

echo ""
echo "=========================================="
echo "✅ Pipeline submitted successfully!"
echo "=========================================="
echo ""
echo "Job IDs: $JOB1 → $JOB2 → $JOB3 → $JOB4 → $JOB5"
echo ""
echo "📊 Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "📝 Check training logs:"
echo "  tail -f logs/slurm/train_${JOB1}.out  # ConvNeXt"
echo "  tail -f logs/slurm/train_${JOB2}.out  # SegFormer"
echo "  tail -f logs/slurm/train_${JOB3}.out  # MaxViT"
echo ""
echo "📦 Results will be in:"
echo "  logs/models/{convnext_tiny,segformer_b3,maxvit_tiny}/"
echo ""
