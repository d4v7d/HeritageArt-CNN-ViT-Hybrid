#!/bin/bash
# POC-5.8: Train all 3 architectures in parallel (2 GPUs)
# Strategy: Launch 2 jobs in parallel, then 3rd job when first one finishes

echo "================================================"
echo "POC-5.8: Parallel Training (2 GPUs)"
echo "================================================"
echo ""

# Change to experiments directory
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard

echo "📝 Submitting parallel jobs..."
echo ""

# Job 1: ConvNeXt-Tiny (GPU 0)
echo "1️⃣  ConvNeXt-Tiny (GPU 0)..."
JOB1=$(sbatch --parsable --gres=gpu:1 scripts/slurm_train.sh configs/convnext_tiny.yaml)
echo "   Job ID: $JOB1"

# Job 2: Swin-Tiny (GPU 1, parallel with Job 1)
echo "2️⃣  Swin-Tiny (GPU 1, parallel)..."
JOB2=$(sbatch --parsable --gres=gpu:1 scripts/slurm_train.sh configs/swin_tiny.yaml)
echo "   Job ID: $JOB2"

# Job 3: CoAtNet-0 (waits for EITHER Job1 or Job2 to finish)
echo "3️⃣  CoAtNet-0 (waits for first GPU to free)..."
JOB3=$(sbatch --parsable --gres=gpu:1 --dependency=afterany:$JOB1:$JOB2 scripts/slurm_train.sh configs/coatnet_0.yaml)
echo "   Job ID: $JOB3"

echo ""
echo "================================================"
echo "✅ All jobs submitted!"
echo "================================================"
echo ""
echo "Strategy:"
echo "  - Jobs $JOB1 & $JOB2 run in parallel (2 GPUs)"
echo "  - Job $JOB3 starts when either finishes"
echo ""
echo "Estimated completion time: ~25-30 minutes"
echo "  - Phase 1 (parallel): ConvNeXt + Swin (~15-16 min)"
echo "  - Phase 2 (single):   CoAtNet (~15-16 min)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/train_$JOB1.out  # ConvNeXt"
echo "  tail -f logs/train_$JOB2.out  # Swin"
echo "  tail -f logs/train_$JOB3.out  # CoAtNet"
echo ""
