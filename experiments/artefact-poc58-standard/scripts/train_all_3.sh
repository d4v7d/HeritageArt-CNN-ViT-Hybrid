#!/bin/bash
# POC-5.8: Train all 3 architectures (ConvNeXt, Swin, MaxViT)

echo "================================================"
echo "POC-5.8: Training 3 Architectures"
echo "================================================"
echo ""

# Change to experiments directory
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard

# Submit 3 jobs in sequence (one after another to avoid GPU conflicts)
echo "📝 Submitting jobs..."
echo ""

# Job 1: ConvNeXt-Tiny
echo "1️⃣  ConvNeXt-Tiny..."
JOB1=$(sbatch --parsable scripts/slurm_train.sh ../configs/convnext_tiny.yaml)
echo "   Job ID: $JOB1"

# Job 2: Swin-Tiny (depends on Job 1)
echo "2️⃣  Swin-Tiny..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm_train.sh ../configs/swin_tiny.yaml)
echo "   Job ID: $JOB2"

# Job 3: MobileViTv2-200 (depends on Job 2)
echo "3️⃣  MobileViTv2-200 (Hybrid CNN+ViT)..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 scripts/slurm_train.sh ../configs/mobilevit_v2_200.yaml)
echo "   Job ID: $JOB3"

echo ""
echo "================================================"
echo "✅ All jobs submitted!"
echo "================================================"
echo ""
echo "Job chain: $JOB1 → $JOB2 → $JOB3"
echo ""
echo "Estimated completion time: ~45-50 minutes"
echo "  - ConvNeXt-Tiny:     ~15-16 min"
echo "  - Swin-Tiny:         ~15-16 min"
echo "  - MobileViTv2-200:   ~12-14 min"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/train_$JOB1.out  # ConvNeXt"
echo "  tail -f logs/train_$JOB2.out  # Swin"
echo "  tail -f logs/train_$JOB3.out  # MaxViT"
echo ""
