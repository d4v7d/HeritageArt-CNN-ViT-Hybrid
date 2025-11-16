#!/bin/bash
# POC-5.8: 2-Phase Training with Shared Memory Pre-loading
# 
# Phase 1: Pre-load dataset to /dev/shm ONCE (30 min)
# Phase 2: Train all 3 models in parallel reading from /dev/shm

echo "================================================"
echo "POC-5.8: 2-Phase Parallel Training"
echo "================================================"
echo ""

cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard

echo "📝 Phase 1: Pre-loading dataset to shared memory..."
echo ""

# Submit pre-load job
PRELOAD_JOB=$(sbatch --parsable scripts/slurm_preload.sh)
echo "   Pre-load Job ID: $PRELOAD_JOB"
echo ""

echo "📝 Phase 2: Training jobs (wait for pre-load to finish)..."
echo ""

# Create temporary configs pointing to /dev/shm
TEMP_DIR="/tmp/poc58_configs_$$"
mkdir -p $TEMP_DIR

for config in convnext_tiny swin_tiny coatnet_0; do
    # Copy config and modify data_dir
    sed 's|data_dir: ../data/artefact|data_dir: /dev/shm/artefact_cache|g; s|use_preload: true|use_preload: false  # Already in /dev/shm|g' \
        configs/${config}.yaml > $TEMP_DIR/${config}.yaml
done

echo "   Created temp configs in: $TEMP_DIR"
echo ""

# Job 1: ConvNeXt (waits for pre-load, then runs)
echo "1️⃣  ConvNeXt-Tiny..."
JOB1=$(sbatch --parsable --gres=gpu:1 --dependency=afterok:$PRELOAD_JOB \
    scripts/slurm_train.sh $TEMP_DIR/convnext_tiny.yaml)
echo "   Job ID: $JOB1 (depends on pre-load)"

# Job 2: Swin (waits for pre-load, runs parallel with ConvNeXt)
echo "2️⃣  Swin-Tiny..."
JOB2=$(sbatch --parsable --gres=gpu:1 --dependency=afterok:$PRELOAD_JOB \
    scripts/slurm_train.sh $TEMP_DIR/swin_tiny.yaml)
echo "   Job ID: $JOB2 (depends on pre-load)"

# Job 3: CoAtNet (waits for either Job1 or Job2 to finish)
echo "3️⃣  CoAtNet-0..."
JOB3=$(sbatch --parsable --gres=gpu:1 --dependency=afterany:$JOB1,$JOB2 \
    scripts/slurm_train.sh $TEMP_DIR/coatnet_0.yaml)
echo "   Job ID: $JOB3 (waits for first GPU to free)"

echo ""
echo "================================================"
echo "✅ All jobs submitted!"
echo "================================================"
echo ""
echo "Job chain:"
echo "  Phase 1: Pre-load ($PRELOAD_JOB) - ~30 min"
echo "  Phase 2a: Jobs $JOB1 & $JOB2 run parallel - ~15 min"
echo "  Phase 2b: Job $JOB3 runs after - ~15 min"
echo ""
echo "Total time: ~30min + 15min + 15min = 60 min"
echo "  vs old: 40min + 40min + 40min = 120 min (2x faster!)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/preload_$PRELOAD_JOB.out  # Pre-load"
echo "  tail -f logs/train_$JOB1.out  # ConvNeXt"
echo "  tail -f logs/train_$JOB2.out  # Swin"
echo "  tail -f logs/train_$JOB3.out  # CoAtNet"
echo ""
echo "Cleanup after training:"
echo "  python scripts/preload_shared_dataset.py --cleanup"
echo ""
