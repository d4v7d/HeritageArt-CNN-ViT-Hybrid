#!/bin/bash
#
# Submit all 9 training jobs for POC-5.9 (3 models × 3 folds)
#
# Usage: bash scripts/submit_all.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"

cd "$WORK_DIR"

echo "================================"
echo "POC-5.9: Submitting 9 training jobs"
echo "================================"

MODELS=("convnext_tiny" "segformer_b3" "maxvit_tiny")
FOLDS=(0 1 2)

JOB_IDS=()

for MODEL in "${MODELS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        echo ""
        echo "Submitting: $MODEL, Fold $FOLD"
        
        JOB_ID=$(sbatch --export=MODEL=$MODEL,FOLD=$FOLD \
                       --job-name="poc59-${MODEL}-f${FOLD}" \
                       scripts/train_fold.sh | awk '{print $NF}')
        
        JOB_IDS+=($JOB_ID)
        echo "  → Job ID: $JOB_ID"
        
        # Small delay to avoid overwhelming scheduler
        sleep 1
    done
done

echo ""
echo "================================"
echo "All jobs submitted!"
echo "Job IDs: ${JOB_IDS[@]}"
echo "================================"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs:"
echo "  tail -f logs/slurm/poc59-*_fold*_*.out"
echo ""
