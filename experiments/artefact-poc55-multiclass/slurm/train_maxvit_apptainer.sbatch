#!/bin/bash
# SBATCH wrapper to run MaxViT training inside an Apptainer/Singularity image
# Usage: sbatch train_maxvit_apptainer.sbatch /path/to/image.sif
# Default: expects image at $SLURM_SUBMIT_DIR/experiments/artefact-poc55-multiclass/artifact_cuda11.4.sif
#SBATCH --job-name=poc55-maxvit-sif
#SBATCH --output=../logs/poc55-maxvit-sif.%j.out
#SBATCH --error=../logs/poc55-maxvit-sif.%j.err
#SBATCH --nodes=1
#SBATCH --mem=48gb
#SBATCH --partition=gpu-wide
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

IMAGE="$1"
if [ -z "$IMAGE" ]; then
    IMAGE="$SLURM_SUBMIT_DIR/experiments/artefact-poc55-multiclass/artefact_poc55_cuda11.4.sif"
fi

echo "Using image: $IMAGE"

module load apptainer || true

EXP_DIR="$SLURM_SUBMIT_DIR/experiments/artefact-poc55-multiclass"

mkdir -p "$EXP_DIR/logs"

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: image not found: $IMAGE" >&2
    exit 1
fi

echo "Running training inside Apptainer..."
srun apptainer exec --nv --bind "$SLURM_SUBMIT_DIR:/workspace" "$IMAGE" \
    python /workspace/experiments/artefact-poc55-multiclass/scripts/train_poc55.py \
    --config /workspace/experiments/artefact-poc55-multiclass/configs/maxvit_tiny.yaml \
    --output-dir /workspace/experiments/artefact-poc55-multiclass/logs
