#!/bin/bash
#
#SBATCH --job-name=poc55-maxvit
#SBATCH --output=poc55-maxvit.out
#SBATCH --error=poc55-maxvit.err
#
#SBATCH --nodes=1
#SBATCH --mem=60gb
#SBATCH --partition=gpu-wide
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

# Load modules
module load miniconda3 cuda11.4

# Activate conda (adjust path if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate poc55 || conda create -n poc55 python=3.11 -y && conda activate poc55

# Install dependencies (first run only)
pip install -r ../requirements.txt

# Run training
cd experiments/artefact-poc55-multiclass
python scripts/train_poc55.py --config configs/maxvit_tiny.yaml --output-dir logs

