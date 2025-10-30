#!/bin/bash
#
#SBATCH --job-name=poc55-maxvit
# NOTE: write Slurm stdout/stderr into the experiment logs folder (uses relative path from submit dir)
#SBATCH --output=experiments/artefact-poc55-multiclass/logs/poc55-maxvit.%j.out
#SBATCH --error=experiments/artefact-poc55-multiclass/logs/poc55-maxvit.%j.err
#
#SBATCH --nodes=1
#SBATCH --mem=48gb
#SBATCH --partition=gpu-wide
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

# Load modules
module load miniconda3 cuda11.4

# Activate conda (try common locations)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
	source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/modules/miniconda3/etc/profile.d/conda.sh" ]; then
	source "/opt/modules/miniconda3/etc/profile.d/conda.sh"
else
	# fallback to conda shell hook if available in PATH
	if command -v conda >/dev/null 2>&1; then
		eval "$(conda shell.bash hook)"
	fi
fi

# Ensure the 'poc55' env exists and activate it
if ! conda info --envs | awk '{print $1}' | grep -q "^poc55$$"; then
	conda create -n poc55 python=3.11 -y || true
fi
conda activate poc55 || true

if [ -n "$SLURM_SUBMIT_DIR" ]; then
	# When running under Slurm, use the submission directory (repo root)
	EXP_DIR="$SLURM_SUBMIT_DIR/experiments/artefact-poc55-multiclass"
else
	EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

# Install dependencies (first run only) using absolute paths
REQ_FILE="$EXP_DIR/docker/requirements.txt"
if [ -f "$REQ_FILE" ]; then
	pip install -r "$REQ_FILE" || echo "⚠️ pip install failed or no internet — ensure dependencies are preinstalled."
else
	echo "⚠️ Requirements file not found: $REQ_FILE"
fi

# Ensure dataset exists (download if missing) using absolute paths
DATA_IMAGES="$EXP_DIR/data/artefact/images"
if [ ! -d "$DATA_IMAGES" ]; then
	echo "Dataset not found in $EXP_DIR/data/artefact. Attempting download..."
	python "$EXP_DIR/scripts/download_dataset.py" --output-dir "$EXP_DIR/data/artefact" || {
		echo "⚠️ Dataset download failed. Please download manually or check network/access to HuggingFace.";
	}
fi

# Run training from experiment folder
cd "$EXP_DIR" || exit 1
python scripts/train_poc55.py --config configs/maxvit_tiny.yaml --output-dir logs

