#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --output=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass/logs/slurm/gpu_check_%j.out
#SBATCH --error=/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass/logs/slurm/gpu_check_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

echo "=========================================="
echo "GPU Hardware Check"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

module load cuda11.4

echo "🔍 All GPUs on this node:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv
echo ""

echo "🔍 Detailed GPU info:"
nvidia-smi
echo ""

echo "🔍 CUDA Environment:"
nvcc --version
echo ""

# Load conda and check PyTorch
source /opt/modules/miniconda3/etc/profile.d/conda.sh
conda activate poc55

echo "🔥 PyTorch GPU Check:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'\\nGPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Total memory: {props.total_memory / 1024**3:.2f} GB')
    print(f'  Multi-processor count: {props.multi_processor_count}')
    print(f'  CUDA capability: {props.major}.{props.minor}')
"
