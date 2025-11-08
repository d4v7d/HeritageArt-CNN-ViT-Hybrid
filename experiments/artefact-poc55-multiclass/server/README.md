# Server Scripts for POC-5.5 (SLURM Deployment)

This directory contains scripts for running POC-5.5 on the CITIC SLURM cluster.

## 📁 Files

- `setup_poc55_complete.sh` - Install all dependencies in conda environment
- `download_dataset_server.sh` - Smart dataset download with verification
- `slurm_train_convnext.sh` - SLURM job for ConvNeXt-Tiny training
- `slurm_train_swin.sh` - SLURM job for Swin-Tiny training
- `slurm_train_maxvit.sh` - SLURM job for MaxViT-Tiny training

## 🚀 Usage

Use the Makefile.server from the parent directory:

```bash
# From artefact-poc55-multiclass/
make -f Makefile.server submit-all     # Submit all 3 models
make -f Makefile.server status         # Check job status
make -f Makefile.server logs-live      # Follow logs
```

## 📋 Requirements

- SLURM cluster with GPU partition
- CUDA 11.4
- Conda environment: poc55
- Dataset: ARTeFACT (418 images)

## 🔧 SLURM Configuration

- **Partition**: `gpu-long`
- **GPUs**: 1x Tesla V100 (or similar)
- **Memory**: 32 GB
- **Time limit**: 2 hours per model
- **CPUs**: 4 cores

## 📊 Expected Training Time

- ConvNeXt-Tiny: ~1.5 hours (30 epochs)
- Swin-Tiny: ~1.5 hours (30 epochs)
- MaxViT-Tiny: ~1.5 hours (30 epochs)
