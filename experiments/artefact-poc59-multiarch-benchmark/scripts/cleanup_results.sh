#!/bin/bash
# POC-59 Results Cleanup Script
# Cleans all logs and results (preserves backup folder and configs)

set -e  # Exit on error

echo "=========================================="
echo "POC-59 Results Cleanup"
echo "=========================================="
echo ""

# Check we're in the right directory
if [ ! -f "scripts/slurm_train.sh" ]; then
    echo "❌ Error: Run this script from artefact-poc59-multiarch-benchmark/"
    exit 1
fi

echo "⚠️  This will DELETE:"
echo "  - logs/models/*/evaluation/"
echo "  - logs/models/*/visualizations/"
echo "  - logs/models/*/best_model.pth"
echo "  - logs/results/*"
echo "  - logs/slurm/*.out and *.err"
echo ""
echo "✅ This will PRESERVE:"
echo "  - experiments/backup/ (archived results)"
echo "  - configs/ (model configurations)"
echo "  - src/ (source code)"
echo "  - scripts/ (SLURM scripts)"
echo ""

read -p "Continue with cleanup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 1
fi

echo ""
echo "🧹 Cleaning results..."
echo ""

# Clean model evaluations
if [ -d "logs/models/convnext_tiny/evaluation" ]; then
    rm -rf logs/models/convnext_tiny/evaluation/
    echo "  ✅ Removed convnext_tiny/evaluation/"
fi

if [ -d "logs/models/segformer_b3/evaluation" ]; then
    rm -rf logs/models/segformer_b3/evaluation/
    echo "  ✅ Removed segformer_b3/evaluation/"
fi

if [ -d "logs/models/maxvit_tiny/evaluation" ]; then
    rm -rf logs/models/maxvit_tiny/evaluation/
    echo "  ✅ Removed maxvit_tiny/evaluation/"
fi

# Clean model visualizations
if [ -d "logs/models/convnext_tiny/visualizations" ]; then
    rm -rf logs/models/convnext_tiny/visualizations/
    echo "  ✅ Removed convnext_tiny/visualizations/"
fi

if [ -d "logs/models/segformer_b3/visualizations" ]; then
    rm -rf logs/models/segformer_b3/visualizations/
    echo "  ✅ Removed segformer_b3/visualizations/"
fi

if [ -d "logs/models/maxvit_tiny/visualizations" ]; then
    rm -rf logs/models/maxvit_tiny/visualizations/
    echo "  ✅ Removed maxvit_tiny/visualizations/"
fi

# Clean model checkpoints
if [ -f "logs/models/convnext_tiny/best_model.pth" ]; then
    rm -f logs/models/convnext_tiny/best_model.pth
    echo "  ✅ Removed convnext_tiny/best_model.pth"
fi

if [ -f "logs/models/segformer_b3/best_model.pth" ]; then
    rm -f logs/models/segformer_b3/best_model.pth
    echo "  ✅ Removed segformer_b3/best_model.pth"
fi

if [ -f "logs/models/maxvit_tiny/best_model.pth" ]; then
    rm -f logs/models/maxvit_tiny/best_model.pth
    echo "  ✅ Removed maxvit_tiny/best_model.pth"
fi

# Clean results folder
if [ -d "logs/results" ]; then
    rm -rf logs/results/*
    echo "  ✅ Cleaned logs/results/"
fi

# Clean SLURM logs
if [ -d "logs/slurm" ]; then
    rm -f logs/slurm/*.out logs/slurm/*.err
    echo "  ✅ Cleaned logs/slurm/*.out and *.err"
fi

echo ""
echo "=========================================="
echo "✅ Cleanup complete!"
echo "=========================================="
echo ""
echo "Preserved:"
echo "  📦 experiments/backup/ (archived results)"
echo "  ⚙️  configs/ (model configurations)"
echo "  📂 Directory structure intact"
echo ""
echo "Ready to run: ./scripts/run_pipeline.sh"
echo ""
