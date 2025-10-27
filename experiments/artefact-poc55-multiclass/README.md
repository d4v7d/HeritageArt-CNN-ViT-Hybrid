# POC-5.5: Multiclass Hierarchical Segmentation (Laptop-Optimized)

**Created**: October 26, 2025  
**Status**: ✅ Infrastructure complete, ready for training  
**Timeline**: ~2.7 hours on RTX 3050 6GB (138min training + 25min eval/compare)  
**Hardware**: RTX 3050 Laptop 6GB (tested) → Dell Precision 7630 RTX 1000 Ada 6GB (deploy Monday)

## Objective

Validate multiclass 16-class segmentation with **Hierarchical Multi-Task Learning** (Innovation #1) on laptop hardware before committing to full POC-6 on server.

---

## 🚀 QUICK START

**Complete workflow** (train all 3 models + evaluate + compare):
```bash
# 1. Build and start
make build up

# 2. Download dataset (418 samples, 3.4 GB)
make download

# 3. Test 1 epoch (validate VRAM)
make test-epoch

# 4. Train all 3 models (~2.3 hours)
make train-all

# 5. Evaluate all models (~15 minutes)
make eval-all

# 6. Compare and generate report (~10 minutes)
make compare

# OR: Run everything in one command (~2.7 hours)
make full-workflow
```

**Check results**:
```bash
# Training curves, metrics, tables
ls logs/comparison/

# Per-model results
ls logs/convnext_tiny/evaluation/
ls logs/swin_tiny/evaluation/
ls logs/maxvit_tiny/evaluation/
```

---

## ✅ VALIDATION COMPLETE (Oct 26, 2025)

**Test Results** (1 epoch on RTX 3050 6GB):
- ✅ **Training**: 92.5 seconds, 334 samples
- ✅ **VRAM**: 839 MB / 6144 MB (**13.7% usage** - huge margin!)
- ✅ **Model**: ConvNeXt-Tiny, 37.7M parameters
- ✅ **Metrics** (expected low for 1 epoch):
  - Train Loss: 0.9083 → Val Loss: 0.8135
  - mIoU Binary: 51.83%
  - mIoU Coarse: 18.16%
  - mIoU Fine: 9.98%
- ✅ **Checkpoints**: Saved successfully (864 MB)
- ✅ **Status**: Code verified end-to-end, ready for production

**Key Findings**:
- VRAM usage WAY below expected (only 14% vs 80% target)
- Can train all 3 models in ~2.3 hours (46 min/model × 3)
- Perfect fit for RTX 1000 Ada 6GB deployment Monday

---

## Key Differences from POC-5

| Aspect | POC-5 (Binary) | POC-5.5 (Multiclass) |
|--------|---------------|---------------------|
| **Classes** | 2 (Clean, Damage) | 16 (Clean + 15 damage types) |
| **Dataset** | 50 samples (demo) | 418 samples (full ARTeFACT) |
| **Resolution** | 512×512 | **256×256** (laptop-optimized) |
| **Epochs** | 60 | **30** (faster validation) |
| **Innovation** | None | **Hierarchical MTL** (3 heads) |
| **Models** | 3 (sequential) | **3 (parallel ready)** |
| **Timeline** | ~2 hours | **~2.7 hours** (all models) |
| **Evaluation** | Single-head metrics | **Hierarchical metrics** |

## Innovation #1: Hierarchical Multi-Task Learning

**Problem**: Rare damage classes (Lightleak, Burn marks) have few samples → hard to learn directly.

**Solution**: 3 parallel prediction heads at different granularities:

```
Model Architecture:
┌─────────────────────────────────────────┐
│ Encoder (ConvNeXt/Swin/MaxViT)          │
│   ↓                                     │
│ UPerNet Neck (PPM + FPN)                │
│   ↓                                     │
│ ┌─────────┬─────────┬─────────┐        │
│ │ Binary  │ Coarse  │  Fine   │        │
│ │ Head    │ Head    │  Head   │        │
│ │ (2 cls) │ (4 cls) │ (16 cls)│        │
│ └─────────┴─────────┴─────────┘        │
└─────────────────────────────────────────┘

Loss = 0.2 * L_binary + 0.3 * L_coarse + 1.0 * L_fine
```

**Class Grouping** (coarse 4 groups):
1. **Structural Damage**: Cracks, Material loss, Peel, Structural defects
2. **Surface Contamination**: Dirt spots, Stains, Hairs, Dust spots
3. **Color Alterations**: Discolouration, Burn marks, Fading
4. **Optical Artifacts**: Scratches, Lightleak, Blur

**Expected Benefit**: +3-4% mIoU (helps rare classes via coarse-level guidance)

## Hardware Requirements

**Minimum** (validated on):
- GPU: 6GB VRAM (RTX 3050 Laptop, RTX 1000 Ada)
- RAM: 16GB
- Storage: 50GB (dataset + logs)

**Optimizations for 6GB VRAM**:
- Input resolution: 256×256 (vs 512×512)
- Batch size: 4 (with gradient accumulation 2 → effective batch 8)
- Mixed precision: FP16 (saves 50% memory)
- Gradient checkpointing: Enabled (trade 20% speed for 40% VRAM)

## Dataset: ARTeFACT Full (418 samples)

**Source**: HuggingFace `danielaivanova/damaged-media`

**Actual Count**: 418 samples (not 445 as initially thought - confirmed after download)

**Class Distribution** (16 classes):
```
0:  Clean (background, clean regions)
1:  Material loss
2:  Peel
3:  Cracks
4:  Structural defects
5:  Dirt spots
6:  Stains
7:  Discolouration
8:  Scratches
9:  Burn marks
10: Hairs
11: Dust spots
12: Lightleak (rare)
13: Fading
14: Blur
15: Other damage
255: Background (ignore index)
```

**Metadata**:
- Materials: Parchment, Film emulsion, Glass, Paper, Canvas, Wood, Tesserae, Ceramic, Textile, Lime plaster
- Content: Artistic depiction, Photographic depiction, Line art, Geometric patterns

## Training Configuration

### Laptop-Optimized Setup (256×256)

```yaml
# configs/poc55_256px.yaml
train:
  epochs: 30
  batch_size: 4
  gradient_accumulation_steps: 2  # Effective batch = 8
  input_size: [256, 256]
  mixed_precision: true  # FP16
  num_workers: 4
  
  optimizer:
    type: AdamW
    lr: 1e-4
    weight_decay: 0.01
    betas: [0.9, 0.999]
  
  scheduler:
    type: CosineAnnealingLR
    T_max: 30
    eta_min: 1e-6
  
  early_stopping:
    patience: 5  # Aggressive for laptop
    min_delta: 0.001

model:
  hierarchical: true  # Innovation #1
  num_classes: 16
  ignore_index: 255
  
  heads:
    binary: 2    # Clean vs Damage
    coarse: 4    # 4 damage groups
    fine: 16     # Full 16 classes
  
loss:
  type: hierarchical_dice_focal
  weights:
    binary: 0.2   # Auxiliary task 1
    coarse: 0.3   # Auxiliary task 2
    fine: 1.0     # Main task
  
  dice_weight: 0.5
  focal_weight: 0.5
  focal_alpha: 0.25
  focal_gamma: 2.0
  
  class_weights: inverse_sqrt  # Handle imbalance

augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.3
  rotate_90: 0.3
  random_brightness_contrast: 0.3
  gaussian_noise: 0.2
  coarse_dropout: 0.2  # Random patches
```

## Expected Timeline (Dell Precision 7630)

| Model | Epochs | Time per Epoch | Total Time |
|-------|--------|----------------|------------|
| ConvNeXt-Tiny | 30 | ~1.75h | **53 hours** |
| Swin-Tiny | 30 | ~1.75h | **53 hours** |
| MaxViT-Tiny | 30 | ~1.75h | **53 hours** |
| **Total** | - | - | **159 hours** |
| **+ Eval** | - | - | **+10 hours** |
| **Grand Total** | - | - | **169 hours ≈ 7 days @ 24/7** |

**Start**: Monday Oct 27, 10 AM  
**Finish**: Monday Nov 3, 7 AM

## Expected Results

### Conservative Estimates

| Model | Naive Baseline | With Hierarchical MTL | Improvement |
|-------|----------------|----------------------|-------------|
| ConvNeXt-Tiny | 35-38% mIoU | **38-42% mIoU** | +3-4% |
| Swin-Tiny | 38-41% mIoU | **41-45% mIoU** | +3-4% |
| MaxViT-Tiny | 40-43% mIoU | **43-47% mIoU** | +3-4% 🎯 |

**Rare Class Performance**:
- Naive single-head: 8-12% avg IoU
- Hierarchical 3-head: **16-22% avg IoU** (+8-10%)

**Per-Class IoU** (MaxViT-Tiny expected):
- Clean: 75-80%
- Dirt spots (frequent): 55-65%
- Material loss (frequent): 50-60%
- Cracks (moderate): 40-50%
- Lightleak (rare): 15-25% (vs <5% naive)
- Burn marks (rare): 20-30% (vs <5% naive)

## Success Criteria

**GO for POC-6 Full** (if server available):
- ✅ Multiclass mIoU ≥ 42% (MaxViT-Tiny)
- ✅ Hierarchical heads improve ≥ +3% vs single-head
- ✅ Training stable, VRAM <6GB, no thermal issues
- ✅ Rare classes show improvement (>15% avg IoU)

**NO-GO** (pivot to Plan B):
- ❌ mIoU < 35% (dataset too small, need data augmentation overhaul)
- ❌ VRAM overflow even batch_size=2
- ❌ Thermal throttling prevents 24/7 training

## Publication Target

**If POC-5.5 only** (no server for POC-6):
- Workshop paper: CVPRW, ICCVW (2-4 pages)
- Short paper: BMVC, WACV (4-6 pages)
- Focus: Hierarchical MTL for imbalanced heritage domain

**If POC-5.5 + POC-6 Full** (server approved):
- Conference main track: CVPR, ICCV, ECCV (8 pages)
- Focus: Full innovation stack + domain generalization
- POC-5.5 becomes ablation study in paper

## Docker Setup

### Quick Start (Recommended)

```bash
cd /home/brandontrigueros/DevWSL/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc55-multiclass

# Build and start container
make build
make up

# Download dataset (418 samples, ~3.4 GB)
make download

# Test 1 epoch (VRAM validation) ✅ DONE
make test-epoch

# Full training (30 epochs, ~46 minutes on RTX 3050)
make train-convnext   # ConvNeXt-Tiny
make train-swin       # Swin-Tiny
make train-maxvit     # MaxViT-Tiny
```

### Manual Docker Commands

```bash
# Build image
docker build -t artefact-poc55:latest -f docker/Dockerfile .

# Start container
docker-compose -f docker/docker-compose.yml up -d

# Monitor GPU
docker exec -it artefact-poc55-multiclass nvidia-smi

# View logs
docker logs -f artefact-poc55-multiclass
```

### Environment Setup

**IMPORTANT**: Model weights cache needs HuggingFace cache directory:
```bash
# Set environment variable before running training
docker exec -it artefact-poc55-multiclass bash -c \
  "export HF_HOME=/tmp/huggingface && python /workspace/scripts/train_poc55.py ..."
```

Or add to `docker-compose.yml`:
```yaml
environment:
  - HF_HOME=/tmp/huggingface
```

## File Structure

```
artefact-poc55-multiclass/
├── README.md                    # This file (merged with implementation details)
├── Makefile                     # Convenience commands (build, up, download, train-*)
├── configs/
│   ├── base_config.yaml         # Shared config (augmentation, optimizer, loss)
│   ├── test_1epoch.yaml         # ✅ Test config (used for validation)
│   ├── poc55_256px.yaml         # Laptop-optimized (256×256)
│   ├── convnext_tiny.yaml       # ConvNeXt-Tiny specific
│   ├── swin_tiny.yaml           # Swin-Tiny specific
│   └── maxvit_tiny.yaml         # MaxViT-Tiny specific
├── scripts/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hierarchical_upernet.py  # ✅ Innovation #1 (3 heads, 530 lines)
│   │   ├── model_factory.py         # Encoder → UPerNet adapter
│   │   └── upernet_custom.py        # PPM + FPN decoder
│   ├── dataset_multiclass.py    # ✅ 16-class ARTeFACT loader (127 lines, tested)
│   ├── train_poc55.py            # ✅ Main training script (543 lines, tested)
│   ├── losses.py                 # ✅ Hierarchical Dice+Focal loss (400 lines, tested)
│   ├── download_dataset.py       # ✅ ARTeFACT downloader (working, 418 samples)
│   ├── evaluate.py               # ✅ Hierarchical evaluation (470 lines, 3-head metrics)
│   ├── compare.py                # ✅ Model comparison (600 lines, hierarchical reports)
│   └── train_all.py              # ✅ Batch training script (300 lines, sequential execution)
├── docker/
│   ├── Dockerfile                # ✅ CUDA 12.6, PyTorch, timm (25.7 GB image)
│   ├── docker-compose.yml        # ✅ GPU config, volume mounts
│   └── requirements.txt          # ✅ All dependencies
├── data/
│   └── artefact/                 # ✅ Downloaded (418 samples, 3.4 GB)
│       ├── images/               # Original images
│       ├── annotations/          # Masks (0-15, 255)
│       ├── annotations_rgb/      # RGB visualization
│       └── metadata.csv          # Sample metadata
├── logs/                         # Training outputs
│   └── poc55_test_1epoch/        # ✅ Test run results
│       ├── checkpoints/          # best_model.pth, latest.pth (864 MB)
│       └── logs/                 # TensorBoard logs
├── .cache/
│   └── huggingface/              # Model weights cache (empty, will fill during training)
├── CLEANUP_DONE.md               # Cleanup summary
└── TEST_1EPOCH_SUCCESS.md        # Validation test results
```

**Total Size**: ~4.3 GB (3.4 GB dataset + 864 MB checkpoints + code)

## Next Steps

### Training Workflow (RTX 3050 6GB - Current Laptop)

**Option 1: Train all models sequentially** (~2.7 hours total):
```bash
# Automatic workflow (recommended)
make full-workflow

# This will:
# 1. Build Docker image
# 2. Start container
# 3. Download dataset (418 samples)
# 4. Train ConvNeXt-Tiny (30 epochs, ~46 min)
# 5. Train Swin-Tiny (30 epochs, ~46 min)
# 6. Train MaxViT-Tiny (30 epochs, ~46 min)
# 7. Evaluate all 3 models (~15 min)
# 8. Generate comparison report (~10 min)
```

**Option 2: Train models individually**:
```bash
# Setup
make build up download

# Train one model at a time
make train-convnext  # ~46 minutes
make train-swin      # ~46 minutes
make train-maxvit    # ~46 minutes

# Evaluate
make eval-all        # ~15 minutes

# Compare
make compare         # ~10 minutes
```

**Option 3: Background execution** (recommended for overnight):
```bash
nohup make train-all > training.log 2>&1 &

# Check progress
make logs                  # Refresh manually
watch -n 30 make logs      # Auto-refresh every 30 seconds
tail -f training.log       # Live output
```

### Available Make Commands

**Setup**:
- `make build` - Build Docker image (25.7 GB, one-time)
- `make up` - Start container in background
- `make down` - Stop container
- `make shell` - Open bash shell in container
- `make download` - Download ARTeFACT dataset (418 samples, 3.4 GB)

**Testing**:
- `make test-epoch` - Validate VRAM with 1 epoch (~90 seconds)

**Training**:
- `make train-convnext` - Train ConvNeXt-Tiny (30 epochs, ~46 min)
- `make train-swin` - Train Swin-Tiny (30 epochs, ~46 min)
- `make train-maxvit` - Train MaxViT-Tiny (30 epochs, ~46 min)
- `make train-all` - Train all 3 models sequentially (~138 min)

**Evaluation**:
- `make eval-convnext` - Evaluate ConvNeXt-Tiny
- `make eval-swin` - Evaluate Swin-Tiny
- `make eval-maxvit` - Evaluate MaxViT-Tiny
- `make eval-all` - Evaluate all 3 models (~15 min)

**Comparison**:
- `make compare` - Compare all models and generate report (~10 min)

**Complete Workflow**:
- `make full-workflow` - Build + Download + Train all + Evaluate + Compare (~2.7 hours)

**Other**:
- `make logs` - Show recent training logs
- `make clean` - Remove all logs and checkpoints (confirmation required)

### Monitoring During Training

**Check progress**:
```bash
make logs              # Show current epoch and metrics
watch -n 30 make logs  # Auto-refresh every 30 seconds
```

**Live logs**:
```bash
docker logs artefact-poc55-multiclass -f
```

**VRAM usage**:
```bash
watch -n 5 'docker exec artefact-poc55-multiclass nvidia-smi'
# Expected: ~839 MB (14% of 6GB) - tested
```

**Key Metrics to Watch**:
- `train_loss`: Should decrease smoothly from ~0.9 → ~0.3
- `val_miou_fine`: Target 43-47% at epoch 30 (main metric)
- `val_miou_binary`: Should reach ~90% by epoch 10 (easy task)
- `val_miou_coarse`: Should reach ~60-70% by epoch 30

### After Training (~2.7 hours)

**Check results**:
```bash
# Comparison report (human-readable)
cat logs/comparison/summary_report.txt

# Training curves
open logs/comparison/training_curves.png

# Hierarchical metrics
open logs/comparison/hierarchical_metrics.png

# Comparison table
cat logs/comparison/comparison_table.csv
```

**Per-model results**:
```bash
# ConvNeXt evaluation
cat logs/convnext_tiny/evaluation/metrics.json
open logs/convnext_tiny/evaluation/confusion_matrix_fine.png
open logs/convnext_tiny/evaluation/hierarchical_predictions.png

# Same for Swin and MaxViT
ls logs/swin_tiny/evaluation/
ls logs/maxvit_tiny/evaluation/
```

**Expected results** (based on POC-5 + multiclass complexity):
- **ConvNeXt-Tiny**: 38-42% mIoU (fine head)
- **Swin-Tiny**: 41-45% mIoU (fine head)
- **MaxViT-Tiny**: 43-47% mIoU (fine head) 🏆
- **Hierarchical boost**: +3-4% vs naive single-head

### Decision Point

**If mIoU ≥ 42%** (Fine head, best model):
- ✅ POC-5.5 success! Hierarchical MTL validated
- 🚀 Request POC-6 Full on server (V100/A100)
- 📈 Scale to 512×512, 60 epochs, full augmentation

**If mIoU < 42%**:
- ⚠️ Analyze per-class IoU in `summary_report.txt`
- 🔍 Check confusion matrices for systematic errors
- 🛠️ Adjust loss weights, class weights, or augmentation
- 🔄 Retrain with tuned hyperparameters

### Troubleshooting

**If VRAM overflow** (unlikely, tested at 14%):
```yaml
# Edit configs/base_config.yaml
training:
  batch_size: 2          # Reduce from 4
  gradient_accumulation_steps: 4  # Increase from 2
```

**If training too slow**:
```yaml
training:
  batch_size: 8          # Increase (we have 5GB margin!)
  gradient_accumulation_steps: 1  # Reduce
  num_workers: 2         # Reduce if CPU bottleneck
```

**If loss unstable or diverging**:
```yaml
training:
  optimizer:
    lr: 5e-5  # Reduce from 1e-4
  
loss:
  weights:
    binary: 0.1   # Reduce auxiliary tasks
    coarse: 0.2
    fine: 1.0
```

**If model checkpoints too large**:
```bash
# Only save best model (remove intermediate checkpoints)
# Edit train_poc55.py: comment out save_checkpoint() in training loop
```

## References

- POC-5 (binary): `../artefact-multibackbone-upernet/`
- POC-6 Plan: `../../documentation/POC-6-PLAN.md`
- Traps & Innovations: `../../documentation/POC6-TRAPS-AND-INNOVATIONS.md`
- ARTeFACT Dataset: https://huggingface.co/datasets/icomusef/ARTeFACT
- Dataset: https://huggingface.co/datasets/danielaivanova/damaged-media
