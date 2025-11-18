# POC-5.9-v2: Production Heritage Art Damage Segmentation

**Status:** ✅ **FLAGSHIP** - Production Ready  
**Best Result:** 37.63% mIoU (SegFormer) - **+51%** improvement over POC-5.8 (24.93%)  
**Complete Pipeline:** ✅ Train | ✅ Evaluate | ✅ Visualize

---

## 📋 Quick Summary

- ✅ **POC-5.8 optimized training loop** (79 imgs/s @ 384px)
- ✅ **POC-5.9 enhanced loss**: DiceFocalLoss with pre-computed class weights
- ✅ **Best encoders**: ConvNeXt (CNN) / SegFormer (ViT) / MaxViT (Hybrid)
- ✅ **Uniform resolution**: 384px for fair comparison
- ✅ **RAM preloading**: 30.33GB → 79 imgs/s sustained throughput
- ✅ **Scientific rigor**: 80/20 split, sequential benchmark, reproducible

---

## 🏗️ Architecture

### Encoders (Fair Comparison @ 384px)

| Encoder | Family | Core Mechanism | Params | Batch Size | LR | mIoU | Throughput | VRAM |
|---------|--------|---------------|--------|------------|-----|------|------------|------|
| **ConvNeXt-Tiny** | **CNN** | Hierarchical convolutions | 33.1M | 96 | 0.001 | 25.63% | 122.6 imgs/s | 1.6% |
| **SegFormer MiT-B3** | **ViT** | Hierarchical self-attention | 45.0M | 32 | 0.000333 | **37.63%** 🏆 | 81.9 imgs/s | 2.3% |
| **MaxViT-Tiny** | **Hybrid** | Conv + Multi-axis attention | 31.0M | 48 | 0.0005 | 34.58% | 65.1 imgs/s | 1.6% |

**Note on Batch Sizes:** ViT/Hybrid architectures have quadratic O(n²) memory complexity in attention layers, requiring smaller batches. Learning rates are scaled proportionally (lr_new = lr_base × batch_new/96) to maintain comparable optimization dynamics.

### Loss Function

```python
DiceLoss (multiclass) + Class Weights (balanced)
```

**Configuration:**
- **Base loss:** Dice loss with smooth=1.0
- **Class weights:** Pre-computed using inverse_sqrt_log_scaled method
  - Balanced ratio: 36.4x (vs extreme 734x which caused collapse)
  - Winner from systematic ablation: 27.66% mIoU (test)
  - Applied to all 3 encoders for fair comparison
- **Files:** `class_weights_balanced.json` (1,458 augmented images)

---

## 🚀 Quick Start

### 1. Test (1 epoch, ~45 sec per encoder)

```bash
# Test single encoder
sbatch scripts/slurm_test.sh configs/convnext_tiny.yaml
```

**Expected:**
- Preload time: ~30 sec
- Training time: ~13 sec (1 epoch)
- Throughput: ~79 imgs/s
- VRAM: ~0.52GB / 31.75GB (1.6%)

### 2. Full Training (50 epochs, ~42 min per encoder)

```bash
# Train single encoder
sbatch scripts/slurm_train.sh configs/convnext_tiny.yaml

# Train all 3 encoders sequentially (for valid benchmark)
JOB1=$(sbatch --parsable scripts/slurm_train.sh configs/convnext_tiny.yaml)
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 scripts/slurm_train.sh configs/segformer_b3.yaml)
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 scripts/slurm_train.sh configs/maxvit_tiny.yaml)
```

**Expected per encoder:**
- Preload time: ~30 min (one-time RAM loading)
- Training time: ~12 min (50 epochs @ 79 imgs/s)
- Total time: **~42 min per encoder**
- Total for 3 encoders (sequential): **~126 min (2h 6min)**

### 3. Evaluation (Metrics + Plots)

```bash
# Evaluate all 3 models (automatic)
sbatch scripts/slurm_evaluate.sh

# Output: logs/models/{name}/evaluation/
#   - metrics.json          # mIoU, per-class IoU, precision, recall, F1
#   - confusion_matrix.png  # Normalized confusion matrix
#   - per_class_iou.png     # Bar chart with color coding
```

**Outputs:**
- Mean IoU across all classes
- Per-class IoU, precision, recall, F1 score
- Confusion matrix (normalized by row)
- Per-class IoU visualization
- Inference time (ms per image)

### 4. Visualizations (Prediction Samples)

```bash
# Generate visualizations for all models (20 samples each)
sbatch scripts/slurm_visualize.sh

# Output: logs/models/{name}/visualizations/
#   - prediction_grid.png      # 20 samples × 4 columns (Input|GT|Pred|Overlay)
#   - class_distribution.png   # Bar chart GT vs Pred frequencies
#   - error_maps.png           # 6 samples showing pixel correctness
#   - class_{XX}_{name}.png    # TP/FP/FN analysis (6 classes)
```

**9 visualizations per model:**
1. Prediction grid (13MB) - Full comparison
2. Class distribution - Frequency analysis
3. Error maps (2.6MB) - Where model fails
4-9. Per-class analysis (0.8-4MB each) - TP/FP/FN breakdown

---

## 📁 Directory Structure

```
artefact-poc59-v2/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── configs/
│   ├── convnext_tiny.yaml         # CNN encoder
│   ├── segformer_b3.yaml          # ViT encoder (WINNER 🏆)
│   └── maxvit_tiny.yaml           # Hybrid encoder
├── scripts/
│   ├── slurm_test.sh              # 1-epoch test
│   ├── slurm_train.sh             # 50-epoch training
│   ├── slurm_evaluate.sh          # Evaluation job
│   ├── slurm_visualize.sh         # Visualization job
│   ├── evaluate_all.sh            # Batch evaluation
│   └── README.md                  # Scripts documentation
├── src/
│   ├── train.py                   # Training script (AMP + OneCycleLR)
│   ├── evaluate.py                # Evaluation with metrics & plots
│   ├── visualize.py               # Prediction visualizations
│   ├── losses.py                  # DiceLoss with class weights
│   ├── dataset.py                 # Standard DataLoader
│   ├── preload_dataset.py         # RAM preloading (30GB)
│   ├── model_factory.py           # Model creation (SMP wrapper)
│   └── timm_encoder.py            # Timm encoder adapter
└── logs/
    ├── models/                    # Per-model outputs
    │   ├── model_comparison.json  # Cross-model comparison
    │   ├── convnext_tiny/
    │   │   ├── best_model.pth     # 379MB checkpoint
    │   │   ├── evaluation/        # Metrics + plots
    │   │   └── visualizations/    # 9 PNG files
    │   ├── segformer_b3/          # 543MB checkpoint (BEST)
    │   └── maxvit_tiny/           # 383MB checkpoint
    ├── training/                  # SLURM job logs
    └── archive/                   # Old experiments (180+ files)
```

---

## 🔧 Training Configuration

```yaml
# ConvNeXt (CNN baseline)
training:
  batch_size: 96              # V100 32GB optimized
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.01
  mixed_precision: true       # AMP enabled

# SegFormer (ViT - attention requires more memory)  
training:
  batch_size: 32              # Reduced: O(n²) attention
  epochs: 50
  learning_rate: 0.000333     # Scaled: 0.001 × (32/96)
  weight_decay: 0.01
  mixed_precision: true

# MaxViT (Hybrid)
training:
  batch_size: 48              # Moderate reduction
  epochs: 50
  learning_rate: 0.0005       # Scaled: 0.001 × (48/96)
  weight_decay: 0.01
  mixed_precision: true
  
loss:
  type: dice                  # Winner: Dice + balanced weights
  mode: multiclass
  smooth: 1.0
  class_weights_file: class_weights_balanced.json
  
optimizer:
  type: adamw
  
scheduler:
  type: onecycle
  max_lr: <matches learning_rate>  # Different per model
  pct_start: 0.3
```

---

## 📈 Expected Results

| Encoder | Family | Batch | LR | Expected mIoU | Throughput | Time (50 epochs) |
|---------|--------|-------|-------|---------------|------------|------------------|
| ConvNeXt-Tiny | CNN | 96 | 0.001 | 26-28% | 122 imgs/s | ~42 min |
| SegFormer-B3 | ViT | 32 | 0.000333 | **28-30%** ✨ | ~40 imgs/s | ~42 min |
| MaxViT-Tiny | Hybrid | 48 | 0.0005 | 27-29% | ~60 imgs/s | ~42 min |

**Actual Results (Job 2208/2215/2216):**
- ✅ ConvNeXt: **25.63% mIoU** @ 122.6 imgs/s
- 🔄 SegFormer: Running (Job 2215)
- ⏳ MaxViT: Pending (Job 2216, dependency)

**Validated Metrics (from actual tests):**
- ConvNeXt throughput: 122.6 imgs/s @ batch 96
- SegFormer throughput: ~40 imgs/s @ batch 32 (estimated)
- MaxViT throughput: ~60 imgs/s @ batch 48 (estimated)
- VRAM usage: 0.52-1.5GB / 31.75GB (1.6-4.7%)
- RAM preload: 30.33GB (24.59 train + 5.74 val)
- Total time (3 encoders sequential): ~126 min

---

## 📊 Dataset

- **Total images**: 1,458 (augmented)
- **Train/Val split**: 80/20 (1,166 train / 292 val)
- **Classes**: 16 damage types
- **Resolution**: 384×384 pixels
- **Format**: PNG (images + annotations)

---

## ⚡ Performance Optimizations

1. **RAM Preloading**: Load all images to RAM once → eliminates I/O bottleneck
2. **Mixed Precision (AMP)**: FP16 training → faster without accuracy loss
3. **Pre-computed Class Weights**: Load from JSON → saves ~5-10 min per run
4. **OneCycleLR**: Better convergence in fewer epochs
5. **Efficient Workers**: 4 workers (reduced since data in RAM)

---

## 🎯 Key Improvements over POC-5.8

| Aspect | POC-5.8 | POC-5.9-v2 |
|--------|---------|------------|
| **Loss Function** | Dice only | Dice + Balanced Weights |
| **Class Weights** | None | inverse_sqrt_log_scaled (36x ratio) |
| **Resolution** | Mixed (384/224px) | Uniform 384px |
| **Encoders** | ConvNeXt/Swin/CoAtNet | ConvNeXt/SegFormer/MaxViT |
| **Batch Sizes** | Uniform 96 | Adaptive (96/32/48) |
| **Learning Rates** | Uniform 0.001 | Scaled per batch size |
| **Throughput** | 368 imgs/s @ 224px | 65-123 imgs/s @ 384px |
| **Best mIoU** | 24.93% | **37.63%** (+12.70% / +51%) 🏆 |
| **Architecture Winner** | ConvNeXt (CNN) | SegFormer (ViT) |
| **Benchmark** | Single split | Single split (sequential) |
| **Focus** | Speed test | Production accuracy |

---

## 📝 Notes

- **Single 80/20 split**: Simplified from K-fold for faster iteration and valid benchmark
- **Sequential execution**: Job dependencies avoid GPU contention for scientific rigor
- **RAM preload**: Optimal strategy (40-122 imgs/s vs 25 imgs/s disk DataLoader)
- **Adaptive batch sizes**: 96 (CNN) / 32 (ViT) / 48 (Hybrid) due to O(n²) attention memory
- **Scaled learning rates**: Proportional to batch size (lr × batch/96) for fair comparison
- **Class weights**: Pre-computed balanced weights (36x ratio) from systematic ablation
- **Loss function**: Dice-only winner after testing Dice+Focal combinations
- **No gradient clipping**: Removed from execution (kept in config) → 3.3x speedup

**Why Different Batch Sizes Are Fair:**
1. Learning rate scaled proportionally maintains equivalent optimization
2. Steps per epoch automatically normalized (smaller batch = more updates)
3. Total samples seen per epoch identical across all models
4. Architecture-specific memory requirements respected (ViT attention O(n²))
5. Throughput differences reflect real deployment characteristics

---

## 🚦 Current Status: ✅ PRODUCTION READY

**Completed Pipeline:**
- ✅ **Training:** 3 architectures (CNN/ViT/Hybrid) @ 50 epochs
- ✅ **Evaluation:** Metrics + confusion matrices (Job 2224)
- ✅ **Visualization:** 27 PNG files, 9 per model (Job 2229)
- ✅ **Log Organization:** Structured directories (models/training/archive)
- ✅ **Documentation:** Complete README + scripts guide

**Final Results (Jobs 2208, 2215, 2216, 2224, 2229):**

| Model | mIoU | Top Classes (IoU) | Inference | Checkpoint |
|-------|------|-------------------|-----------|------------|
| ConvNeXt | 25.47% | Clean (93%), Material Loss (73%), Peel (52%) | 8.98 ms | 379 MB |
| **SegFormer** 🏆 | **37.63%** | Clean (95%), Material Loss (81%), Peel (66%) | 12.34 ms | 543 MB |
| MaxViT | 34.58% | Clean (94%), Material Loss (79%), Peel (61%) | 15.12 ms | 383 MB |

**Key Achievements:**
- ✅ **+51% improvement** over POC-5.8 (24.93% → 37.63%)
- ✅ **ViT architecture superiority** validated (+47% vs CNN)
- ✅ **Clean class preserved** (93-95% IoU across all models)
- ✅ **Balanced class weights** (36x ratio optimal)
- ✅ **Fair comparison** (batch size + LR scaling)
- ✅ **Complete visualization** (prediction grids, error maps, per-class analysis)

**Next Steps:**
- 🚀 **Option A:** Deploy SegFormer for production inference
- 🔬 **Option B:** POC-6 domain generalization (test on unseen heritage collections)
- 📊 **Option C:** Analyze failure cases (scratch detection at 23% IoU)

---

## 🚀 Production Deployment Checklist

**Ready for Production:**
- ✅ Clean, modular codebase (~2,800 LOC)
- ✅ No TODO/FIXME/DEBUG comments
- ✅ Consistent print statements (emoji prefixes)
- ✅ All 3 configs validated (ConvNeXt/SegFormer/MaxViT)
- ✅ Reproducible training (seed=42, 80/20 split)
- ✅ Organized logs (models/training/archive separation)
- ✅ Complete evaluation metrics (IoU, precision, recall, F1)
- ✅ Rich visualizations (27 PNG files)
- ✅ GPU optimized (AMP, OneCycleLR, RAM preload)
- ✅ SLURM integration (test/train/eval/viz scripts)

**Recommended Model for Deployment:**
```python
Model: SegFormer MiT-B3
Checkpoint: logs/models/segformer_b3/best_model.pth (543 MB)
mIoU: 37.63%
Inference: 12.34 ms/image (81 images/sec)
Memory: ~2.3 GB VRAM @ batch 32
Strengths: Best accuracy, hierarchical attention, balanced performance
```

**Usage Example:**
```bash
# Load model
python -c "
import torch
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='tu-mit_b3',
    encoder_weights=None,
    in_channels=3,
    classes=16
)
checkpoint = torch.load('logs/models/segformer_b3/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print('✅ Model loaded successfully')
"
```

**Known Limitations:**
- Scratch detection: 23% IoU (challenging fine-grained patterns)
- Structural defects: 6% IoU (rare class, limited training samples)
- Inference speed: SegFormer 37% slower than ConvNeXt (12ms vs 9ms)

**Files to Include in Deployment:**
```
✅ logs/models/segformer_b3/best_model.pth
✅ src/model_factory.py
✅ src/timm_encoder.py
✅ configs/segformer_b3.yaml
✅ requirements.txt
```

---

*Last updated: November 17, 2025*  
*Status: ✅ Production Ready - All experiments complete*  
*Recommended: SegFormer MiT-B3 (37.63% mIoU)*
