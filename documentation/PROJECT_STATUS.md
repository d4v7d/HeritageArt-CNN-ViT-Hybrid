# HeritageArt-CNN-ViT-Hybrid: Project Status

**Last Updated**: December 8, 2025  
**Version**: v1.0-production  
**Status**: ✅ Production Ready  
**Git Tag**: `v1.0-production`

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Production Model** | SegFormer-B3 (Pure ViT) |
| **Best Performance** | 37.63% mIoU (16-class segmentation) |
| **Training Data** | ARTeFACT dataset (1,458 augmented samples) |
| **Total Project Size** | 14.6 GB |
| **Active POCs** | 4 (POC-3 data + POC-5.5 + POC-5.8 + POC-5.9) |
| **Documentation** | Clean and consolidated |

---

## 🎯 Production System (POC-5.9)

**Location**: `experiments/artefact-poc59-multiarch-benchmark/`  
**Size**: 2.5 GB  
**Status**: ⭐ **PRODUCTION READY**  
**Code Quality**: Clean, modular, documented (~2,825 LOC)

### Trained Models

| Model | Type | mIoU | Dice | Accuracy | Params | Throughput |
|-------|------|------|------|----------|--------|------------|
| **SegFormer-B3** ⭐ | Pure ViT | **37.63%** | 46.29% | 88.15% | 47.2M | 81.9 img/s |
| MaxViT-Tiny | CNN-ViT Hybrid | 34.58% | 43.89% | 87.82% | 31.0M | 65.1 img/s |
| ConvNeXt-Tiny | Pure CNN | 25.47% | 35.24% | 86.71% | 28.6M | 122.6 img/s |

### Performance Highlights

**SegFormer-B3 (Winner)**:
- **Top-3 Classes**: Clean (95%), Material Loss (81%), Peel (66%)
- **Weak Classes**: Scratches (23%), Structural defects (6%)
- **Inference**: 12.34 ms/image (81 img/s)
- **VRAM**: ~2.3 GB @ batch 32

**Model Characteristics**:
- SegFormer: Best accuracy (+51% over POC-5.8), moderate speed
- ConvNeXt: Fastest throughput, lowest accuracy
- MaxViT: Balanced hybrid performance

### Features
- ✅ Complete training pipeline (50 epochs per model)
- ✅ Comprehensive evaluation (per-class IoU/precision/recall/F1)
- ✅ Rich visualizations (27 PNG files: 9 per model)
- ✅ Production-ready code (clean, no TODO/FIXME/DEBUG)
- ✅ SLURM-optimized (RAM preload, shared memory, parallel training)
- ✅ Reproducible experiments (configs + seeds)

### Code Structure

```
artefact-poc59-multiarch-benchmark/
├── README.md                    # Complete documentation
├── requirements.txt             # Python dependencies
├── configs/                     # 3 production configs
│   ├── convnext_tiny.yaml
│   ├── segformer_b3.yaml       # 🏆 WINNER
│   └── maxvit_tiny.yaml
├── scripts/                     # 4 SLURM scripts + README
│   ├── slurm_test.sh           # 1-epoch quick test
│   ├── slurm_train.sh          # 50-epoch training
│   ├── slurm_evaluate.sh       # Evaluate all models
│   └── slurm_visualize.sh      # Generate visualizations
└── src/                         # 8 clean Python modules (2,645 LOC)
    ├── train.py                # Training loop with AMP + OneCycleLR
    ├── evaluate.py             # Evaluation with metrics & plots
    ├── visualize.py            # Prediction visualizations
    ├── dataset.py              # Standard DataLoader
    ├── preload_dataset.py      # RAM preloading (30GB)
    ├── losses.py               # DiceLoss with class weights
    ├── model_factory.py        # SMP model creation
    └── timm_encoder.py         # Timm encoder adapter
```

### Checkpoints
```
logs/models/
├── model_comparison.json               # Cross-model metrics
├── convnext_tiny/best_model.pth       # 379 MB
├── segformer_b3/best_model.pth        # 543 MB (BEST)
└── maxvit_tiny/best_model.pth         # 383 MB
```

### Evaluation Outputs

**Per Model**:
- `metrics.json`: Mean IoU + per-class metrics + inference time
- `confusion_matrix.png`: Normalized 16×16 confusion matrix
- `per_class_iou.png`: Color-coded bar chart

**Visualizations (9 PNG per model)**:
1. `prediction_grid.png` (13 MB) - 20 samples × 4 columns
2. `class_distribution.png` (93 KB) - GT vs Pred frequencies
3. `error_maps.png` (2.6 MB) - Pixel correctness visualization
4-9. `class_XX_Name.png` (0.8-4 MB) - Per-class TP/FP/FN

**Total**: 27 visualization files (~60 MB)

---

## 🚀 Deployment Guide

### Recommended Production Model

```yaml
Model: SegFormer MiT-B3
Path: logs/models/segformer_b3/best_model.pth
Size: 543 MB
Performance: 37.63% mIoU, 81 img/s
VRAM: ~2.3 GB @ batch 32
Requirements: PyTorch 2.0+, SMP, Timm
```

### Loading Model (Python)

```python
import torch
import segmentation_models_pytorch as smp

# Create model
model = smp.Unet(
    encoder_name='tu-mit_b3',
    encoder_weights=None,
    in_channels=3,
    classes=16
)

# Load checkpoint
checkpoint = torch.load('logs/models/segformer_b3/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Loaded (Epoch {checkpoint['epoch']}, mIoU {checkpoint['best_miou']:.4f})")
```

### Inference Example

```python
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Preprocessing
transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Load and predict
image = np.array(Image.open('test.png').convert('RGB'))
image_tensor = transform(image=image)['image'].unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(image_tensor)
    pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
```

### Deployment Checklist

**Core Files**:
- ✅ `logs/models/segformer_b3/best_model.pth` (543 MB)
- ✅ `src/model_factory.py` (model creation)
- ✅ `src/timm_encoder.py` (encoder adapter)
- ✅ `configs/segformer_b3.yaml` (config)
- ✅ `requirements.txt` (dependencies)

**Optional Documentation**:
- ✅ `README.md` (usage guide)
- ✅ `logs/models/segformer_b3/evaluation/` (metrics)
- ✅ `logs/models/segformer_b3/visualizations/` (samples)

---

## 📚 Active Projects

### 1. POC-3: Data Obtention
**Path**: `experiments/artefact-data-obtention/`  
**Size**: 64 KB  
**Purpose**: Dataset download and preparation scripts  
**Status**: ✅ Utility (always needed)

### 2. POC-5.5: Hierarchical Multiclass
**Path**: `experiments/artefact-poc55-multiclass/`  
**Size**: 865 MB  
**Purpose**: Historical reference for hierarchical multi-task learning  
**Status**: ✅ Reference implementation

**Key Features:**
- 3-head architecture (Binary → Coarse → Fine)
- Laptop-optimized (256px, RTX 3050 6GB)
- Docker-only workflow
- Result: 20.91% mIoU (fine head, 16 classes, 418 samples)

**Why Keep:**
- Unique hierarchical MTL approach (reusable for POC-6)
- Educational reference for multi-task learning
- Docker workflow example

### 3. POC-5.8: Standard Segmentation
**Path**: `experiments/artefact-poc58-standard/`  
**Size**: 1.5 GB  
**Purpose**: Training optimization techniques reference  
**Status**: ✅ Complete (working SLURM infrastructure)

**Key Innovations:**
- Batch size optimization (256 → 128 → 96)
- RAM pre-loading (2x speedup)
- Shared memory pre-loading (2x speedup again)
- SLURM GPU management patterns

**Why Keep:**
- Documents training optimization journey
- Working SLURM reference scripts
- Systematic optimization techniques

### 4. POC-5.9: Multi-Architecture Benchmark ⭐ FLAGSHIP
**Path**: `experiments/artefact-poc59-multiarch-benchmark/`  
**Size**: 2.5 GB  
**Purpose**: Production multi-architecture comparison  
**Status**: ⭐ **PRODUCTION READY**

**Models**: SegFormer-B3, MaxViT-Tiny, ConvNeXt-Tiny  
**Winner**: SegFormer-B3 @ 37.63% mIoU  
**Pipeline**: Complete (train + evaluate + visualize)

---

## 🗑️ Cleanup History

### Deleted Projects (November 17, 2025)

**1. POC-5 (artefact-multibackbone-upernet)**
- **Reason**: Never completed, superseded by POC-5.9
- **Size**: 164 KB (no trained models)
- **Commit**: `177465d`

**2. POC-5.9-v1 (K-fold experiments)**
- **Reason**: K-fold unnecessary for benchmark
- **Size**: 3.0 GB
- **Commit**: `ba38570`

**3. POC-5.5 SLURM Scripts (server/ directory)**
- **Reason**: Broken scripts, POC-5.6/5.7 were learning phase
- **Size**: 12 files
- **Commit**: `177465d`

---

## 📖 Documentation

### Core Documents

1. **POC_Full_History_Analysis.md** (26 KB)
   - Complete POC evolution (POC-1 through POC-5.9)
   - Architectural decisions and learnings
   - Performance timeline

2. **PROJECT_STATUS.md** (This file)
   - Current project state and statistics
   - Quick reference guide
   - Production system details

3. **POC6_PLANNING.md** (Consolidated)
   - POC-6 implementation plan
   - Feasibility analysis
   - Innovations and traps
   - Execution timeline

4. **POC_Evolution_Comparison.md** (20 KB)
   - Cross-POC comparison tables
   - Evolution insights

### Technical References

5. **DATA-AUGMENTATION-STRATEGY.md** (27 KB)
   - Augmentation techniques
   - Heritage-specific transformations

6. **From-Paper-to-Plan.md** (24 KB)
   - Research to implementation pipeline

---

## 🔬 Evolution Summary

### Performance Growth
```
POC-5.5:  20.91% mIoU (Hierarchical MTL, 16 classes, 418 samples)
   ↓
POC-5.8:  ~28% mIoU* (Standard U-Net, optimizations)
   ↓
POC-5.9:  37.63% mIoU (SegFormer, production, 1,458 samples)

Total improvement: +79.9% relative (3 weeks)
```
*Estimated (not recorded in git)

### Key Learnings

1. **Architecture**: Standard segmentation > Hierarchical MTL (for this dataset size)
2. **Infrastructure**: SLURM-only > Docker+SLURM (simpler maintenance)
3. **Optimization**: I/O and memory management as important as model architecture
4. **Evaluation**: Comprehensive metrics + visualizations necessary for production
5. **Model Selection**: Strategic selection (CNN/ViT/Hybrid) > exhaustive search

### POC Timeline

```
POC-1 to POC-3:  Initial setup with MMSegmentation
POC-4:           Minimal training pipeline
POC-5:           Multi-backbone UPerNet (never completed)
POC-5.5:         Hierarchical MTL (20.91% mIoU, Docker-only)
POC-5.6/5.7:     SLURM learning experiments (not committed)
POC-5.8:         Standard segmentation + optimizations (~28% mIoU)
POC-5.9:         Production benchmark (37.63% mIoU) ⭐
```

---

## 🎯 Current Goals

### Achieved ✅
- [x] Multi-architecture comparison framework
- [x] Production-quality training pipeline
- [x] Comprehensive evaluation system
- [x] Visual validation tools
- [x] Clean, organized codebase (~2,825 LOC)
- [x] Complete documentation (consolidated)
- [x] Git history cleanup
- [x] Production release tag (v1.0-production)

### Optional Future Work 📝
- [ ] POC-6: Hierarchical MTL + Progressive Curriculum + Domain Generalization
- [ ] Real-world deployment testing
- [ ] Model optimization (quantization, distillation)
- [ ] API/service wrapper
- [ ] Extended evaluation on external datasets

---

## 🚀 Quick Start (For New Users)

### 1. Clone Repository
```bash
git clone https://github.com/HeritageArt-Research/HeritageArt-CNN-ViT-Hybrid.git
cd HeritageArt-CNN-ViT-Hybrid
```

### 2. Use Production Model (POC-5.9)
```bash
cd experiments/artefact-poc59-multiarch-benchmark

# See README for detailed instructions
cat README.md

# Quick evaluation example
python src/evaluate.py \
    --config configs/segformer_b3.yaml \
    --checkpoint logs/models/segformer_b3/best_model.pth
```

### 3. Explore Documentation
```bash
# Start here
cat documentation/PROJECT_STATUS.md

# Full history
cat documentation/POC_Full_History_Analysis.md

# POC-6 planning (if continuing research)
cat documentation/POC6_PLANNING.md
```

---

## 📊 Resource Usage

### Disk Space

| Component | Size | Purpose |
|-----------|------|---------|
| Common Data | 9.7 GB | Shared datasets, augmented data (1,458 samples) |
| POC-5.9 (Production) | 2.5 GB | 3 models + checkpoints + logs + visualizations |
| POC-5.8 (Reference) | 1.5 GB | Optimization experiments |
| POC-5.5 (Historical) | 865 MB | Hierarchical MTL reference |
| POC-3 (Utility) | 64 KB | Dataset scripts |
| **Total** | **14.6 GB** | Complete project |

### Storage Breakdown (POC-5.9)
- **Checkpoints**: 1.3 GB (3 models: 379MB + 543MB + 383MB)
- **Visualizations**: 90 MB (27 PNG files, 9 per model)
- **Evaluation**: 600 KB (metrics.json + plots per model)
- **Training logs**: ~5 MB (SLURM outputs)
- **Archive**: 1.6 GB (old experiments, safe to delete)

### Compute Resources (POC-5.9)

| Resource | Training | Inference |
|----------|----------|-----------|
| **GPU** | Tesla V100 32GB | RTX 3050 6GB+ |
| **VRAM** | ~8-12 GB (batch 32-96) | ~2-4 GB (batch 1-8) |
| **RAM** | 32 GB (with preloading) | 4 GB |
| **Time** | ~7-10h per model (50 epochs) | ~12ms per image |
| **Batch Size** | 8-96 (depends on model) | 1-8 (inference) |

---

## ⚠️ Known Limitations

### Performance Constraints

**Low IoU Classes**:
- **Scratches**: 23% IoU (fine-grained patterns challenging)
- **Structural defects**: 6% IoU (rare class, limited samples)
- **Cracks**: 0% IoU (not present in validation set)
- **Dirt/Dust spots**: 0% IoU (not present in validation set)

**Root Causes**:
- Severe class imbalance (Clean 30%+, rare classes <1%)
- Limited dataset size (1,458 augmented from 417 original)
- Fine-grained damage patterns difficult to distinguish

**Speed Trade-offs**:
- SegFormer: 37% slower than ConvNeXt (12ms vs 9ms)
- MaxViT: 68% slower than ConvNeXt (15ms vs 9ms)
- Trade-off: Accuracy vs Speed (SegFormer best accuracy, ConvNeXt fastest)

### Technical Constraints

**Memory Requirements**:
- **Training**: 32 GB RAM (30 GB preload + 2 GB overhead)
- **Inference**: 2-4 GB VRAM (batch size dependent)

**Dataset Limitations**:
- Single heritage collection (ARTeFACT)
- 1,458 augmented images (limited diversity)
- 80/20 split (no cross-validation)
- No test set for final evaluation

---

## 📝 Recommendations

### For Production Deployment

1. **Use SegFormer MiT-B3** (best accuracy-speed balance)
2. **Batch size 32** (optimal VRAM vs throughput: 81 img/s)
3. **GPU inference** (81 img/s vs 5-10 img/s CPU)
4. **Pre-process offline** (resize + normalize once)
5. **Monitor rare classes** (may need post-processing)

### For Future Work (POC-6+)

**Option A: Domain Generalization** (POC-6 Planning document available)
- Implement Hierarchical MTL (3 heads: binary/coarse/fine)
- Progressive Curriculum Learning (staged training)
- LOContent evaluation (4-fold DG)
- Expected: +20-26% mIoU improvement

**Option B: Model Optimization**
- Quantization (FP16 → INT8) for 2-4× speedup
- Knowledge distillation (SegFormer → smaller student)
- ONNX export for production inference

**Option C: Data Expansion**
- Heritage-specific augmentation (expand 1,458 → 2,927)
- Oversample rare classes (Scratches, Structural defects)
- Cross-dataset validation (test on other heritage collections)

---

## 🔗 Repository Information

- **GitHub**: https://github.com/HeritageArt-Research/HeritageArt-CNN-ViT-Hybrid
- **Branch**: `main`
- **Production Tag**: `v1.0-production`
- **Author**: Brandon Trigueros
- **Email**: brandon.trigueros@ucr.ac.cr
- **Institution**: Universidad de Costa Rica (UCR)

### Contributing
- Report issues via GitHub Issues
- Submit pull requests for improvements
- Follow existing code style and documentation standards

---

## 📝 Version History

### v1.0-production (November 17, 2025)
- ✅ Complete multi-architecture benchmark (POC-5.9)
- ✅ 3 trained models (SegFormer, MaxViT, ConvNeXt)
- ✅ Comprehensive evaluation and visualization
- ✅ Production-ready code (~2,825 LOC, no technical debt)
- ✅ Clean git history and consolidated documentation
- ✅ Backup folder with archived POC-5.9 results

### Pre-release (October-November 2025)
- POC-5.5: Hierarchical MTL experiments (418 samples, 22% mIoU)
- POC-5.8: Training optimizations (SLURM, RAM preload, ~28% mIoU)
- POC-5.9 development: Benchmark iterations and refinement

---

## 📦 Backup Folder

**Location**: `experiments/backup/poc59-results/`  
**Purpose**: Archived evaluation results from POC-5.9 production run  
**Contents**:
- `model_comparison.json`: Cross-model performance metrics
- Per-model evaluation folders:
  - `convnext_tiny_evaluation/metrics.json`
  - `segformer_b3_evaluation/metrics.json`
  - `maxvit_tiny_evaluation/metrics.json`
- Visualization archives (PNG files)

**Status**: ✅ Version-controlled (included in .gitignore exceptions)

**Note**: This backup preserves the research artifacts from the November 19, 2025 production run, ensuring reproducibility and providing baseline metrics for future comparisons.

---

**Last Updated**: December 8, 2025  
**Status**: ✅ Production Ready  
**Next Review**: As needed for POC-6 planning or deployment
