# HeritageArt-CNN-ViT-Hybrid: Project Status

**Last Updated**: November 17, 2025  
**Version**: v1.0-production  
**Status**: ✅ Production Ready

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Production Model** | SegFormer-B3 |
| **Best Performance** | 37.63% mIoU (16-class segmentation) |
| **Training Data** | ARTeFACT dataset (augmented) |
| **Total Project Size** | 14.6 GB |
| **Active POCs** | 4 (data + POC-5.5 + POC-5.8 + POC-5.9) |
| **Documentation** | 9 comprehensive documents |

---

## 🎯 Production System (POC-5.9)

**Location**: `experiments/artefact-poc59-multiarch-benchmark/`  
**Size**: 2.5 GB  
**Git Tag**: `v1.0-production`

### Trained Models

| Model | Type | mIoU | Dice | Accuracy | Params |
|-------|------|------|------|----------|--------|
| **SegFormer-B3** ⭐ | Pure ViT | **37.63%** | 46.29% | 88.15% | 47.2M |
| MaxViT-Tiny | CNN-ViT Hybrid | 34.58% | 43.89% | 87.82% | 31.0M |
| ConvNeXt-Tiny | Pure CNN | 25.47% | 35.24% | 86.71% | 28.6M |

### Features
- ✅ Complete training pipeline (50 epochs per model)
- ✅ Comprehensive evaluation (per-class metrics, confusion matrices)
- ✅ Visual validation (9 samples per model, 27 PNG files)
- ✅ Production-ready code (clean, documented, organized)
- ✅ SLURM-optimized (RAM preload, shared memory, parallel training)

### Checkpoints
```
logs/models/
├── convnext_tiny/checkpoint/best_model.pth     (379 MB)
├── maxvit_tiny/checkpoint/best_model.pth       (474 MB)
└── segformer_b3/checkpoint/best_model.pth      (543 MB)
```

---

## 📚 Active Projects

### 1. Data Obtention (POC-3)
**Path**: `experiments/artefact-data-obtention/`  
**Size**: 64 KB  
**Purpose**: Dataset download and preparation scripts  
**Status**: ✅ Utility (always needed)

### 2. POC-5.5: Hierarchical Multiclass
**Path**: `experiments/artefact-poc55-multiclass/`  
**Size**: 865 MB  
**Purpose**: Historical reference for hierarchical multi-task learning  
**Status**: ✅ Docker-only (cleaned, no broken SLURM)

**Key Features:**
- 3-head architecture (Binary → Coarse → Fine)
- Laptop-optimized (256px, RTX 3050 6GB)
- Result: 20.91% mIoU (fine head, 16 classes)

**Why Keep:**
- Unique hierarchical MTL approach
- Educational reference
- Docker workflow example

### 3. POC-5.8: Standard Segmentation
**Path**: `experiments/artefact-poc58-standard/`  
**Size**: 1.5 GB  
**Purpose**: Optimization techniques reference  
**Status**: ✅ Complete (working SLURM infrastructure)

**Key Innovations:**
- Batch size optimization (256 → 128 → 96)
- RAM pre-loading (2x speedup)
- Shared memory pre-loading (2x speedup again)
- SLURM GPU management

**Why Keep:**
- Documents training optimization journey
- Working SLURM reference
- Systematic optimization techniques

### 4. POC-5.9: Multi-Architecture Benchmark (FLAGSHIP)
**Path**: `experiments/artefact-poc59-multiarch-benchmark/`  
**Size**: 2.5 GB  
**Purpose**: Production multi-architecture comparison  
**Status**: ⭐ **PRODUCTION READY**

**Models**: SegFormer-B3, MaxViT-Tiny, ConvNeXt-Tiny  
**Winner**: SegFormer-B3 @ 37.63% mIoU  
**Pipeline**: Complete (train + evaluate + visualize)

---

## 🗑️ Recently Cleaned

### Deleted Projects

**1. POC-5 (artefact-multibackbone-upernet)**
- **Reason**: Never completed, superseded by POC-5.9
- **Size**: 164 KB (no trained models)
- **Commit**: `177465d` (Nov 17, 2025)

**2. POC-5.9-v1 (K-fold experiments)**
- **Reason**: K-fold unnecessary for benchmark
- **Size**: 3.0 GB
- **Commit**: `ba38570` (Nov 17, 2025)

**3. POC-5.5 SLURM Scripts (server/ directory)**
- **Reason**: Broken scripts, POC-5.6/5.7 were learning phase
- **Size**: 12 files
- **Commit**: `177465d` (Nov 17, 2025)

---

## 📖 Documentation Library

### Core Documents

1. **POC_Full_History_Analysis.md** (26 KB)
   - Complete POC evolution (POC-1 through POC-5.9)
   - Architectural decisions and learnings
   - Performance timeline
   - **Note**: POC-5.6/5.7 were SLURM learning experiments (never committed)

2. **POC59_PRODUCTION_REPORT.md** (11 KB)
   - Code quality analysis
   - Production readiness checklist
   - Deployment recommendations

3. **POC55_vs_POC58_Comparison.md** (12 KB)
   - Hierarchical MTL vs Standard segmentation
   - Performance comparison
   - Approach tradeoffs

4. **POC_Evolution_Comparison.md** (20 KB)
   - Cross-POC comparison tables
   - Evolution insights

### Planning Documents

5. **POC-6-PLAN.md** (20 KB)
   - Next steps (if pursuing POC-6)
   - Advanced techniques (domain adaptation, etc.)

6. **POC6-TRAPS-AND-INNOVATIONS.md** (82 KB)
   - Potential pitfalls
   - Innovation opportunities

### Technical References

7. **DATA-AUGMENTATION-STRATEGY.md** (27 KB)
   - Augmentation techniques
   - Heritage-specific transformations

8. **From-Paper-to-Plan.md** (24 KB)
   - Research to implementation pipeline

9. **PROJECT_STATUS.md** (This file)
   - Current status overview
   - Quick reference

---

## 🔬 Evolution Summary

### Performance Growth
```
POC-5.5:  20.91% mIoU (Hierarchical MTL, 16 classes)
   ↓
POC-5.8:  ~28% mIoU* (Standard U-Net, optimizations)
   ↓
POC-5.9:  37.63% mIoU (SegFormer, production)

Total improvement: +79.9% relative (3 weeks)
```
*Estimated (not recorded in git)

### Key Learnings

1. **Architecture**: Standard segmentation > Hierarchical MTL (for this problem)
2. **Infrastructure**: SLURM-only > Docker+SLURM (simpler maintenance)
3. **Optimization**: I/O and memory as important as model architecture
4. **Evaluation**: Metrics + visualizations necessary for production
5. **Model Selection**: Strategic selection (CNN/ViT/Hybrid) > exhaustive search

### POC Timeline

```
POC-1 to POC-3:  Initial setup with MMSegmentation
POC-4:           Minimal training pipeline
POC-5:           Multi-backbone UPerNet (never completed)
POC-5.5:         Hierarchical MTL (20.91% mIoU, Docker-only)
POC-5.6/5.7:     SLURM learning experiments (not committed)
POC-5.8:         Standard segmentation + optimizations
POC-5.9:         Production benchmark (37.63% mIoU) ⭐
```

---

## 🎯 Current Goals

### Achieved ✅
- [x] Multi-architecture comparison framework
- [x] Production-quality training pipeline
- [x] Comprehensive evaluation system
- [x] Visual validation tools
- [x] Clean, organized codebase
- [x] Complete documentation
- [x] Git history cleanup
- [x] Production release tag

### Optional Future Work 📝
- [ ] Domain adaptation techniques (POC-6)
- [ ] Real-world deployment testing
- [ ] Model ensemble experiments
- [ ] API/service wrapper
- [ ] Extended evaluation on external datasets

---

## 🚀 Quick Start (For New Users)

### 1. Clone Repository
```bash
git clone https://github.com/d4v7d/HeritageArt-CNN-ViT-Hybrid.git
cd HeritageArt-CNN-ViT-Hybrid
```

### 2. Use Production Model (POC-5.9)
```bash
cd experiments/artefact-poc59-multiarch-benchmark

# See README for detailed instructions
cat README.md

# Quick inference (example)
python src/evaluate.py \
    --config configs/segformer_b3.yaml \
    --checkpoint logs/models/segformer_b3/checkpoint/best_model.pth \
    --input your_image.jpg \
    --output prediction.png
```

### 3. Explore Documentation
```bash
# Start here
cat documentation/PROJECT_STATUS.md

# Full history
cat documentation/POC_Full_History_Analysis.md

# Production details
cat documentation/POC59_PRODUCTION_REPORT.md
```

---

## 📊 Resource Usage

### Disk Space

| Component | Size | Purpose |
|-----------|------|---------|
| Common Data | 9.7 GB | Shared datasets, augmented data |
| POC-5.9 (Production) | 2.5 GB | 3 models + checkpoints + logs |
| POC-5.8 (Reference) | 1.5 GB | Optimization experiments |
| POC-5.5 (Historical) | 865 MB | Hierarchical MTL reference |
| POC-3 (Utility) | 64 KB | Dataset scripts |
| **Total** | **14.6 GB** | Complete project |

### Compute Resources (POC-5.9)

| Resource | Training | Inference |
|----------|----------|-----------|
| **GPU** | Tesla V100 32GB | RTX 3050 6GB+ |
| **VRAM** | ~8-12 GB | ~2-4 GB |
| **Time** | ~8-10h per model | ~100ms per image |
| **Batch Size** | 8 (train) | 1 (inference) |

---

## 🔗 Repository Links

- **GitHub**: https://github.com/d4v7d/HeritageArt-CNN-ViT-Hybrid
- **Branch**: `main`
- **Latest Commit**: `177465d` (Nov 17, 2025)
- **Production Tag**: `v1.0-production`

---

## 📞 Contact & Contribution

**Author**: Brandon Trigueros  
**Email**: brandon.trigueros@ucr.ac.cr  
**Institution**: Universidad de Costa Rica (UCR)

### Contributing
- Report issues via GitHub Issues
- Submit pull requests for improvements
- Follow existing code style and documentation standards

---

## 📝 Version History

### v1.0-production (Nov 17, 2025)
- ✅ Complete multi-architecture benchmark
- ✅ 3 trained models (SegFormer, MaxViT, ConvNeXt)
- ✅ Comprehensive evaluation and visualization
- ✅ Production-ready code and documentation
- ✅ Clean git history

### Pre-release (Oct-Nov 2025)
- POC-5.5: Hierarchical MTL experiments
- POC-5.8: Training optimizations
- POC-5.9-v1/v2: Benchmark development

---

**Last Updated**: November 17, 2025  
**Status**: ✅ Production Ready  
**Next Review**: As needed for POC-6 planning
