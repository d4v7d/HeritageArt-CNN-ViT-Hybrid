# POC-60: Hierarchical Multi-Task Learning (HMTL Only)

**Date**: November 18, 2025  
**Status**: 🟡 **IN PROGRESS** - Training Started (Job 2230)  
**Innovation**: #1 Hierarchical MTL (Binary + Coarse + Fine heads)  
**Base**: POC-59 Production (37.63% mIoU SegFormer baseline)  
**Strategy**: Sequential validation → POC-60 (HMTL) → POC-61 (+Curriculum)

---

## 🎯 Objective

Validate **Innovation #1: Hierarchical Multi-Task Learning** in isolation before adding Progressive Curriculum Learning (POC-61).

**Why Sequential Approach?**
- ✅ Isolate each innovation for proper ablation study
- ✅ Better debugging (know exactly what causes improvements/failures)
- ✅ No throwaway code (POC-60 becomes foundation for POC-61)
- ✅ Scientific rigor: measure HMTL contribution independently

**Expected Improvement:** +7-13% mIoU over POC-59

| Model | POC-59 Baseline | POC-60 Target | Improvement |
|-------|-----------------|---------------|-------------|
| ConvNeXt | 25.47% | **33-36%** | +7-11% |
| SegFormer | 37.63% | **45-50%** �� | +7-13% |
| MaxViT | 34.58% | **42-45%** | +7-11% |

---

## 🏗️ Architecture: Hierarchical UPerNet

### Innovation: 3-Head Multi-Task Learning

```
┌─────────────────────────────────────────────────┐
│ Encoder (ConvNeXt/SegFormer/MaxViT)             │
│   ↓ Multi-scale features [C2, C3, C4, C5]       │
│ ┌─────────────────────────────────────┐         │
│ │ UPerNet Decoder (PPM + FPN)         │         │
│ │  - Pyramid Pooling Module (PPM)     │         │
│ │  - Feature Pyramid Network (FPN)    │         │
│ └─────────────────────────────────────┘         │
│   ↓ Fused multi-scale context                   │
│ ┌──────────┬──────────┬──────────┐              │
│ │ Binary   │ Coarse   │  Fine    │              │
│ │ Head     │ Head     │  Head    │              │
│ │ (2 cls)  │ (4 cls)  │ (16 cls) │              │
│ │          │          │          │              │
│ │ Clean vs │ 4 Damage │ Full     │              │
│ │ Damage   │ Groups   │ Classes  │              │
│ └──────────┴──────────┴──────────┘              │
└─────────────────────────────────────────────────┘

Hierarchical Loss = 0.2 * L_binary + 0.3 * L_coarse + 1.0 * L_fine
```

**Why This Helps:**
- **Binary head**: Easy task (Clean vs Damage) provides strong gradient signal
- **Coarse head**: Groups rare classes together, easier to learn
- **Fine head**: Benefits from auxiliary supervision from binary/coarse
- **Result**: Better performance on rare classes (Lightleak, Burn marks, Scratches)

### Class Hierarchy

**Binary Head** (2 classes):
- 0: Clean (background)
- 1: Damage (any type)

**Coarse Head** (4 groups):
- 0: Structural (Cracks, Material loss, Peel, Structural defects)
- 1: Surface (Dirt spots, Stains, Hairs, Dust spots)
- 2: Color (Discolouration, Burn marks, Fading)
- 3: Optical (Scratches, Lightleak, Blur)

**Fine Head** (16 classes):
- Full ARTeFACT taxonomy (0=Clean, 1-15=damage types)

### Key Differences from POC-59

| Aspect | POC-59 (Baseline) | POC-60 (Hierarchical) |
|--------|-------------------|----------------------|
| **Decoder** | UNet | **UPerNet (PPM + FPN)** |
| **Heads** | 1 (fine only) | **3 (binary + coarse + fine)** |
| **Loss** | DiceFocalLoss | **Hierarchical weighted loss** |
| **Epochs** | 50 | **100** (more complex model) |
| **Batch Size** | 32 (SegFormer) | **32** (same) |
| **Innovation** | None | **Hierarchical MTL** |
| **Expected mIoU** | 37.63% | **45-50%** (+7-13%) |

---

## 🚀 Quick Start

### 1. Full Training (100 epochs, ~1.4h per model on V100)

```bash
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc60-hierarchical-mtl

# Train single model
sbatch scripts/slurm_train.sh configs/hierarchical_segformer.yaml

# Train all 3 models sequentially (recommended)
JOB1=$(sbatch --parsable scripts/slurm_train.sh configs/hierarchical_convnext.yaml)
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 scripts/slurm_train.sh configs/hierarchical_segformer.yaml)
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 scripts/slurm_train.sh configs/hierarchical_maxvit.yaml)

# Check queue
squeue -u $USER
```

**Expected per model:**
- Preload time: ~30 sec
- Training time: ~1.4 hours (100 epochs)
- Total for 3 models: **~4.2 hours**

### 2. Monitor Progress

```bash
# Check current epoch and metrics
tail -50 logs/train_XXXX.out

# Watch live (updates every 10s)
watch -n 10 'tail -50 logs/train_XXXX.out'

# Check GPU usage
srun --jobid=XXXX nvidia-smi
```

**Key Metrics to Watch:**
- `Binary mIoU`: Should reach ~70-76% (easy task)
- `Coarse mIoU`: Should reach ~52-62% (medium difficulty)
- `Fine mIoU`: Should reach ~45-50% (main target) 🎯
- `Total Loss`: Should decrease from ~1.2 → ~0.4

### 3. Check Results

```bash
# After training completes
ls -lh logs/models/HierarchicalUPerNet_*/

# View final metrics
cat logs/models/HierarchicalUPerNet_*/best_metrics.txt
```

---

## 📊 Expected Results

### SegFormer-B3 (Best Model - Target)

| Metric | POC-59 | POC-60 Expected | Improvement |
|--------|--------|-----------------|-------------|
| **Fine mIoU** | 37.63% | **45-50%** 🎯 | +7-13% |
| Binary mIoU | N/A | **72-76%** | New |
| Coarse mIoU | N/A | **58-62%** | New |
| Training Time | 42 min | **~84 min** | 2× (100 epochs) |

**Per-Class IoU Expectations:**
- Clean: 95% (maintained from POC-59)
- Material Loss: 81% → **85%** (+4%)
- Peel: 66% → **72%** (+6%)
- Cracks: 48% → **56%** (+8%)
- **Rare classes** (Lightleak, Burn, Scratch): **15-25%** (vs 5-12% in POC-59)

### ConvNeXt & MaxViT

| Model | POC-59 | POC-60 Target | Improvement |
|-------|--------|---------------|-------------|
| ConvNeXt | 25.47% | **33-36%** | +7-11% |
| MaxViT | 34.58% | **42-45%** | +7-11% |

---

## ⚙️ Training Configuration

```yaml
# configs/hierarchical_segformer.yaml
model:
  hierarchical: true
  encoder: nvidia/segformer-b3-finetuned-ade-512-512
  decoder: upernet
  num_classes: 16

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.000333
  weight_decay: 0.01
  mixed_precision: true

loss:
  type: hierarchical
  weights:
    binary: 0.2
    coarse: 0.3
    fine: 1.0
  dice_weight: 0.5
  focal_weight: 0.5

optimizer:
  type: adamw

scheduler:
  type: onecycle
  max_lr: 0.000333
  pct_start: 0.3
```

---

## 📁 Directory Structure

```
artefact-poc60-hierarchical-mtl/
├── README.md                       # This file
├── requirements.txt                # Same as POC-59
├── configs/
│   ├── hierarchical_convnext.yaml  # CNN encoder
│   ├── hierarchical_segformer.yaml # ViT encoder (BEST)
│   └── hierarchical_maxvit.yaml    # Hybrid encoder
├── scripts/
│   ├── slurm_train.sh              # Updated for POC-60
│   ├── slurm_evaluate.sh           # Hierarchical evaluation
│   └── slurm_visualize.sh          # Visualization job
├── src/
│   ├── train.py                    # ✅ Updated for hierarchical
│   ├── evaluate.py                 # From POC-59
│   ├── visualize.py                # From POC-59
│   ├── dataset.py                  # ✅ Added label conversion
│   ├── preload_dataset.py          # From POC-59
│   ├── losses.py                   # From POC-59
│   ├── model_factory.py            # ✅ Updated for hierarchical
│   ├── timm_encoder.py             # From POC-59
│   ├── hierarchical_upernet.py     # ✅ New (from POC-55)
│   ├── upernet_custom.py           # ✅ New (PPM + FPN)
│   ├── hierarchical_losses.py      # ✅ New (3-head loss)
│   └── hierarchical_utils.py       # ✅ New (metrics helpers)
└── logs/
    ├── train_XXXX.out              # SLURM output logs
    ├── train_XXXX.err              # SLURM error logs
    └── models/                     # Checkpoints (pending)
```

---

## 🔬 Code Changes from POC-59

### New Files (from POC-55)

1. **src/hierarchical_upernet.py** (530 LOC)
   - 3-head architecture with UPerNet decoder
   - Handles transformer feature transposition
   - PPM + FPN for multi-scale context

2. **src/upernet_custom.py** (300 LOC)
   - Pyramid Pooling Module (PPM)
   - Feature Pyramid Network (FPN)
   - Multi-scale feature aggregation

3. **src/hierarchical_losses.py** (400 LOC)
   - HierarchicalDiceFocalLoss
   - Weighted combination of 3 heads
   - Returns dict with per-head losses

4. **src/hierarchical_utils.py** (150 LOC)
   - Helper functions for hierarchical training
   - Metrics computation (3-head mIoU)
   - Label conversion utilities

### Modified Files

1. **src/model_factory.py**
   - Added hierarchical flag detection
   - Creates HierarchicalUPerNet if `hierarchical=true`
   - Falls back to UNet for standard models

2. **src/dataset.py**
   - Added `fine_to_binary()` function
   - Added `fine_to_coarse()` function
   - Label conversion handles ignore_index=255

3. **src/train.py**
   - Updated to handle dict outputs from hierarchical model
   - Computes hierarchical loss
   - Logs 3-head metrics (binary, coarse, fine)
   - Detects is_hierarchical from config

---

## ✅ Success Criteria

**Minimum (GO for POC-61):**
- ✅ Fine mIoU ≥ 40% (SegFormer)
- ✅ Binary mIoU ≥ 70%
- ✅ Coarse mIoU ≥ 50%
- ✅ Training stable, no NaN losses

**Target (Paper-worthy):**
- 🎯 Fine mIoU ≥ 45% (+7-8% vs POC-59)
- 🎯 Rare class IoU improvement ≥ +5%
- 🎯 Binary/Coarse heads converge properly

**Stretch (Exceptional):**
- 🚀 Fine mIoU ≥ 50% (+12-13% vs POC-59)
- 🚀 All 3 models show consistent improvement
- 🚀 Coarse mIoU ≥ 60%

---

## 📝 Current Status

**Job 2230** (SegFormer-B3 Hierarchical):
- Status: ✅ Running on V100
- Started: Nov 18, 2025
- Expected completion: ~1.4 hours
- Current phase: Pre-loading dataset to RAM

**Next Steps:**
1. ⏳ Wait for Job 2230 to complete (~1.4h)
2. ⏳ Analyze results (binary, coarse, fine metrics)
3. ⏳ Train ConvNeXt and MaxViT (if SegFormer succeeds)
4. ⏳ Evaluate all 3 models
5. ⏳ Compare vs POC-59 baseline
6. ⏳ Decide: Proceed to POC-61 or iterate POC-60

---

## 🚀 Next: POC-61 (Progressive Curriculum)

**If POC-60 succeeds** (mIoU ≥ 40%):
- Copy POC-60 to POC-61
- Add Progressive Curriculum Learning:
  - **Stage 1** (20 epochs): Binary head only
  - **Stage 2** (20 epochs): Binary + Coarse heads
  - **Stage 3** (60 epochs): All 3 heads
- Expected additional gain: +4-6% mIoU
- **Total improvement**: POC-59 (37.63%) → POC-61 (50-56%)

---

*Last updated: November 18, 2025*  
*Status: 🟡 In Progress - Training Job 2230*  
*Next milestone: Job 2230 completion + results analysis*
