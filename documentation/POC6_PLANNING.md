# POC-6: Multiclass Segmentation + Domain Generalization

**Date**: November 17, 2025  
**Status**: 🟢 READY TO EXECUTE (Conservative Approach)  
**Base**: POC-5.9 Production (37.63% mIoU, SegFormer-B3)  
**Target**: Conference Workshop Paper

---

## 🎯 Executive Summary

**Objective**: Answer RQ1 (CNN vs ViT vs Hybrid on multiclass) and RQ2 (Domain Generalization) using a pragmatic approach that validates core innovations with existing data before expanding.

**Strategy**: Use existing 1,458 augmented samples, implement proven innovations (#1 Hierarchical MTL + #5 Progressive Curriculum), evaluate DG with LOContent-only (most robust split).

**Timeline**: 5 days (4 days code + 1 day GPU = 21h)  
**Expected Impact**: +20-26% mIoU improvement (37.63% → 45-50%)  
**Publication Target**: Conference workshop (CVPRW, ICCVW, BMVC)

---

## 📊 Research Questions

### **RQ1**: Which model family (CNN/ViT/Hybrid) best detects and classifies painting deterioration across damage types?

**Metrics**: 
- mIoU (mean IoU across 16 classes)
- macro-F1 (equal weight per class)
- Per-class IoU/F1 for each of 15 damage types + Clean

**Approach**:
1. Train ConvNeXt-Tiny, SegFormer-B3, MaxViT-Tiny on multiclass (16 classes)
2. Implement Hierarchical Multi-Task Learning (3 heads: binary, coarse, fine)
3. Use Progressive Curriculum (binary → coarse → fine stages)
4. Report in-domain performance on held-out test set

### **RQ2**: Which family generalizes better to unseen cultural heritage collections?

**Metrics**:
- DG gap = in-domain mIoU - OOD mIoU
- Bootstrap 95% CIs for statistical significance

**Approach**:
1. **LOContent** (Leave-One-Content-Out): 4 content types as domains
   - Artistic depiction, Photographic, Line art, Geometric patterns
   - ~365 samples/content (3.6x over minimum threshold = robust)
2. Measure OOD drop for each architecture
3. Compare which architecture maintains performance best

---

## 📊 Current State Analysis

### **POC-5.9 Production Baseline**
```
Best Model: SegFormer-B3
- mIoU: 37.63%
- Top-3 classes: Clean (95%), Material Loss (81%), Peel (66%)
- Weak classes: Scratches (23%), Structural defects (6%)
- Training: 50 epochs, 384px, batch 32, AMP
- Dataset: 1,458 augmented (1,166 train / 292 val)
- Infrastructure: SLURM optimized, V100 32GB, reproducible
```

### **Dataset Reality Check**

**POC-6 Original Plan Assumption**: ~11,000 samples  
**Actual Available**: 1,458 augmented samples (87% less)

**Impact on DG Evaluation**:

| DG Method | Samples per Fold | Status | Recommendation |
|-----------|------------------|--------|----------------|
| **LOMO (10 materials)** | ~146/material | 🟡 Marginal | ⚠️ Skip in POC-6.1 |
| **LOContent (4 types)** | ~365/content | ✅ Robust (3.6x) | ✅ Primary approach |

**Decision**: Focus on LOContent for POC-6.1 (robust evaluation), defer LOMO to POC-6.2 if expanding dataset.

---

## 🚨 Critical Implementation Traps & Solutions

### Trap 1: Class Imbalance
**Issue**: Severe class imbalance (frequent classes 30%+, rare classes <1%)

**Impact**: Naive training learns only frequent classes, rare class IoU ≈ 0%

**Solution**: Hierarchical Multi-Task Learning (Innovation #1)
- Binary head: Clean vs Damage (easy, high performance)
- Coarse head: 4 damage groups (moderate difficulty)
- Fine head: 16 classes (hard, guided by coarse)
- Coarse head provides learning signal for rare classes

### Trap 2: Training Time Underestimation
**Issue**: Multiclass is 3-5x slower than binary

**Original Estimate**: 6-8h per model  
**Reality with 1,458 samples**: 140h per model (60 epochs, AMP, 384px)

**Solution**: 
- Reduce epochs: 100 → 60 (POC-5.9 showed sufficient)
- Aggressive early stopping: patience=10
- Mixed precision AMP: 30-40% speedup
- Resolution 384px (vs 512px): 44% compute savings

### Trap 3: Multiclass Difficulty
**Issue**: Expected "50-55% mIoU" too optimistic without innovations

**Expected Naive Performance**:
- ConvNeXt-Tiny: 25-30% mIoU (CNN struggles with rare classes)
- SegFormer-B3: 35-40% mIoU (better global context)
- MaxViT-Tiny: 32-37% mIoU (hybrid advantage)

**Solution**: Combined innovations
- Hierarchical MTL: +5-8% mIoU
- Progressive Curriculum: +4-6% mIoU
- **Target with innovations**: 45-50% mIoU achievable

---

## 💡 Innovations (Included in POC-6.1)

### Innovation #1: Hierarchical Multi-Task Learning ⭐⭐⭐⭐⭐

**Status**: ✅ Already validated in POC-5.5 (22% mIoU with 418 samples)

**Architecture**:
```
Encoder (ConvNeXt/SegFormer/MaxViT)
   ↓
UPerNet Neck (PPM + FPN)
   ↓
┌─────────┬─────────┬─────────┐
│ Binary  │ Coarse  │  Fine   │
│ Head    │ Head    │  Head   │
│ (2 cls) │ (4 cls) │ (16 cls)│
└─────────┴─────────┴─────────┘

Loss = 1.0 * L_fine + 0.3 * L_coarse + 0.2 * L_binary
```

**Class Grouping (Coarse Head)**:
1. **Structural Damage**: Cracks, Material loss, Peel, Structural defects
2. **Surface Contamination**: Dirt spots, Stains, Hairs, Dust spots
3. **Color Alterations**: Discolouration, Burn marks, Fading
4. **Optical Artifacts**: Scratches, Lightleak, Blur

**Benefits**:
- ✅ Easier learning cascade: binary 70% → coarse 60% → fine 50%
- ✅ Rare classes benefit from coarse-level guidance
- ✅ +5-8% mIoU improvement (vs +3-4% with 418 samples)
- ✅ Code already exists in POC-5.5 (530 LOC)

**Implementation Effort**: 2 days (adapt POC-5.5 to POC-5.9 structure)

---

### Innovation #5: Progressive Curriculum Learning ⭐⭐⭐⭐⭐

**Status**: New implementation (validated concept)

**Training Stages**:
```
Stage 1: Binary (20 epochs)
- Task: Clean vs Damage
- Head: Binary only (freeze coarse + fine)
- LR: 1e-5 (conservative start)
- Goal: Learn "what is damage?"

Stage 2: Coarse (20 epochs)
- Task: 4 damage groups
- Heads: Binary + Coarse (freeze fine)
- LR: 5e-5 (medium)
- Transfer: Load Stage 1 checkpoint
- Goal: Learn "what type of damage?"

Stage 3: Fine (60 epochs)
- Task: 16 classes full
- Heads: All 3 (binary + coarse + fine)
- LR: 1e-4 (full learning)
- Transfer: Load Stage 2 checkpoint
- Goal: Learn "precise damage class"
```

**Benefits**:
- ✅ Free in GPU time: 100 epochs (20+20+60) = same as 100 direct
- ✅ +4-6% mIoU from warm start
- ✅ More stable: avoids early overfitting on rare classes
- ✅ Low effort: 1 day of code (100 LOC staging logic)

**Implementation Effort**: 1 day (modify train.py)

---

### Domain Generalization: LOContent-Only

**Approach**: Leave-One-Content-Out (4 splits)

**Content Types**:
1. Artistic depiction (~365 samples)
2. Photographic depiction (~365 samples)
3. Line art (~365 samples)
4. Geometric patterns (~365 samples)

**Training Protocol**:
- For each content type:
  - Train: 3 other contents (~1,093 samples)
  - Test: Held-out content (~365 samples)
  - Repeat: 4 folds × 3 models = **12 training runs**

**Expected Results**:
- In-domain mIoU: 45-50% (with innovations)
- OOD mIoU: 30-40% (LOContent held-out)
- **DG Gap**: 15-20%

**GPU Time**: 4 folds × 3 models × 3.5h = 42h training  
(Baseline training: 21h, DG evaluation: +21h)

---

## 📅 Implementation Timeline (5 Days)

### **Day 1-2: Hierarchical Multi-Task Learning** (2 days)

**Tasks**:
1. Copy POC-5.5 hierarchical code to POC-5.9
   - `hierarchical_upernet.py` (530 LOC)
   - `hierarchical_loss.py` (400 LOC)

2. Adapt to POC-5.9 structure:
   - Integrate with `src/model_factory.py`
   - Update configs with hierarchical parameters
   - Test 1 epoch (verify VRAM < 8GB, output shapes correct)

3. Create ground truth labels:
   - Binary: Clean (0) vs Damage (1-15 → 1)
   - Coarse: 4 damage groups (map 16 classes → 4 groups)
   - Fine: 16 classes (original)

**Deliverables**:
- ✅ `src/models/hierarchical_upernet.py`
- ✅ `src/losses/hierarchical_loss.py`
- ✅ Updated configs: `configs/hierarchical_*.yaml`
- ✅ Test run validated (1 epoch, 3 models)

**Verification**:
```bash
sbatch scripts/slurm_test_hierarchical.sh configs/hierarchical_segformer_b3.yaml

# Expected output:
# - train_loss_binary, train_loss_coarse, train_loss_fine
# - val_miou_binary, val_miou_coarse, val_miou_fine
# - VRAM: ~2.5 GB (vs 2.3 GB baseline, acceptable)
```

---

### **Day 3: Progressive Curriculum Implementation** (1 day)

**Tasks**:
1. Modify `src/train.py` with staging logic:
   - Epochs 1-20: Binary head only (freeze coarse + fine)
   - Epochs 21-40: Binary + Coarse heads (freeze fine)
   - Epochs 41-100: All 3 heads active

2. Checkpoint transfer logic:
   - Stage 1 → Stage 2: Transfer encoder + binary head weights
   - Stage 2 → Stage 3: Transfer encoder + binary + coarse weights

3. Per-stage metrics logging:
   - TensorBoard: separate sections for each stage
   - CSV: stage indicator column

**Deliverables**:
- ✅ `src/train_curriculum.py` (or modify train.py)
- ✅ Updated configs with staging parameters
- ✅ Test run validated (curriculum stages working)

**Verification**:
```bash
# Test curriculum (10 epochs: 4 binary + 3 coarse + 3 fine)
sbatch scripts/slurm_test_curriculum.sh configs/curriculum_test.yaml

# Expected behavior:
# - Epochs 1-4: Only binary loss decreases
# - Epochs 5-7: Binary + coarse losses decrease
# - Epochs 8-10: All 3 losses decrease
```

---

### **Day 4: LOContent Splits Creation** (1 day)

**Tasks**:
1. Create DG split generation script:
   - Read ARTeFACT metadata (content type per image)
   - Generate 4 manifests (one per held-out content)
   - Verify balance: ~365 samples/content

2. Verify split quality:
   - Check all 16 classes present in train split
   - Check no data leakage (unique images per fold)
   - Analyze class distribution per fold

3. Create training script for LOContent:
   - Loop over 4 content types
   - Train 3 models per fold
   - Log DG metrics separately

**Deliverables**:
- ✅ `scripts/create_locontent_splits.py`
- ✅ `manifests/locontent_fold{1-4}.json` (4 files)
- ✅ `scripts/train_locontent.sh`
- ✅ Split analysis report

**Verification**:
```bash
python scripts/create_locontent_splits.py
python scripts/analyze_splits.py manifests/locontent_*.json

# Expected output:
# - Fold 1 (artistic held-out): 1,093 train / 365 test
# - Fold 2 (photographic): 1,093 train / 365 test
# - Fold 3 (line_art): 1,093 train / 365 test
# - Fold 4 (geometric): 1,093 train / 365 test
# - All folds have all 16 classes in train
```

---

### **Day 5: Training Execution (GPU Day)**

**Morning: Baseline Training** (3 models × 100 epochs = 21h)
```bash
# Train all 3 models with hierarchical + curriculum
sbatch scripts/train_hierarchical.sh configs/hierarchical_convnext.yaml
sbatch scripts/train_hierarchical.sh configs/hierarchical_segformer.yaml
sbatch scripts/train_hierarchical.sh configs/hierarchical_maxvit.yaml

# Expected completion: 21h (7h per model)
```

**Evening: Evaluate Baseline** (3 models × 5 min = 15 min)
```bash
python src/evaluate.py --all
python src/visualize.py --all

# Expected output:
# - Baseline mIoU: 45-50% (with innovations)
# - Per-model comparison table
# - 27 visualization PNGs
```

**Next Day: LOContent DG Training** (12 runs × 3.5h = 42h)
```bash
# Train LOContent folds
sbatch scripts/train_locontent.sh

# Expected completion: 42h total
# Expected results:
# - In-domain: 45-50% mIoU
# - OOD: 30-40% mIoU
# - DG Gap: 15-20%
```

---

## 📊 Expected Results

### Baseline Performance (In-Domain)

| Model | POC-5.9 (Naive) | POC-6.1 (Hierarchical + Curriculum) | Improvement |
|-------|-----------------|-------------------------------------|-------------|
| ConvNeXt-Tiny | 25.47% | 32-35% | +25-37% |
| SegFormer-B3 | 37.63% | 45-50% | +20-33% |
| MaxViT-Tiny | 34.58% | 42-47% | +21-36% |

### Domain Generalization (LOContent)

| Model | In-Domain mIoU | OOD mIoU | DG Gap | Winner |
|-------|----------------|----------|--------|--------|
| ConvNeXt-Tiny | 32-35% | 22-28% | ~10% | ✅ Best DG |
| SegFormer-B3 | 45-50% | 30-40% | ~15% | ⚠️ Moderate |
| MaxViT-Tiny | 42-47% | 28-37% | ~12% | ✅ Good DG |

**Hypothesis**: CNNs and Hybrids generalize better than pure ViTs (local inductive biases help OOD)

---

## 🚫 Excluded from POC-6.1 (Saved for POC-6.2)

### ❌ Innovation #4: Damage-Aware Attention
**Reason**: 12-58 samples/rare class marginal for prototype learning  
**Alternative**: Hierarchical MTL coarse head already addresses rare classes  
**Saves**: 2 days development

### ❌ Innovation #6: Heritage Augmentation Expansion (1,458 → 2,927)
**Reason**: Current 1,458 samples sufficient for LOContent (365/content robust)  
**Alternative**: Validate baseline first, expand only if DG gap >20%  
**Saves**: 1 day dev + 3h CPU regen + validation time

### ❌ Innovation #2: MAE Self-Supervised Pretraining
**Reason**: ROI moderate with 1,458 samples (+4-6% vs +10-15% with 11k)  
**Alternative**: Not critical for workshop paper  
**Saves**: +150h GPU time (3 encoders × 50 epochs pretraining)

### ❌ Innovation #3: MAML Meta-Learning
**Reason**: Complexity high (400 LOC), LOContent doesn't benefit (only 4 tasks)  
**Alternative**: Save for LOMO 10-way if dataset expands  
**Saves**: 5 days dev + 143.5h GPU

### ❌ LOMO 10-way DG
**Reason**: 146 samples/material marginal (high variance, unreliable)  
**Alternative**: LOContent more robust (365/content, 3.6x margin)  
**Saves**: +42h GPU time (30 runs vs 12 runs)

---

## 🎯 Success Criteria

### Must Have (Minimum Viable)
- ✅ Hierarchical MTL implemented and working
- ✅ Progressive Curriculum implemented and working
- ✅ Baseline mIoU: 45-50% (RQ1 answered)
- ✅ LOContent DG evaluation complete (RQ2 answered)
- ✅ Per-class metrics for all 16 classes
- ✅ Winner identified (CNN/ViT/Hybrid for in-domain + DG)

### Nice to Have (Extended Analysis)
- 📊 Confusion matrices per model
- 📊 Per-stage learning curves (binary → coarse → fine)
- 📊 DG gap analysis per content type
- 📊 Statistical significance tests (bootstrap 95% CI)

### Future Work (POC-6.2)
- 🔬 LOMO evaluation (if dataset expanded to 2,927)
- 🔬 MAE pretraining (if targeting conference main track)
- 🔬 Damage-Aware Attention (if rare class IoU still <5%)
- 🔬 Heritage-specific augmentation validation

---

## 📝 Publication Strategy

### Target: Conference Workshop

**Venues**:
- CVPR Workshop on Cultural Heritage
- ICCV Workshop on AI for Digital Heritage
- BMVC Workshop on Document Analysis

**Contributions**:
1. **Hierarchical MTL for imbalanced heritage segmentation** (Novel)
2. **Progressive Curriculum for multiclass damage detection** (Novel)
3. **Multi-architecture DG benchmark** (ARTeFACT dataset) (Useful)
4. **Empirical insights**: CNN/Hybrid > ViT for heritage DG (Interesting)

**Paper Structure** (6 pages + references):
- Abstract (150 words)
- Introduction (1 page): Motivation, RQs
- Related Work (0.5 page): Segmentation, DG, Heritage
- Method (2 pages): Hierarchical MTL, Curriculum, DG protocol
- Experiments (1.5 pages): Results tables, ablations
- Conclusion (0.5 page): Insights, future work
- References (1 page)

**Estimated Timeline**:
- POC-6.1 execution: 1 week (this plan)
- Paper writing: 2 weeks
- Internal review: 1 week
- Submission: Week 4
- **Total**: 1 month from start to submission

---

## ✅ Risk Mitigation

### Technical Risks

**Risk**: Hierarchical MTL doesn't improve over baseline  
**Mitigation**: POC-5.5 already validates +3-4% mIoU with 418 samples  
**Probability**: Low (10%)

**Risk**: Progressive Curriculum overfits early stages  
**Mitigation**: Transfer only encoder weights, fine-tune heads  
**Probability**: Low (15%)

**Risk**: LOContent DG gap too high (>30%)  
**Mitigation**: Expected for heritage domain, report as-is, propose innovations for POC-6.2  
**Probability**: Medium (40%)

### Schedule Risks

**Risk**: GPU unavailable for 21h continuous run  
**Mitigation**: Split into 3 sequential 7h runs, queue efficiently  
**Probability**: Medium (30%)

**Risk**: Code bugs delay implementation  
**Mitigation**: POC-5.5 code already tested, adaptation should be straightforward  
**Probability**: Low (20%)

**Risk**: Dataset splits unbalanced  
**Mitigation**: Verify splits before training, script generates balanced folds  
**Probability**: Very Low (5%)

---

## 📦 Deliverables

### Code
- ✅ `src/models/hierarchical_upernet.py` (Hierarchical architecture)
- ✅ `src/losses/hierarchical_loss.py` (Multi-task loss)
- ✅ `src/train_curriculum.py` (Progressive training)
- ✅ `scripts/create_locontent_splits.py` (DG splits)
- ✅ `scripts/train_locontent.sh` (DG training loop)
- ✅ Updated configs (hierarchical parameters)

### Results
- ✅ Baseline metrics: `logs/baseline/model_comparison.json`
- ✅ LOContent metrics: `logs/locontent/dg_results.json`
- ✅ Visualizations: 27 PNGs (9 per model)
- ✅ Confusion matrices: 3 matrices (1 per model)
- ✅ Per-class IoU bar charts: 3 charts

### Documentation
- ✅ POC-6.1 execution report (this document)
- ✅ Updated README.md (POC-6 section)
- ✅ Training logs analysis (best epochs, convergence)
- ✅ DG gap analysis (per content type, per model)

---

**Document Status**: ✅ Ready for Execution  
**Last Updated**: November 17, 2025  
**Next Action**: Begin Day 1 implementation (Hierarchical MTL adaptation)
