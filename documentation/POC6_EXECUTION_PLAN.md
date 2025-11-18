# POC-6.1 LITE: Execution Plan (Conservative Approach)

**Date**: November 17, 2025  
**Status**: 🟢 READY TO EXECUTE  
**Base**: POC-5.9 Production (37.63% mIoU, SegFormer-B3)  
**Target**: Conference Workshop Paper

---

## 🎯 Executive Summary

**Objective**: Respond to RQ2 (Domain Generalization) with a pragmatic, conservative approach that validates core innovations before expanding.

**Strategy**: Use existing 1,458 augmented dataset, implement proven innovations (#1 Hierarchical MTL + #5 Progressive Curriculum), evaluate DG with LOContent-only (most robust split).

**Timeline**: 5 días (4 días código + 1 día GPU)  
**Expected Impact**: +20-26% mIoU improvement (37.63% → 45-50%)  
**Publication Target**: Conference workshop (CVPRW, ICCVW, BMVC)

---

## 📊 Current State Analysis

### **POC-5.9 Baseline (Production)**
```
Best Model: SegFormer-B3
- mIoU: 37.63%
- Top-3 classes: Clean (95%), Material Loss (81%), Peel (66%)
- Weak classes: Scratch (23%), Structural defects (6%)
- Training: 50 epochs, 384px, batch 32, AMP
- Dataset: 1,458 augmented (1,166 train / 292 val)
- Augmentation: Basic (HFlip, VFlip, Rotate90)
```

### **Available Resources**
- ✅ **Dataset**: 1,458 samples augmented (9.7 GB, in git)
- ✅ **Infrastructure**: SLURM scripts, V100 32GB, reproducible pipeline
- ✅ **Validated code**: POC-5.5 Hierarchical MTL (22% mIoU with 418 samples)
- ✅ **3 architectures**: ConvNeXt (CNN), SegFormer (ViT), MaxViT (Hybrid)

---

## 🎯 POC-6.1 LITE Scope

### **Core Innovations (MUST-HAVE)**

#### **Innovation #1: Hierarchical Multi-Task Learning** ⭐⭐⭐⭐⭐
**Status**: ✅ Already validated in POC-5.5

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

Loss = 0.2 * L_binary + 0.3 * L_coarse + 1.0 * L_fine
```

**Class Grouping (Coarse Head)**:
1. **Structural Damage**: Cracks, Material loss, Peel, Structural defects
2. **Surface Contamination**: Dirt spots, Stains, Hairs, Dust spots
3. **Color Alterations**: Discolouration, Burn marks, Fading
4. **Optical Artifacts**: Scratches, Lightleak, Blur

**Expected Benefit**: +5-8% mIoU (vs +3-4% with 418 samples in POC-5.5)

**Implementation**:
- Source: `experiments/artefact-poc55-multiclass/scripts/models/hierarchical_upernet.py`
- Adaptation needed: Replace SMP Unet → HierarchicalUPerNet
- Effort: 2 días (already implemented, only adapt to POC-5.9 structure)

---

#### **Innovation #5: Progressive Curriculum Learning** ⭐⭐⭐⭐⭐
**Status**: New implementation (validated concept)

**Training Stages**:
```
Stage 1: Binary (20 epochs)
- Task: Clean vs Damage (2 classes)
- Head: Binary head only (freeze coarse + fine)
- LR: 1e-5 (conservative start)
- Goal: Learn "what is damage?"

Stage 2: Coarse (20 epochs)
- Task: 4 damage groups
- Heads: Binary + Coarse (freeze fine)
- LR: 5e-5 (medium)
- Transfer: Load Stage 1 checkpoint
- Goal: Learn "what type of damage?"

Stage 3: Fine (60 epochs)
- Task: 16 classes (full multiclass)
- Heads: All 3 (binary + coarse + fine)
- LR: 1e-4 (full learning)
- Transfer: Load Stage 2 checkpoint
- Goal: Learn "precise damage class"
```

**Expected Benefit**: +4-6% mIoU, more stable convergence

**Implementation**:
- Modify: `experiments/artefact-poc59-multiarch-benchmark/src/train.py`
- Add: Staging logic (100 LOC)
- Effort: 1 día

---

### **Domain Generalization Evaluation**

#### **LOContent-Only (4 splits)** ✅ ROBUST
**Rationale**: Most robust DG split with 1,458 samples

**Math**:
```
1,458 samples ÷ 4 contents = ~365 samples/content
Mínimo recomendado: 100 samples/content
Margin: 3.6x sobre mínimo ✅ VERY ROBUST
```

**Content Types** (ARTeFACT metadata):
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
- In-domain mIoU: 45-50% (baseline with innovations)
- OOD mIoU: 30-40% (LOContent held-out)
- **DG Gap**: 15-20%

---

### **Dataset Strategy**

#### **Use Existing 1,458 Augmented** ✅ NO REGENERATE
**Rationale**: POC-5.9 already validates this works (37.63% mIoU)

**Current Augmentation** (from POC-5.9):
```python
# src/dataset.py
A.Resize(384, 384)
A.HorizontalFlip(p=0.5)
A.VerticalFlip(p=0.3)
A.RandomRotate90(p=0.3)
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Sufficient for**:
- ✅ Baseline training (no overfitting at 37.63% mIoU)
- ✅ LOContent DG (~365 samples/content, robust)
- ✅ Hierarchical MTL (already validated with 418 → works better with 1,458)

**NOT sufficient for**:
- ❌ LOMO 10-way (~146 samples/material, marginal)
- ❌ Rare class oversampling (will address in POC-6.2 if needed)

---

### **EXCLUDED from POC-6.1 LITE**

#### **❌ Innovation #4: Damage-Aware Attention**
**Reason**: 
- Clases raras: 12-58 samples post-split (marginal for prototypes)
- Hierarchical MTL already addresses rare classes (coarse head)
- Better to validate #1 + #5 first, add #4 in POC-6.2 if needed
- Saves: 2 días development

#### **❌ Innovation #6: Heritage Augmentation Expansion**
**Reason**:
- Already have 1,458 augmented (3.5x over 417 original)
- POC-5.9 validates no severe overfitting at 37.63% mIoU
- Expanding 1,458 → 2,927 requires regeneration (+3h CPU, +9.7 GB disk)
- Better to validate baseline first, expand in POC-6.2 if DG gap too high
- Saves: 1 día development + 3h CPU time

#### **❌ Innovation #2: MAE Pretraining**
**Reason**:
- ROI moderado con 1,458 samples (+4-6% vs +10-15% with 11k)
- Requires: +150h GPU time (3 encoders × 50 epochs)
- Not critical for workshop paper
- Save for POC-6.2 if targeting conference main track

#### **❌ Innovation #3: MAML Meta-Learning**
**Reason**:
- Complexity: 400 LOC, 5 días development
- GPU time: +143.5h (meta-train + adaptations)
- LOContent-only doesn't benefit from meta-learning (only 4 tasks)
- Save for POC-6.2 if doing LOMO 10-way

#### **❌ LOMO 10-way DG**
**Reason**:
- 1,458 ÷ 10 = 146 samples/material (marginal, high variance)
- LOContent more robust (365 samples/content, 3.6x margin)
- Saves: +42h GPU time (30 runs vs 12 runs)
- Can add in POC-6.2 with Heritage Augmentation (1,458 → 2,927)

---

## 📅 Implementation Timeline

### **Day 1-2: Hierarchical Multi-Task Learning**

**Tasks**:
1. Copy POC-5.5 hierarchical code to POC-5.9
   - `hierarchical_upernet.py` (530 LOC)
   - `losses.py` (Hierarchical Dice+Focal, 400 LOC)
   
2. Adapt to POC-5.9 structure:
   - Integrate with `src/model_factory.py`
   - Update configs (add hierarchical params)
   - Test with 1 epoch (verify VRAM, output shapes)

3. Create ground truth labels:
   - Binary: Clean (0) vs Damage (1-15 → 1)
   - Coarse: 4 groups (mapping from 16 classes)
   - Fine: 16 classes (original)

**Deliverables**:
- ✅ `src/models/hierarchical_upernet.py`
- ✅ `src/losses/hierarchical_loss.py`
- ✅ Updated configs: `configs/hierarchical_*.yaml`
- ✅ Test run log (1 epoch, 3 models)

**Verification**:
```bash
# Test hierarchical model
sbatch scripts/slurm_test_hierarchical.sh configs/hierarchical_segformer_b3.yaml

# Expected output:
# - train_loss_binary, train_loss_coarse, train_loss_fine
# - val_miou_binary, val_miou_coarse, val_miou_fine
# - VRAM: ~2.5 GB (vs 2.3 GB baseline, acceptable)
```

---

### **Day 3: Progressive Curriculum Implementation**

**Tasks**:
1. Modify `src/train.py` with staging logic:
   ```python
   # Stage 1: Binary (epochs 1-20)
   if epoch <= 20:
       freeze_heads(['coarse', 'fine'])
       optimizer = AdamW(binary_head.parameters(), lr=1e-5)
   
   # Stage 2: Coarse (epochs 21-40)
   elif epoch <= 40:
       freeze_heads(['fine'])
       optimizer = AdamW([binary_head, coarse_head].parameters(), lr=5e-5)
   
   # Stage 3: Fine (epochs 41-100)
   else:
       unfreeze_all_heads()
       optimizer = AdamW(model.parameters(), lr=1e-4)
   ```

2. Update checkpoint loading:
   - Stage 1 → Stage 2: Transfer encoder + binary head weights
   - Stage 2 → Stage 3: Transfer encoder + binary + coarse weights

3. Add curriculum metrics logging:
   - Log per-stage loss and mIoU
   - TensorBoard: Stage 1 (binary), Stage 2 (coarse), Stage 3 (fine)

**Deliverables**:
- ✅ `src/train_curriculum.py` (or modify `train.py`)
- ✅ Updated configs with staging params
- ✅ Test run log (curriculum stages working)

**Verification**:
```bash
# Test curriculum (10 epochs: 4 binary + 3 coarse + 3 fine)
sbatch scripts/slurm_test_curriculum.sh configs/curriculum_test.yaml

# Expected behavior:
# - Epochs 1-4: Only binary loss decreases
# - Epochs 5-7: Binary + coarse loss decrease
# - Epochs 8-10: All 3 losses decrease
```

---

### **Day 4: LOContent Splits Creation**

**Tasks**:
1. Create DG split generation script:
   ```python
   # scripts/create_locontent_splits.py
   import pandas as pd
   from pathlib import Path
   
   # Read metadata (content type per image)
   metadata = pd.read_csv('experiments/common-data/artefact_augmented/metadata.csv')
   
   # Create 4 splits (one per content type)
   contents = ['artistic', 'photographic', 'line_art', 'geometric']
   for held_out_content in contents:
       train_samples = metadata[metadata.content != held_out_content]
       test_samples = metadata[metadata.content == held_out_content]
       
       save_split(f'locontent_{held_out_content}.json', train_samples, test_samples)
   ```

2. Verify split balance:
   - Check: Each content has ~365 samples
   - Check: Train/test no overlap
   - Check: All 16 classes present in train split

3. Create training script for LOContent:
   ```bash
   # scripts/train_locontent.sh
   for content in artistic photographic line_art geometric; do
       for model in convnext segformer maxvit; do
           sbatch scripts/slurm_train.sh \
               --config configs/curriculum_${model}.yaml \
               --split manifests/locontent_${content}.json \
               --exp_name locontent_${content}_${model}
       done
   done
   ```

**Deliverables**:
- ✅ `scripts/create_locontent_splits.py`
- ✅ `manifests/locontent_*.json` (4 files)
- ✅ `scripts/train_locontent.sh`
- ✅ Split analysis report (sample counts, class distribution)

**Verification**:
```bash
python scripts/create_locontent_splits.py
python scripts/analyze_splits.py manifests/locontent_*.json

# Expected output:
# - Split 1 (artistic): 1,093 train / 365 test
# - Split 2 (photographic): 1,093 train / 365 test
# - Split 3 (line_art): 1,093 train / 365 test
# - Split 4 (geometric): 1,093 train / 365 test
# - All splits have all 16 classes in train
```

---

### **Day 5: Training Execution (GPU Day)**

**Tasks**:
1. **Baseline Training** (3 models × 100 epochs):
   ```bash
   # Sequential execution for reproducibility
   JOB1=$(sbatch --parsable scripts/slurm_train_curriculum.sh configs/curriculum_convnext.yaml)
   JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 scripts/slurm_train_curriculum.sh configs/curriculum_segformer.yaml)
   JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 scripts/slurm_train_curriculum.sh configs/curriculum_maxvit.yaml)
   ```
   - Time: 1.4h × 3 = **4.2 hours**

2. **LOContent Evaluation** (12 runs):
   ```bash
   # After baseline completes, run LOContent
   sbatch --dependency=afterany:$JOB3 scripts/train_locontent.sh
   ```
   - Time: 1.4h × 12 runs = **16.8 hours**

3. **Total GPU Time**: 4.2h + 16.8h = **21 hours** (~1 día en V100)

**Monitoring**:
```bash
# Check job status
squeue -u $USER

# Monitor training
tail -f logs/curriculum_segformer/train.log

# Check GPU usage
ssh node && watch -n 5 nvidia-smi
```

**Expected Checkpoints**:
- `logs/curriculum_convnext/best_model.pth` (379 MB)
- `logs/curriculum_segformer/best_model.pth` (543 MB)
- `logs/curriculum_maxvit/best_model.pth` (383 MB)
- `logs/locontent_*/best_model.pth` (12 files)

---

### **Day 6-7: Evaluation & Documentation**

**Tasks**:
1. **Baseline Evaluation**:
   ```bash
   sbatch scripts/slurm_evaluate.sh logs/curriculum_*/best_model.pth
   ```
   - Generate: metrics.json, confusion matrices, per-class IoU
   - Compare: Hierarchical (binary/coarse/fine) vs POC-5.9 (fine only)

2. **DG Gap Analysis**:
   ```bash
   python scripts/analyze_dg_gap.py \
       --baseline logs/curriculum_*/evaluation/metrics.json \
       --locontent logs/locontent_*/evaluation/metrics.json
   ```
   - Compute: In-domain mIoU, OOD mIoU, DG Gap per model
   - Bootstrap: 95% confidence intervals

3. **Visualization**:
   - Training curves: Binary → Coarse → Fine progression
   - DG gap plot: In-domain vs OOD per model
   - Per-class IoU: Hierarchical heads analysis
   - Confusion matrices: Where does model fail?

4. **Documentation**:
   - Update `POC6_EXECUTION_PLAN.md` with actual results
   - Create `POC6_RESULTS.md` with tables and figures
   - Prepare workshop paper draft (4-6 pages)

**Deliverables**:
- ✅ Evaluation metrics (JSON + CSV)
- ✅ Plots (training curves, DG gap, per-class IoU)
- ✅ Comparison table (vs POC-5.9 baseline)
- ✅ Workshop paper draft

---

## 📊 Expected Results

### **Baseline (with Hierarchical MTL + Progressive Curriculum)**

| Model | POC-5.9 (Fine) | POC-6.1 (Binary) | POC-6.1 (Coarse) | POC-6.1 (Fine) | Improvement |
|-------|----------------|------------------|------------------|----------------|-------------|
| **ConvNeXt** | 25.47% | 68-72% | 52-56% | **33-36%** | +7-11% |
| **SegFormer** | 37.63% | 72-76% | 58-62% | **45-50%** | +7-12% |
| **MaxViT** | 34.58% | 70-74% | 55-59% | **42-45%** | +7-11% |

**Key Observations**:
- Binary head: 68-76% mIoU (easy task, high performance)
- Coarse head: 52-62% mIoU (moderate difficulty)
- Fine head: 33-50% mIoU (hard task, but improved vs POC-5.9)
- **Cascade effect**: Easier tasks guide harder ones

---

### **Domain Generalization (LOContent)**

| Model | In-Domain mIoU | OOD mIoU (LOContent) | DG Gap | DG Gap (POC-5.9 expected) |
|-------|----------------|----------------------|--------|---------------------------|
| **ConvNeXt** | 33-36% | 20-24% | 13-14% | 18-22% |
| **SegFormer** | 45-50% | 30-36% | 15-16% | 22-28% |
| **MaxViT** | 42-45% | 28-32% | 14-15% | 20-26% |

**Key Improvements**:
- DG Gap reduction: -5 to -8% vs naive baseline
- Hierarchical MTL: Coarse head provides cross-domain guidance
- Progressive Curriculum: More stable features, better generalization

---

### **Per-Class IoU Analysis**

**Well-Detected Classes** (IoU > 40%):
- Clean: 88-92% (easy baseline, binary head strong)
- Material loss: 55-65% (structural, prominent)
- Peel: 38-48% (structural, visible)
- Discolouration: 35-45% (frequent, color-based)

**Moderately-Detected Classes** (IoU 20-40%):
- Stains: 28-38%
- Dirt spots: 25-35%
- Cracks: 22-32%
- Scratches: 20-30%

**Poorly-Detected Classes** (IoU < 20%):
- Lightleak: 8-15% (rare, <1% dataset)
- Burn marks: 10-18% (rare, subtle)
- Hairs: 12-20% (thin, hard to segment)
- Structural defects: 5-12% (very rare, <0.5% dataset)

**Hierarchical MTL Impact**:
- Rare classes: +5-10% IoU (coarse head guidance)
- Moderate classes: +3-5% IoU (better features)
- Frequent classes: +1-3% IoU (already well-detected)

---

## 🎓 Publication Strategy

### **Target Venue**: Conference Workshop

**Options**:
1. **CVPR Workshop** (Computer Vision for Art Analysis)
   - Submission: February 2026
   - Acceptance: ~40-50%
   - Format: 4-6 pages
   - Focus: Heritage art preservation

2. **ICCV Workshop** (Cultural Heritage Preservation)
   - Submission: June 2026
   - Acceptance: ~45-55%
   - Format: 4-6 pages
   - Focus: DG for heritage domain

3. **BMVC** (Short paper track)
   - Submission: April 2026
   - Acceptance: ~30-40%
   - Format: 6 pages
   - Focus: Multiclass segmentation + DG

---

### **Paper Structure**

**Title**: "Hierarchical Multi-Task Learning for Domain Generalization in Heritage Art Damage Detection"

**Abstract** (150-200 words):
- Problem: Heritage art damage detection suffers from domain shift (materials, content types)
- Solution: Hierarchical MTL (binary → coarse → fine) + Progressive Curriculum
- Dataset: ARTeFACT 1,458 augmented samples, 16 damage classes
- Results: 45-50% mIoU (SegFormer), 15-16% DG gap (LOContent)
- Impact: +20-26% vs naive baseline, -5 to -8% DG gap reduction

**1. Introduction** (1 page):
- Heritage art preservation challenges
- Domain shift problem (materials, content, lighting)
- Research questions: RQ1 (multiclass), RQ2 (DG)
- Contributions: Hierarchical MTL + Progressive Curriculum for small datasets

**2. Related Work** (0.5 pages):
- Heritage art segmentation
- Domain generalization techniques
- Multi-task learning for segmentation

**3. Method** (1.5 pages):
- 3.1 Hierarchical Multi-Task Learning
  - Architecture diagram (3 heads)
  - Class grouping (4 coarse groups)
  - Loss function (weighted sum)
- 3.2 Progressive Curriculum Learning
  - 3-stage training (binary → coarse → fine)
  - Transfer learning between stages
- 3.3 Domain Generalization Evaluation
  - LOContent protocol (4-fold)
  - Metrics: In-domain mIoU, OOD mIoU, DG gap

**4. Experiments** (1.5 pages):
- 4.1 Dataset: ARTeFACT 1,458 samples
- 4.2 Implementation: 3 architectures (CNN/ViT/Hybrid)
- 4.3 Baseline comparison (vs POC-5.9)
- 4.4 Ablation study:
  - Hierarchical MTL only
  - Progressive Curriculum only
  - Both combined

**5. Results** (1 page):
- Table 1: Baseline comparison (3 models × 3 heads)
- Table 2: DG gap (LOContent, 4 contents × 3 models)
- Figure 1: Training curves (binary → coarse → fine)
- Figure 2: Per-class IoU (rare class improvement)

**6. Conclusion** (0.5 pages):
- Summary: Hierarchical MTL + Curriculum effective for small datasets
- Limitations: LOMO not evaluated (marginal samples/material)
- Future work: Damage-Aware Attention, Heritage Augmentation expansion

---

### **Key Contributions for Paper**

1. **Novel for Heritage Domain**:
   - Hierarchical MTL adapted for heritage damage taxonomy
   - 4 coarse groups (structural, contamination, color, optical)
   - Progressive Curriculum for small datasets (<2k samples)

2. **Empirical Validation**:
   - +20-26% mIoU improvement (37% → 45-50%)
   - -5 to -8% DG gap reduction
   - Cross-architecture validation (CNN/ViT/Hybrid)

3. **Practical Impact**:
   - Works with small datasets (1,458 augmented samples)
   - No complex DG techniques (MAML, MAE, etc.)
   - Reproducible on single GPU (21h training time)

---

## 🚨 Risk Management

### **Risk 1: Hierarchical MTL doesn't improve**
**Mitigation**:
- POC-5.5 already validates +22% mIoU with 418 samples
- With 1,458 samples, improvement more likely
- Fallback: Use POC-5.9 baseline + Progressive Curriculum only

**Contingency**:
- If fine head < 40% mIoU: Adjust loss weights (increase fine weight)
- If coarse head < 50%: Re-group classes (merge similar damage types)

---

### **Risk 2: DG gap too high (>25%)**
**Mitigation**:
- LOContent more robust than LOMO (365 vs 146 samples)
- Hierarchical MTL provides cross-domain guidance (coarse head)
- Progressive Curriculum learns more stable features

**Contingency**:
- If DG gap > 25%: Add Heritage Augmentation expansion (POC-6.2)
- If specific content fails: Analyze failure modes, augment that content type
- If all contents fail: May need MAE pretraining (POC-6.2)

---

### **Risk 3: Training time exceeds 21h**
**Mitigation**:
- POC-5.9 validated 42 min per model (50 epochs)
- Doubling epochs (100) → 1.4h per model (conservative estimate)
- V100 32GB well-optimized (AMP, OneCycleLR)

**Contingency**:
- If baseline > 6h: Reduce epochs to 60 (sacrifice 5-10% final mIoU)
- If LOContent > 24h: Run in parallel on 2 GPUs (if available)
- If total > 30h: Skip weakest model (ConvNeXt, lowest baseline)

---

### **Risk 4: VRAM overflow with Hierarchical MTL**
**Mitigation**:
- POC-5.5 tested on 6GB RTX 3050 (839 MB usage, 13.7%)
- V100 32GB has 38x more VRAM
- Hierarchical heads add ~20% parameters vs single head

**Contingency**:
- If VRAM > 90%: Reduce batch size (32 → 24 → 16)
- If still overflow: Enable gradient checkpointing (trade 20% speed)
- If critical: Skip MaxViT (highest memory), focus on SegFormer + ConvNeXt

---

### **Risk 5: Results not publishable (<40% mIoU)**
**Mitigation**:
- POC-5.5 achieved 22% mIoU with 418 samples
- 1,458 samples (3.5x more) + innovations → 40%+ expected
- Even 40-45% mIoU publishable in workshop (focus on DG gap reduction)

**Contingency**:
- If < 40%: Focus paper on DG gap reduction (novel contribution)
- If < 35%: Analyze failure modes, propose future work (POC-6.2)
- If < 30%: Pivot to technical report, skip publication until POC-6.2

---

## 📋 Success Criteria

### **Minimum Viable (Workshop Paper)**
- ✅ Hierarchical MTL implemented and working (3 heads, proper loss)
- ✅ Progressive Curriculum implemented (3 stages, checkpoint transfer)
- ✅ Baseline mIoU ≥ 40% (best model, fine head)
- ✅ DG gap ≤ 25% (LOContent average)
- ✅ All code reproducible (configs, scripts, manifests)

### **Target (Strong Workshop Paper)**
- ✅ Baseline mIoU ≥ 45% (SegFormer, fine head)
- ✅ DG gap ≤ 20% (LOContent average)
- ✅ Rare class IoU > 15% (vs < 10% in POC-5.9)
- ✅ Cross-architecture validation (3 models consistent)
- ✅ Ablation study (Hierarchical only, Curriculum only, Both)

### **Stretch (Conference Main Track)**
- ✅ Baseline mIoU ≥ 50% (would require POC-6.2 expansions)
- ✅ DG gap ≤ 15% (would require MAE/MAML)
- ✅ LOMO validation (would require Heritage Aug expansion)
- ✅ Damage-Aware Attention (would require +2 días dev)

---

## 🔄 Next Steps After POC-6.1

### **If Results Good (mIoU ≥ 45%, DG gap ≤ 20%)**
**Action**: Write workshop paper, submit Q1 2026

**Optional POC-6.2 Expansion**:
- Add Damage-Aware Attention (#4): +2 días dev, +3-5% mIoU expected
- Expand Heritage Augmentation (1,458 → 2,927): +1 día dev, +3h regen
- Evaluate LOMO 10-way (now robust with 2,927): +42h GPU
- Target: Conference main track (CVPR, ICCV)

---

### **If Results Marginal (mIoU 40-45%, DG gap 20-25%)**
**Action**: Analyze bottlenecks, iterate POC-6.1.1

**Potential Improvements**:
1. **Loss weight tuning**: Adjust binary/coarse/fine ratios
2. **Class grouping**: Re-organize 4 coarse groups
3. **Curriculum stages**: Try different epoch splits (30+30+40)
4. **Augmentation**: Add ColorJitter, GaussianNoise (in-place, no regen)

**Timeline**: +2 días experimentation, +21h GPU (re-train)

---

### **If Results Poor (mIoU < 40%, DG gap > 25%)**
**Action**: Deep analysis, consider POC-6.2 mandatory

**Root Cause Analysis**:
1. **Hierarchical MTL failing**: Check loss curves, head contributions
2. **Curriculum not converging**: Verify checkpoint transfer working
3. **Dataset too small**: May need Heritage Aug expansion (critical)
4. **Architecture mismatch**: SegFormer may not suit hierarchical (unlikely)

**Pivot to POC-6.2**:
- Add Heritage Augmentation (1,458 → 2,927): MANDATORY
- Add Damage-Aware Attention: MANDATORY
- Re-evaluate with expanded dataset
- Timeline: +1 semana adicional

---

## 📁 Deliverables Checklist

### **Code**
- [ ] `src/models/hierarchical_upernet.py` (530 LOC)
- [ ] `src/losses/hierarchical_loss.py` (400 LOC)
- [ ] `src/train_curriculum.py` (modified from train.py)
- [ ] `scripts/create_locontent_splits.py` (200 LOC)
- [ ] `configs/curriculum_*.yaml` (3 files: convnext, segformer, maxvit)
- [ ] `scripts/train_locontent.sh` (batch training script)

### **Data**
- [ ] `manifests/locontent_artistic.json`
- [ ] `manifests/locontent_photographic.json`
- [ ] `manifests/locontent_line_art.json`
- [ ] `manifests/locontent_geometric.json`
- [ ] Split analysis report (sample counts, balance verification)

### **Results**
- [ ] Baseline checkpoints: `logs/curriculum_*/best_model.pth` (3 files)
- [ ] LOContent checkpoints: `logs/locontent_*/best_model.pth` (12 files)
- [ ] Evaluation metrics: `logs/*/evaluation/metrics.json` (15 files)
- [ ] Comparison table: Baseline vs POC-5.9
- [ ] DG gap table: LOContent per model
- [ ] Training curves: Binary → Coarse → Fine progression
- [ ] Per-class IoU plots: Hierarchical heads analysis
- [ ] Confusion matrices: Error analysis

### **Documentation**
- [ ] `POC6_EXECUTION_PLAN.md` (this document, updated with results)
- [ ] `POC6_RESULTS.md` (detailed results, tables, figures)
- [ ] Workshop paper draft (4-6 pages, IEEE/ACM format)
- [ ] Code README: How to reproduce
- [ ] Commit + push to git

---

## 🎯 Final Timeline Summary

| Day | Tasks | Hours | Deliverables |
|-----|-------|-------|--------------|
| **1** | Hierarchical MTL setup | 8h | Code + test run |
| **2** | Hierarchical MTL testing | 8h | Verified working |
| **3** | Progressive Curriculum | 8h | Staging logic + test |
| **4** | LOContent splits | 8h | Manifests + scripts |
| **5** | Training (GPU) | 21h | Checkpoints (15 models) |
| **6** | Evaluation | 8h | Metrics + plots |
| **7** | Documentation | 8h | Paper draft + README |

**Total**: 5 días development + 1 día GPU + 2 días analysis = **8 días calendario**

**Critical Path**:
- Days 1-4: Development (can parallelize Day 3-4 if needed)
- Day 5: GPU training (blocking, 21h continuous)
- Days 6-7: Analysis & writing (can extend if needed)

---

## ✅ Ready to Execute

**Status**: 🟢 All requirements met, ready to start Day 1

**Prerequisite Checklist**:
- ✅ POC-5.9 baseline validated (37.63% mIoU)
- ✅ Dataset available (1,458 augmented, 9.7 GB in git)
- ✅ POC-5.5 code available (Hierarchical MTL reference)
- ✅ SLURM scripts working (V100 access confirmed)
- ✅ Git repository clean (no uncommitted changes)

**Starting Command**:
```bash
# Day 1: Begin Hierarchical MTL implementation
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid
git checkout -b poc6.1-lite
mkdir -p experiments/artefact-poc61-lite
# ... start coding
```

---

**Document Version**: 1.0  
**Last Updated**: November 17, 2025  
**Status**: Ready to Execute  
**Next Action**: Start Day 1 (Hierarchical MTL implementation)

