# POC-6: Multi-Backbone Multiclass Segmentation + Domain Generalization

**Parent**: POC-5 (Binary segmentation, 50 samples)  
**Goal**: Answer **RQ1** (CNN vs ViT vs Hybrid on multiclass) + **RQ2** (Domain Generalization) completely  
**Timeline**: 4-6 weeks  
**Status**: 🚧 Planning Phase

---

## 🎯 Research Questions (from main.tex)

### **RQ1**: Which model family (CNN/ViT/Hybrid) best detects and classifies painting deterioration **across damage types**?

**Metrics**: 
- mIoU (mean IoU across 16 classes)
- macro-F1 (equal weight per class)
- Per-class IoU/F1 for each of 15 damage types + Clean

**Approach**:
1. Scale from **binary** (Clean vs Damage) to **multiclass** (16 classes)
2. Train ConvNeXt-Tiny, Swin-Tiny, MaxViT-Tiny on ARTeFACT full dataset
3. Report in-domain performance (train on X, test on held-out X)

### **RQ2**: Which family generalizes better to **unseen cultural heritage collections**?

**Metrics**:
- DG gap = in-domain mIoU - OOD mIoU
- Bootstrap 95% CIs for statistical significance

**Approach**:
1. **LOMO** (Leave-One-Material-Out): 10 materials as domains
2. **LOContent** (Leave-One-Content-Out): 4 contents as domains
3. Measure OOD drop for each architecture
4. Apply DG techniques to close the gap

---

## 📊 Dataset Scaling Plan

### **POC-5 (Current)**
- **Samples**: 50 images (40 train / 10 val)
- **Classes**: 2 (Clean, Damage) - binary
- **Source**: `artefact-data-obtention/data/demo/`
- **Annotations**: ~50 binary masks

### **POC-6 (Target)**
- **Samples**: ~11,000+ annotations (ARTeFACT full dataset)
- **Classes**: 16 (Clean + 15 damage types)
- **Source**: HuggingFace `danielaivanova/damaged-media`
- **Metadata**: Material (10 types), Content (4 types), Type (unknown count)

**Class Taxonomy** (ARTeFACT):
```
0:  Clean
1:  Material loss
2:  Peel
3:  Dust
4:  Scratch
5:  Hair
6:  Dirt
7:  Fold
8:  Writing
9:  Cracks
10: Staining
11: Stamp
12: Sticker
13: Puncture
14: Burn marks
15: Lightleak
255: Background (IGNORE_INDEX)
```

---

## 🏗️ Architecture (No Changes from POC-5)

All models keep the **same UPerNet decoder** for fair comparison:

| Model | Type | Encoder | Channels | Params | ImageNet-1k |
|-------|------|---------|----------|--------|-------------|
| ConvNeXt-Tiny | CNN | ConvNeXt-Tiny | [96,192,384,768] | 37.7M | 82.1% |
| Swin-Tiny | Transformer | Swin-Tiny | [96,192,384,768] | 37.4M | 81.3% |
| MaxViT-Tiny | Hybrid | MaxViT-Tiny | [64,128,256,512] | 39.2M | 83.6% |

**UPerNet Decoder** (unified):
- PPM (Pyramid Pooling) at scales [1,2,3,6]
- FPN (Feature Pyramid Network) with 256 channels
- Dynamic upsampling to 512×512

---

## 📅 Implementation Phases

### **Phase 1: Dataset Preparation** (Week 1)

**Objective**: Download and prepare ARTeFACT full dataset with multiclass labels.

**Tasks**:
1. ✅ **Download ARTeFACT full** from HuggingFace
   ```bash
   # In artefact-data-obtention/
   huggingface-cli download danielaivanova/damaged-media --repo-type dataset
   ```

2. ✅ **Verify class distribution**
   - Check samples per class (0-15)
   - Identify rare classes (e.g., Lightleak)
   - Compute class imbalance ratios

3. ✅ **Create train/val/test splits** (stratified)
   - Train: 70%, Val: 15%, Test: 15%
   - Stratify by damage type to ensure all 16 classes in each split
   - Save manifest CSVs with metadata (material, content, type)

4. ✅ **Update dataset.py for multiclass**
   - Change `num_classes=2` → `num_classes=16`
   - Keep `ignore_index=255` for Background
   - Verify one-hot encoding for rare classes

**Verification**:
- [ ] Print class histogram (train/val/test)
- [ ] Visual inspection: 1 sample per damage type
- [ ] Confirm no data leakage (perceptual hashing)

**Deliverables**:
- `data/artefact_full/` with train/val/test splits
- `manifests/train.csv`, `val.csv`, `test.csv` with metadata
- Exploratory notebook: `notebooks/dataset_analysis.ipynb`

---

### **Phase 2: Multiclass Training (RQ1)** (Week 2-3)

**Objective**: Train 3 models on multiclass task and establish in-domain performance.

**Tasks**:

1. ✅ **Update training config**
   - `num_classes=16` (0-15 + ignore 255)
   - Loss: DiceFocalLoss with class weights (inverse frequency)
   - Consider focal loss `alpha` per class for rare types
   - Epochs: 100 (longer than binary due to complexity)
   - Early stopping: patience=15 on macro-F1

2. ✅ **Train ConvNeXt-Tiny**
   ```bash
   make train-convnext-multiclass
   ```
   - Expected time: ~6-8 hours (depends on dataset size)
   - Monitor per-class IoU during training

3. ✅ **Train Swin-Tiny**
   ```bash
   make train-swin-multiclass
   ```
   - Expected time: ~7-9 hours

4. ✅ **Train MaxViT-Tiny**
   ```bash
   make train-maxvit-multiclass
   ```
   - Expected time: ~8-10 hours

**Hyperparameters** (inherit from POC-5):
- Batch size: 4
- Learning rate: 0.0003 (AdamW)
- Warmup: 10 epochs
- Scheduler: Cosine annealing (min_lr=5e-7)
- Augmentations: Flip, Rotate ±20°, ColorJitter, Blur

**Metrics to log**:
- Overall: mIoU, macro-F1, accuracy
- Per-class: IoU, F1, Precision, Recall for each of 16 classes
- Confusion matrix (16×16)

**Verification**:
- [ ] All 3 models complete 100 epochs
- [ ] mIoU > 0.40 (reasonable for 16-class multiclass)
- [ ] No class has 0.0 IoU (all classes detected)

**Deliverables**:
- `logs/{model}_multiclass/checkpoints/best_model.pth` (3 models)
- `logs/{model}_multiclass/training/training_log.csv` (3 files)
- Training curves plot (loss, mIoU, macro-F1 over epochs)

---

### **Phase 3: Multiclass Evaluation (RQ1)** (Week 3)

**Objective**: Comprehensive evaluation on held-out test set.

**Tasks**:

1. ✅ **Evaluate all 3 models**
   ```bash
   make evaluate-all-multiclass
   ```
   - Generate per-class metrics table
   - Confusion matrices (16×16)
   - Prediction visualizations (6 samples per model)

2. ✅ **Cross-model comparison**
   ```bash
   make compare-multiclass
   ```
   - Generate comparison table (similar to POC-5)
   - Per-class IoU bar chart (15 damage types)
   - Identify which model excels at which damage type

3. ✅ **Qualitative analysis**
   - Which damage types are easy? (expected: Material Loss, Staining)
   - Which are hard? (expected: Hair, Dust, Lightleak)
   - Does CNN/ViT/Hybrid show complementary strengths?

**Expected Results** (hypothesis):
- **MaxViT**: Best overall mIoU (~0.50-0.55)
- **Swin**: Good at large-scale damage (Staining, Material Loss)
- **ConvNeXt**: Good at fine-grained (Cracks, Scratch)

**Verification**:
- [ ] Results table matches paper format (Table X in main.tex)
- [ ] Statistical significance test (bootstrap paired comparison)
- [ ] Per-class IoU plot generated

**Deliverables**:
- `logs/comparison_multiclass/comparison_table.txt`
- `logs/comparison_multiclass/per_class_iou.png`
- `logs/comparison_multiclass/summary_report.txt`
- **RQ1 ANSWERED**: Table for paper

---

### **Phase 4: Domain Generalization Setup** (Week 4)

**Objective**: Prepare LOMO and LOContent splits for DG evaluation.

**Tasks**:

1. ✅ **Implement LOMO splits** (Leave-One-Material-Out)
   ```python
   # scripts/create_lomo_splits.py
   materials = ['Paintings', 'Photographs', 'Textiles', 'Paper', ...]  # 10 total
   for held_out_material in materials:
       train = filter(lambda x: x.material != held_out_material)
       test_ood = filter(lambda x: x.material == held_out_material)
       save_split(f'lomo_{held_out_material}.json')
   ```

2. ✅ **Implement LOContent splits** (Leave-One-Content-Out)
   ```python
   # scripts/create_locontent_splits.py
   contents = ['Portraits', 'Landscapes', 'Still Life', 'Abstract']  # 4 total
   for held_out_content in contents:
       train = filter(lambda x: x.content != held_out_content)
       test_ood = filter(lambda x: x.content == held_out_content)
       save_split(f'locontent_{held_out_content}.json')
   ```

3. ✅ **Verify splits**
   - Each LOMO split: 9 materials in-domain, 1 OOD
   - Each LOContent split: 3 contents in-domain, 1 OOD
   - Check sample count sufficiency (min 100 images per split)

**Verification**:
- [ ] 10 LOMO split JSONs created
- [ ] 4 LOContent split JSONs created
- [ ] No overlap between train and OOD test

**Deliverables**:
- `manifests/lomo/*.json` (10 files)
- `manifests/locontent/*.json` (4 files)
- `notebooks/dg_splits_analysis.ipynb`

---

### **Phase 5: Domain Generalization Baseline** (Week 4-5)

**Objective**: Measure OOD performance drop (DG gap) for each architecture.

**Tasks**:

1. ✅ **LOMO evaluation** (10 experiments per model)
   ```bash
   # For each material (e.g., 'Photographs')
   python scripts/train_lomo.py --held_out_material Photographs --model maxvit
   python scripts/evaluate_lomo.py --held_out_material Photographs --model maxvit
   ```
   - Train on 9 materials → test on held-out material
   - Repeat for all 3 models × 10 materials = **30 training runs**

2. ✅ **LOContent evaluation** (4 experiments per model)
   ```bash
   # For each content (e.g., 'Portraits')
   python scripts/train_locontent.py --held_out_content Portraits --model swin
   python scripts/evaluate_locontent.py --held_out_content Portraits --model swin
   ```
   - Train on 3 contents → test on held-out content
   - Repeat for all 3 models × 4 contents = **12 training runs**

3. ✅ **Compute DG gap**
   - For each model:
     - In-domain mIoU (from Phase 3)
     - OOD mIoU (average across LOMO/LOContent)
     - DG gap = In-domain - OOD
   - Bootstrap 95% CIs for statistical significance

**Expected Results** (hypothesis):
- **ViT/Hybrid** generalize better than CNN (smaller DG gap)
- LOMO harder than LOContent (materials more diverse)
- DG gap: ~10-15% mIoU drop

**Verification**:
- [ ] All 42 training runs complete (30 LOMO + 12 LOContent)
- [ ] DG gap table with CIs generated
- [ ] OOD mIoU > 0.30 (not catastrophic collapse)

**Deliverables**:
- `logs/lomo_results/` (30 subdirs with checkpoints+metrics)
- `logs/locontent_results/` (12 subdirs)
- `logs/dg_baseline/dg_gap_table.csv`
- **RQ2 BASELINE**: DG gap per architecture

---

### **Phase 6: Closing the DG Gap** (Week 5-6)

**Objective**: Apply DG techniques to reduce OOD performance drop.

**Tasks**:

1. ✅ **Style/Frequency Augmentation**
   - Implement `RandomStyleTransfer` (palette swap)
   - Implement `FourierDomainAdaptation` (frequency perturbation)
   - Train MaxViT with aggressive augmentation
   - Measure ΔLOMO mIoU

2. ✅ **MixUp/CutMix for Segmentation**
   - Adapt MixUp for masks (blend images + masks)
   - Train MaxViT with MixUp (α=0.4)
   - Measure ΔLOMO mIoU

3. ✅ **Domain Alignment Regularizers**
   - Implement **Deep CORAL** (feature distribution alignment)
   - Implement **IRM** (Invariant Risk Minimization)
   - Implement **Fishr** (gradient variance minimization)
   - Train MaxViT with each regularizer
   - Ablation study: which works best?

4. ✅ **Test-Time Adaptation (TTA)**
   - Implement **TENT** (entropy minimization with BN affine params)
   - Run TENT on OOD test sets (per LOMO split)
   - Measure uplift without retraining

**Ablation Matrix**:
| Technique | LOMO mIoU | LOContent mIoU | Δ from Baseline |
|-----------|-----------|----------------|-----------------|
| Baseline (MaxViT) | X.XX | Y.YY | - |
| +Style/Fourier | X.XX | Y.YY | +Z.ZZ |
| +MixUp | X.XX | Y.YY | +Z.ZZ |
| +Deep CORAL | X.XX | Y.YY | +Z.ZZ |
| +IRM | X.XX | Y.YY | +Z.ZZ |
| +Fishr | X.XX | Y.YY | +Z.ZZ |
| +TENT (TTA) | X.XX | Y.YY | +Z.ZZ |

**Expected Results**:
- Style/Fourier: +2-3% mIoU
- MixUp: +1-2% mIoU
- CORAL/IRM: +3-5% mIoU
- TENT: +2-4% mIoU (no training cost!)

**Verification**:
- [ ] Each technique improves OOD mIoU
- [ ] No in-domain collapse (ID mIoU unchanged or improved)
- [ ] Best combination identified

**Deliverables**:
- `logs/dg_closures/` with ablation results
- Ablation table (CSV + plot)
- **RQ2 COMPLETE**: DG gap + methods to close it

---

## 📊 Final Deliverables (End of POC-6)

### **Code**:
- [ ] Updated `dataset.py` (multiclass support)
- [ ] Updated `train.py` (class weighting)
- [ ] New `create_lomo_splits.py`
- [ ] New `create_locontent_splits.py`
- [ ] New `train_lomo.py`, `train_locontent.py`
- [ ] New `dg_techniques.py` (CORAL, IRM, Fishr, TENT)

### **Results for Paper**:
- [ ] **Table RQ1**: In-domain multiclass performance (3 models × 16 classes)
- [ ] **Table RQ2**: DG gap comparison (3 models × LOMO/LOContent)
- [ ] **Figure 1**: Per-class IoU bar chart (15 damage types × 3 models)
- [ ] **Figure 2**: Training curves (multiclass)
- [ ] **Figure 3**: DG gap plot (in-domain vs OOD)
- [ ] **Figure 4**: Ablation study (DG techniques)

### **Documentation**:
- [ ] POC-6 README with full results
- [ ] Update main README with POC-6 summary
- [ ] Add results to `main.tex` (Results section)
- [ ] Commit + push all code and results

---

## ⏱️ Timeline Summary

| Week | Phase | Tasks | Hours |
|------|-------|-------|-------|
| **1** | Dataset Prep | Download, verify, create splits | 20h |
| **2** | Multiclass Training | Train 3 models (100 epochs each) | 40h (mostly GPU time) |
| **3** | Multiclass Eval | Evaluate, compare, analyze RQ1 | 20h |
| **4** | DG Setup | Create LOMO/LOContent splits | 15h |
| **4-5** | DG Baseline | Train 42 models (LOMO+LOContent) | 60h (mostly GPU time) |
| **5-6** | DG Closures | Implement techniques, ablations | 30h |
| **6** | Documentation | Write results, update paper | 15h |

**Total**: 200 hours over 6 weeks (33h/week = sustainable pace)

---

## 🚨 Risk Mitigation

### **Risk 1**: Dataset too large for GPU memory
- **Mitigation**: Keep batch_size=4, use gradient accumulation if needed
- **Fallback**: Reduce resolution to 448×448

### **Risk 2**: Some damage classes have <10 samples
- **Mitigation**: Merge rare classes (e.g., Lightleak → Other)
- **Fallback**: Report metrics on subset of 10 most common classes

### **Risk 3**: DG gap too large (>30% mIoU drop)
- **Mitigation**: Apply multiple DG techniques simultaneously
- **Fallback**: Report DG as open problem, focus on RQ1

### **Risk 4**: Training 42 models takes too long
- **Mitigation**: Parallelize LOMO runs across multiple GPUs
- **Fallback**: Sample 5/10 materials, 3/4 contents for faster iteration

### **Risk 5**: TENT diverges on some OOD splits
- **Mitigation**: Add clip_grad_norm and early stopping on entropy
- **Fallback**: Report TENT as unstable, skip this technique

---

## 🎯 Success Criteria

**Minimum Viable**:
- ✅ RQ1 answered: Table with 3 models × 16 classes
- ✅ RQ2 baseline: DG gap measured for 3 models

**Target**:
- ✅ Per-class analysis identifies strengths/weaknesses
- ✅ DG gap reduced by ≥5% mIoU with closures
- ✅ All results reproducible with manifests + seeds

**Stretch**:
- ✅ Publish intermediate results (RQ1 only) while DG runs
- ✅ Create interactive visualization (Gradio app)
- ✅ Open-source full pipeline as benchmark

---

## 📚 References (for Implementation)

1. **ARTeFACT Dataset**: https://huggingface.co/datasets/danielaivanova/damaged-media
2. **Deep CORAL**: https://arxiv.org/abs/1607.01719
3. **IRM**: https://arxiv.org/abs/1907.02893
4. **Fishr**: https://proceedings.mlr.press/v162/rame22a.html
5. **TENT**: https://arxiv.org/abs/2006.10726
6. **MixUp for Segmentation**: https://arxiv.org/abs/1905.04899
7. **UPerNet**: https://arxiv.org/abs/1807.10221

---

## 🔄 Iterative Approach

**Week 1 Checkpoint**: 
- If dataset too complex → simplify to top 10 classes
- If GPU OOM → reduce batch size or resolution

**Week 3 Checkpoint** (RQ1 done):
- If results publishable → write draft
- If DG seems infeasible → pivot to data augmentation study

**Week 5 Checkpoint** (DG baseline done):
- If gap small (<10%) → skip closures, focus on analysis
- If gap huge (>40%) → investigate data leakage

---

**Status**: 🟢 Ready to start Phase 1 (Dataset Preparation)  
**Next Action**: Download ARTeFACT full dataset and analyze class distribution
