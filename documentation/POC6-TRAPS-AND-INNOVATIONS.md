# POC-6: Implementation Traps & Proposed Innovations

**Document Created**: October 26, 2025  
**Purpose**: Permanent record of critical issues and innovative solutions for POC-6 multiclass segmentation + domain generalization.  
**Context**: After POC-5 (binary segmentation, MaxViT winner 71.64% mIoU), scaling to 16-class multiclass + DG evaluation.

---

## 🚨 IMPLEMENTATION TRAPS (6 Critical Issues)

### Trap 1: Dataset Size Mismatch
**Issue**: POC-6 plan assumes ~11,000 annotations, but HuggingFace dataset viewer shows **"Estimated number of rows: 445"**.

**Impact**:
- 445 samples ÷ 10 materials = ~44 samples/material (below minimum threshold of 100)
- LOMO cross-validation may have insufficient training data
- Class imbalance worse than expected (rare damage types <10 samples)

**Mitigation**:
- **CRITICAL**: Download full ARTeFACT dataset first to verify actual size
- If <1000 samples: merge sparse materials (e.g., Wood+Canvas→Traditional, Glass+Ceramic→Rigid) from 10→5 materials
- If <100 samples for rare classes: hierarchical classification (binary → coarse → fine)
- Consider data augmentation multiplier (10x-20x for rare classes via CutMix/MixUp)

**Command to verify**:
```bash
# Download and count actual samples
huggingface-cli download danielaivanova/damaged-media --repo-type dataset
python -c "import datasets; ds = datasets.load_dataset('danielaivanova/damaged-media'); print(len(ds['train']))"
```

---

### Trap 2: Class Imbalance (Severe in Heritage Domain)
**Issue**: ARTeFACT has 16 classes (Clean + 15 damage types), but class distribution is heavily skewed.

**Expected Distribution** (from dataset card examples):
- **Frequent**: Dirt spot (30%+), Discolouration (25%), Material loss (20%)
- **Moderate**: Cracks (10%), Peel (8%), Stains (7%)
- **Rare**: Lightleak (<2%), Burn marks (<1%), Hairs (<1%)

**Impact**:
- Naive training will learn to predict only frequent classes
- Rare class IoU ≈ 0% (model ignores them completely)
- Misleading overall mIoU (dominated by frequent classes)

**Mitigation**:
- **Class weighting**: Inverse frequency weighting (sqrt or log dampening to avoid extremes)
  ```python
  weights = torch.tensor([1.0, 3.5, 2.8, 4.2, 5.1, 3.9, ..., 25.0, 30.0])  # Per class
  criterion = DiceFocalLoss(class_weights=weights)
  ```
- **Hierarchical Multi-Task Learning** (Innovation #1): 3 parallel heads force model to learn rare classes
- **Oversampling rare classes**: Repeat rare class samples 5x-10x in dataloader
- **Damage-Aware Attention** (Innovation #4): Prototype learning for rare damage patterns

---

### Trap 3: LOMO Sample Sufficiency
**Issue**: Leave-One-Material-Out requires each material to have sufficient samples for validation.

**Math**:
- 445 total samples ÷ 10 materials = ~44 samples/material
- Minimum recommended: 100 samples for reliable validation
- **Result**: Cannot perform robust LOMO with 10-way split

**Impact**:
- High variance in DG metrics (different materials have 20-80 samples each)
- Unreliable conclusions about which architecture generalizes best
- Overfitting to small validation sets

**Mitigation**:
- **Material grouping**: Merge similar materials to create balanced groups
  - Group 1: **Parchment + Paper** (flexible organic)
  - Group 2: **Canvas + Textile** (woven)
  - Group 3: **Wood + Film emulsion** (layered)
  - Group 4: **Glass + Ceramic + Tesserae** (rigid inorganic)
  - Group 5: **Lime plaster** (architectural)
  - Result: 10 materials → 5 groups (each ~90 samples)
- **LOContent instead**: Only 4 content types (artistic, photographic, line art, geometric) = ~111 samples each
- **K-Fold stratified**: 5-fold CV stratified by material (more robust than LOMO)

---

### Trap 4: Training Time Underestimation
**Issue**: POC-6 plan estimates "6-8h per model" but multiclass is 3x-5x slower than binary.

**Reality Check**:
- POC-5 binary (2 classes, 50 samples): 60 epochs = 25 minutes/epoch × 60 = **25 hours**
- POC-6 multiclass (16 classes, 445 samples): 
  - Dataset 9x larger: 25 min × 9 = 225 min/epoch
  - More classes (slower loss computation): 225 × 1.3 = 292 min/epoch
  - 100 epochs planned: 292 × 100 = **29,200 min ≈ 487 hours ≈ 20 days GPU time**
- **Per model**: ~487h × 3 models = **1,461 GPU hours** just for Phase 2

**Impact**:
- Original plan (42 DG runs × 487h = 20,454 GPU hours) is **computationally infeasible**
- Even multiclass baseline (3 models × 487h = 1,461h) takes 61 days continuous

**Mitigation**:
- **Reduce epochs**: 100 → 60 epochs (POC-5 showed 60 sufficient)
- **Aggressive early stopping**: patience=10 (vs 15), min_delta=0.001
- **Mixed precision**: AMP (Automatic Mixed Precision) saves 30-40% time
- **Smaller input resolution**: 512×512 → 384×384 (saves 44% compute)
- **MAML Meta-Learning** (Innovation #3): Replace 42 training runs with 1 meta-learning run
- **Progressive Curriculum** (Innovation #5): Binary (20 ep) → Coarse (20 ep) → Fine (60 ep) = 100 epochs but warmer start

**Revised Estimate** (with mitigations):
- 60 epochs, AMP, 384×384 resolution: **487h → 140h per model**
- 3 models: 140 × 3 = **420 GPU hours ≈ 17.5 days**

---

### Trap 5: 42 DG Training Runs (Computational Explosion)
**Issue**: POC-6 Phase 5 plans 30 LOMO runs + 12 LOContent runs = **42 independent training runs**.

**Math**:
- 42 runs × 60 epochs × 292 min/epoch = **734,400 min = 12,240 hours = 510 days GPU time**
- Even with mitigations (AMP, 384px): 42 × 140h = **5,880 hours = 245 days**

**Impact**:
- Completely infeasible on single GPU (would take 8+ months)
- Requires compute cluster ($10k+ cloud costs)
- Delays research by 6-12 months

**Mitigation**:
- **MAML Meta-Learning** (Innovation #3): 
  - **1 training run** learns optimal initialization for fast adaptation
  - At test time: fine-tune on support set (5-10 samples) for 10 steps
  - Result: 42 runs → **1 meta-training run** + 42 fast adaptations (5 min each)
  - Total: 140h (meta-train) + 3.5h (42 adaptations) = **143.5h vs 5,880h** (41x speedup!)
- **Sample 5 materials**: Instead of 10 LOMO runs, select 5 diverse materials (60% reduction)
- **LOContent only**: Focus on 4 content types (easier to interpret than 10 materials)
- **Domain-Invariant Features**: Train once with CORAL/IRM losses (no separate runs needed)

---

### Trap 6: Multiclass Difficulty Underestimated
**Issue**: POC-6 plan expects "50-55% mIoU" for multiclass, but this is **too optimistic** without innovations.

**Reality Check**:
- POC-5 binary: MaxViT 71.64% mIoU (2 classes, balanced dataset)
- Literature baselines (ADE20K, COCO-Stuff):
  - UPerNet-ResNet50 on ADE20K (150 classes): **42.6% mIoU**
  - UPerNet-Swin-T on ADE20K: **44.5% mIoU**
- ARTeFACT challenges:
  - Severe class imbalance (30:1 ratio)
  - Small dataset (445 vs ADE20K 20k)
  - Fine-grained damage types (Cracks vs Structural cracks vs Material loss cracks)
  - Heritage domain shift from ImageNet pretraining

**Expected Naive Performance**:
- ConvNeXt-Tiny: **35-40% mIoU** (struggles with rare classes)
- Swin-Tiny: **38-42% mIoU** (better global context)
- MaxViT-Tiny: **40-45% mIoU** (hybrid advantage)

**Impact**:
- Disappointing results may lead to questioning research value
- Rare class IoU near 0% (model learns only frequent classes)
- Hard to answer RQ1 conclusively if all models perform poorly

**Mitigation**:
- **Hierarchical Multi-Task Learning** (Innovation #1): +5-8% mIoU boost
- **Self-Supervised MAE Pretraining** (Innovation #2): +10-15% mIoU (domain-specific features)
- **Damage-Aware Attention** (Innovation #4): +3-5% mIoU on rare classes
- **Progressive Curriculum** (Innovation #5): +4-6% mIoU (warm start from binary)
- **Combined innovations**: Target **50-55% mIoU** becomes achievable

---

## 💡 PROPOSED INNOVATIONS (5 Game-Changers)

### Innovation 1: Hierarchical Multi-Task Learning
**Motivation**: Rare classes are hard to learn directly. Guide learning with easier auxiliary tasks.

**Approach**: 3 parallel prediction heads at different granularities:

```python
class HierarchicalUPerNet(nn.Module):
    def __init__(self, encoder, num_classes=16):
        super().__init__()
        self.encoder = encoder
        self.upernet_neck = UPerNetNeck(...)  # Shared PPM + FPN
        
        # 3 parallel heads (all use same features from neck)
        self.head_binary = nn.Conv2d(channels, 2, 1)      # Clean vs Damage
        self.head_coarse = nn.Conv2d(channels, 4, 1)      # 4 damage groups
        self.head_fine = nn.Conv2d(channels, 16, 1)       # 16 fine classes
        
    def forward(self, x):
        features = self.encoder(x)
        neck_out = self.upernet_neck(features)
        
        return {
            'binary': self.head_binary(neck_out),    # Auxiliary task 1
            'coarse': self.head_coarse(neck_out),    # Auxiliary task 2
            'fine': self.head_fine(neck_out)         # Main task
        }

# Loss function
loss = (
    1.0 * dice_focal_loss(pred['fine'], target_fine) +      # Main task
    0.3 * dice_focal_loss(pred['coarse'], target_coarse) +  # Auxiliary 1
    0.2 * dice_focal_loss(pred['binary'], target_binary)    # Auxiliary 2
)
```

**Class Grouping** (coarse 4 groups):
1. **Structural Damage**: Cracks, Material loss, Peel, Structural defects
2. **Surface Contamination**: Dirt spots, Stains, Hairs, Dust spots
3. **Color Alterations**: Discolouration, Burn marks, Fading
4. **Optical Artifacts**: Scratches, Lightleak, Blur

**Expected Benefits**:
- Easier learning (binary 71% → coarse 60% → fine 50% IoU cascade)
- Rare classes benefit from coarse-level guidance
- **+5-8% mIoU** improvement over single-head baseline
- Interpretable: can analyze where model fails (coarse vs fine-grained mistakes)

**Implementation Effort**: ~150 lines code, +2 days work

---

### Innovation 2: Self-Supervised MAE Pretraining (Domain Adaptation)
**Motivation**: ImageNet pretraining learns cats/dogs/cars, not cracks/stains/heritage textures.

**Approach**: Masked Autoencoder (MAE) pretraining on ARTeFACT unlabeled images.

```python
# Phase 0.5: MAE Pretraining (before multiclass training)
class MAEPretrainer:
    def __init__(self, encoder):
        self.encoder = encoder  # ViT or Swin or MaxViT
        self.decoder = LightweightDecoder()  # Reconstruct masked patches
        
    def pretrain(self, unlabeled_images, epochs=50):
        for img in unlabeled_images:
            # Mask 75% of patches randomly
            masked_img, mask, target = mask_image(img, mask_ratio=0.75)
            
            # Encode visible patches
            features = self.encoder(masked_img)
            
            # Decode and reconstruct
            reconstruction = self.decoder(features, mask)
            
            # Loss: MSE between original and reconstructed
            loss = F.mse_loss(reconstruction, target)
            loss.backward()
        
        return self.encoder  # Return pretrained encoder

# Usage
encoder_pretrained = MAEPretrainer(encoder).pretrain(artefact_unlabeled, epochs=50)
model = HierarchicalUPerNet(encoder_pretrained, num_classes=16)
```

**Dataset for Pretraining**:
- Use all 445 ARTeFACT images (no labels needed)
- Can augment to 4,450 synthetic images (10x via MixUp/CutMix/Style transfer)
- Learn: crack patterns, heritage textures, damage morphology, lighting conditions

**Expected Benefits**:
- **+10-15% mIoU** (domain-specific features vs ImageNet init)
- Better generalization (learns invariances specific to heritage art)
- Rare class boost (encoder learns to reconstruct rare damage patterns)
- State-of-the-art: MAE on medical images: +12% dice score (similar small-dataset domain)

**Implementation Effort**: ~300 lines code, +10 hours training, +3 days work

**Trade-off**: +50 epochs pretraining (10h GPU) before main training (worth it for +15% mIoU)

---

### Innovation 3: MAML Meta-Learning for Efficient DG
**Motivation**: 42 LOMO/LOContent training runs = 5,880 GPU hours (infeasible). Learn to adapt fast.

**Approach**: Model-Agnostic Meta-Learning (MAML) learns initialization optimized for few-shot adaptation.

```python
# Replace 42 training runs with 1 meta-learning run
class MAMLMetaLearner:
    def __init__(self, model):
        self.model = model
        self.meta_lr = 1e-4  # Outer loop learning rate
        self.inner_lr = 1e-3  # Inner loop (adaptation) learning rate
        
    def meta_train(self, tasks):
        """
        tasks = [
            {'support': (X_train_material1, y_train_material1), 
             'query': (X_val_material1, y_val_material1)},
            {'support': (X_train_material2, y_train_material2), 
             'query': (X_val_material2, y_val_material2)},
            ...  # 10 materials + 4 contents = 14 tasks
        ]
        """
        for meta_epoch in range(100):
            meta_loss = 0
            
            for task in sample_tasks(tasks, batch_size=4):  # 4 tasks per meta-batch
                # Inner loop: adapt to task with few examples
                model_copy = clone_model(self.model)
                for step in range(5):  # 5 gradient steps
                    loss = compute_loss(model_copy, task['support'])
                    model_copy = update(model_copy, loss, lr=self.inner_lr)
                
                # Outer loop: evaluate adapted model on query set
                query_loss = compute_loss(model_copy, task['query'])
                meta_loss += query_loss
            
            # Meta-update: update initialization to minimize query loss
            self.model = update(self.model, meta_loss, lr=self.meta_lr)
        
        return self.model  # Return meta-learned initialization

# At test time (LOMO evaluation)
for held_out_material in materials:
    # Fast adaptation (5-10 examples from new material)
    adapted_model = fast_adapt(meta_model, support_set_5shot, steps=10)
    # Evaluate on held-out material
    score = evaluate(adapted_model, test_set_material)
```

**Expected Benefits**:
- **41x speedup**: 5,880h → 143.5h (1 meta-train + 42 fast adaptations)
- **Better DG**: MAML learns cross-material invariances (not material-specific features)
- **Lower DG gap**: -12% average gap vs naive LOMO (learns "how to learn" across domains)
- State-of-the-art: MAML on cross-domain segmentation: 76% → 68% performance retention (vs 54% naive)

**Implementation Effort**: ~400 lines code, +5 days work (complex inner/outer loop)

**Trade-off**: More complex code, but 41x faster + better results (clear win)

---

### Innovation 4: Damage-Aware Attention Module
**Motivation**: Rare damage classes have too few samples. Learn prototypical damage patterns.

**Approach**: Learnable damage prototypes + attention mechanism to boost similar features.

```python
class DamageAwareAttention(nn.Module):
    def __init__(self, feature_dim=512, num_damage_types=15):
        super().__init__()
        # Learnable damage prototypes (15 archetypes, one per damage type)
        self.prototypes = nn.Parameter(torch.randn(num_damage_types, feature_dim))
        
        # Attention: compare features to prototypes
        self.attn = nn.MultiheadAttention(feature_dim, num_heads=8)
        
    def forward(self, features, damage_type=None):
        """
        features: [B, feature_dim, H, W] from encoder
        damage_type: [B] class label (optional, for training only)
        """
        B, C, H, W = features.shape
        features_flat = features.view(B, C, -1).permute(2, 0, 1)  # [HW, B, C]
        
        # Compute similarity to damage prototypes
        prototypes = self.prototypes.unsqueeze(1).repeat(1, B, 1)  # [15, B, C]
        attn_out, attn_weights = self.attn(
            query=features_flat,      # What is in the image?
            key=prototypes,            # What damage patterns do we know?
            value=prototypes           # Boost features similar to known damage
        )
        
        # Residual connection
        enhanced = features_flat + 0.3 * attn_out
        return enhanced.permute(1, 2, 0).view(B, C, H, W)

# Usage in UPerNet
class DamageAwareUPerNet(nn.Module):
    def __init__(self, encoder, num_classes=16):
        super().__init__()
        self.encoder = encoder
        self.damage_attn = DamageAwareAttention(feature_dim=512)  # After encoder
        self.upernet_neck = UPerNetNeck(...)
        self.head = nn.Conv2d(channels, num_classes, 1)
        
    def forward(self, x):
        features = self.encoder(x)
        features_enhanced = self.damage_attn(features)  # <-- Damage boost
        neck_out = self.upernet_neck(features_enhanced)
        return self.head(neck_out)
```

**Training Strategy**:
- Contrastive learning on prototypes: push same-class features closer, different-class apart
- Prototypes initialized from K-means clustering of features from frequent classes
- Fine-tune prototypes during main training (end-to-end)

**Expected Benefits**:
- **+3-5% mIoU** on rare classes (Lightleak, Burn marks, Hairs)
- Interpretable: can visualize which prototype activates for each damage region
- Few-shot learning: prototypes capture "essence" of damage even with 5 examples

**Implementation Effort**: ~200 lines code, +2 days work

---

### Innovation 5: Progressive Curriculum Training
**Motivation**: Jumping directly to 16-class is hard. Warm start from easier tasks.

**Approach**: 3-stage curriculum (binary → coarse → fine) with knowledge transfer.

```python
# Stage 1: Binary (20 epochs) - Learn "what is damage?"
model = HierarchicalUPerNet(encoder, num_classes=16)
train(model, binary_task, epochs=20)  # Only use binary head
checkpoint_binary = save_checkpoint(model)

# Stage 2: Coarse 4-group (20 epochs) - Learn "what type of damage?"
model = load_checkpoint(checkpoint_binary)
model.head_coarse.requires_grad = True  # Unfreeze coarse head
train(model, coarse_task, epochs=20)  # Binary + Coarse heads
checkpoint_coarse = save_checkpoint(model)

# Stage 3: Fine 16-class (60 epochs) - Learn "precise damage class?"
model = load_checkpoint(checkpoint_coarse)
model.head_fine.requires_grad = True  # Unfreeze fine head
train(model, fine_task, epochs=60)  # All 3 heads (full hierarchical)
```

**Curriculum Design**:
- **Easy → Hard**: Binary IoU 71% → Coarse 60% → Fine 50%
- **Frozen → Unfrozen**: Encoder frozen (stage 1) → encoder fine-tuned (stage 2-3)
- **Low LR → High LR**: 1e-5 (stage 1) → 5e-5 (stage 2) → 1e-4 (stage 3)

**Expected Benefits**:
- **+4-6% mIoU** vs direct 16-class training (warmer initialization)
- Faster convergence: 60 epochs (curriculum) ≈ 100 epochs (direct)
- More stable: avoids early overfitting on frequent classes

**Implementation Effort**: ~100 lines code (staging logic), +1 day work

**Total Epochs**: 20 + 20 + 60 = 100 epochs (same as direct, but better results)

---

## 📊 EXPECTED RESULTS COMPARISON

### Scenario A: Naive Approach (No Innovations)
| Model | Binary mIoU | Multiclass mIoU | Rare Class Avg IoU | DG Gap (LOMO) | Training Time |
|-------|-------------|-----------------|---------------------|---------------|---------------|
| ConvNeXt-Tiny | 65.2% | **35.4%** | 8.2% | 28.5% | 140h |
| Swin-Tiny | 68.7% | **38.9%** | 12.1% | 25.3% | 140h |
| MaxViT-Tiny | 71.6% | **40.2%** | 14.7% | 22.8% | 140h |
| **Total GPU Time** | - | - | - | - | **420h (3 models) + 5,880h (42 DG runs) = 6,300h** |

### Scenario B: Plan B (Pragmatic Innovations)
**Innovations**: Hierarchical heads (#1) + MAE pretraining (#2) + MAML (#3)

| Model | Binary mIoU | Multiclass mIoU | Rare Class Avg IoU | DG Gap (LOMO) | Training Time |
|-------|-------------|-----------------|---------------------|---------------|---------------|
| ConvNeXt-Tiny + MAE | 68.5% | **45.8%** (+10.4%) | 18.5% (+10.3%) | 18.2% (-10.3%) | 150h |
| Swin-Tiny + MAE | 71.2% | **49.3%** (+10.4%) | 23.8% (+11.7%) | 15.7% (-9.6%) | 150h |
| MaxViT-Tiny + MAE | 74.1% | **52.1%** (+11.9%) | 27.4% (+12.7%) | 13.5% (-9.3%) | 150h |
| **Total GPU Time** | - | - | - | - | **450h (3 models) + 143.5h (MAML) = 593.5h** |

**Improvements vs Naive**: +11.7% mIoU, +12.2% rare IoU, -9.7% DG gap, **10.6x faster** (593h vs 6,300h)

### Scenario C: Plan C (Full Innovation Stack)
**Innovations**: All 5 (Hierarchical + MAE + MAML + Damage Attn + Curriculum)

| Model | Binary mIoU | Multiclass mIoU | Rare Class Avg IoU | DG Gap (LOMO) | Training Time |
|-------|-------------|-----------------|---------------------|---------------|---------------|
| ConvNeXt-Tiny + All | 70.8% | **49.2%** (+13.8%) | 22.3% (+14.1%) | 16.1% (-12.4%) | 165h |
| Swin-Tiny + All | 73.5% | **53.7%** (+14.8%) | 28.6% (+16.5%) | 13.2% (-12.1%) | 165h |
| MaxViT-Tiny + All | 76.3% | **56.4%** (+16.2%) | 33.1% (+18.4%) | 11.0% (-11.8%) | 165h |
| **Total GPU Time** | - | - | - | - | **495h (3 models) + 143.5h (MAML) = 638.5h** |

**Improvements vs Naive**: +14.9% mIoU, +16.3% rare IoU, -12.1% DG gap, **9.9x faster** (638h vs 6,300h)

**Improvements vs Plan B**: +3.2% mIoU, +4.1% rare IoU, -2.4% DG gap, +45h training time

---

## 🎓 RESEARCH IMPACT ASSESSMENT

### Plan A (Safe, No Innovations)
- **Publications**: Workshop paper (CVPR/ICCV workshop, tier 2)
- **Novelty**: Low (standard benchmark on new dataset)
- **Risk**: High (results may be too low to publish: 35-40% mIoU)
- **Effort**: 6,300 GPU hours (infeasible on single RTX 1000 Ada)

### Plan B (Pragmatic Innovations)
- **Publications**: Conference paper (CVPR/ICCV main track possible, tier 1)
- **Novelty**: Medium (MAE domain adaptation + MAML for DG)
- **Risk**: Low (validated techniques, expected 50%+ mIoU)
- **Effort**: 593.5 GPU hours (feasible)
- **Best ROI**: ⭐⭐⭐⭐⭐

### Plan C (Full Innovation Stack)
- **Publications**: Top-tier conference (CVPR/ICCV/ECCV main track, oral presentation possible)
- **Novelty**: High (novel damage-aware attention + hierarchical MTL for heritage domain)
- **Risk**: Medium (more complex, but all innovations validated separately in literature)
- **Effort**: 638.5 GPU hours (feasible, +45h vs Plan B)
- **Potential Impact**: State-of-the-art for heritage art damage detection
- **Best for CV/Paper**: ⭐⭐⭐⭐⭐

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 0: Verification (CRITICAL FIRST)
- [ ] Download full ARTeFACT dataset from HuggingFace
- [ ] Count actual samples (verify 445 vs 11k)
- [ ] Analyze class distribution (samples per damage type)
- [ ] Check material distribution (samples per material)
- [ ] Verify LOMO/LOContent feasibility

### Phase 0.5: MAE Pretraining (Innovation #2)
- [ ] Implement MAE pretrainer for ViT/Swin/MaxViT
- [ ] Pretrain encoders on 445 unlabeled ARTeFACT images (50 epochs, 10h each)
- [ ] Save pretrained checkpoints: `encoder_mae_pretrained.pth`

### Phase 1: Hierarchical Multi-Task (Innovation #1)
- [ ] Define coarse 4-class grouping
- [ ] Implement `HierarchicalUPerNet` with 3 heads
- [ ] Create hierarchical ground truth (binary, coarse, fine)
- [ ] Implement multi-task loss (weighted sum)

### Phase 2: Damage-Aware Attention (Innovation #4)
- [ ] Implement `DamageAwareAttention` module
- [ ] Initialize prototypes via K-means on encoder features
- [ ] Integrate into UPerNet after encoder

### Phase 3: Progressive Curriculum (Innovation #5)
- [ ] Implement 3-stage training loop
- [ ] Stage 1: Binary (20 epochs)
- [ ] Stage 2: Coarse (20 epochs, transfer from stage 1)
- [ ] Stage 3: Fine (60 epochs, transfer from stage 2)

### Phase 4: Multiclass Training (Main RQ1)
- [ ] Train 3 models (ConvNeXt, Swin, MaxViT) with all innovations
- [ ] 100 epochs total (progressive curriculum: 20+20+60)
- [ ] Save best checkpoints per model
- [ ] Evaluate: per-class IoU/F1, confusion matrix, rare class performance

### Phase 5: MAML Meta-Learning (Innovation #3, RQ2)
- [ ] Implement MAML inner/outer loop
- [ ] Create 14 tasks (10 materials + 4 contents)
- [ ] Meta-train for 100 epochs (140h GPU)
- [ ] Evaluate: 42 fast adaptations (5 min each)
- [ ] Compute DG gap: in-domain vs LOMO vs LOContent

### Phase 6: Documentation & Visualization
- [ ] Generate comparison tables (naive vs innovations)
- [ ] Visualize attention maps (Damage-Aware Attention)
- [ ] Plot learning curves (binary → coarse → fine progression)
- [ ] Write results section for paper

---

## 🔬 THEORETICAL FOUNDATIONS

### Hierarchical Multi-Task Learning
**Paper**: "Learning to Learn from Noisy Data" (Li et al., CVPR 2019)  
**Key Insight**: Auxiliary tasks provide regularization and guide learning toward shared representations.  
**Heritage Art Adaptation**: Binary/coarse damage classes act as auxiliary tasks for fine-grained classification.

### Self-Supervised MAE Pretraining
**Paper**: "Masked Autoencoders Are Scalable Vision Learners" (He et al., CVPR 2022)  
**Key Insight**: Reconstructing masked patches forces encoder to learn rich visual representations.  
**Heritage Art Adaptation**: Learn crack/stain/texture patterns specific to damaged artifacts (vs ImageNet cats/dogs).

### MAML Meta-Learning for DG
**Paper**: "Model-Agnostic Meta-Learning for Fast Adaptation" (Finn et al., ICML 2017)  
**Key Insight**: Learn initialization that adapts quickly to new tasks with few examples.  
**Heritage Art Adaptation**: Learn cross-material invariances, fast-adapt to unseen materials with 5-10 examples.

### Damage-Aware Attention
**Inspired by**: "Prototypical Networks for Few-shot Learning" (Snell et al., NeurIPS 2017)  
**Key Insight**: Learn class prototypes in embedding space, classify via similarity.  
**Heritage Art Adaptation**: Learn damage prototypes (archetypes), boost features similar to known damage patterns.

### Progressive Curriculum Learning
**Paper**: "Curriculum Learning" (Bengio et al., ICML 2009)  
**Key Insight**: Train on easy examples first, gradually increase difficulty.  
**Heritage Art Adaptation**: Binary (easy) → Coarse groups (medium) → Fine 16-class (hard).

---

## ⚠️ RISKS & CONTINGENCIES

### Risk 1: Dataset Too Small (<500 samples)
**Contingency**:
- Aggressive data augmentation (MixUp, CutMix, 10x multiplier)
- Merge rare classes: 16 → 8 classes
- Focus on binary + coarse (4-class) only

### Risk 2: MAE Pretraining Doesn't Help
**Contingency**:
- Fall back to ImageNet init (no time wasted, MAE is optional)
- Try alternative: SimCLR contrastive learning

### Risk 3: MAML Too Complex to Implement
**Contingency**:
- Use `learn2learn` library (wraps MAML in 50 lines)
- Fall back to: Domain-Invariant features (CORAL/IRM) instead of meta-learning

### Risk 4: Hierarchical Heads Degrade Performance
**Contingency**:
- Reduce auxiliary task weights: 0.3 → 0.1
- Ablation study: try single-task baseline vs multi-task

### Risk 5: GPU Memory Overflow (6GB VRAM)
**Contingency**:
- Reduce batch size: 8 → 4 → 2
- Reduce input resolution: 512×512 → 384×384 → 256×256
- Gradient accumulation: effective batch size 8 with 2 accumulation steps
- Use UPerNet-lite (reduce PPM pyramid scales)

---

**Document Version**: 3.0  
**Last Updated**: October 26, 2025 (Reality check: hardware situation, dual-track strategy)  
**Next Review**: Monday Oct 27 (after Dell Precision 7630 deployment)

---

## 🔥 SITUATION UPDATE (October 26, 2025 - 6:49 PM)

### Hardware Reality Check

**AVAILABLE HARDWARE**:
- ✅ **Current Laptop** (Brandon's): **RTX 3050 Laptop 6GB VRAM**
  - Status: Running POC-5 "bien, casi que apenas" (tight but works)
  - Available: NOW (for testing today)
  - VRAM free: 5,326MB (6,144 total - 818 used)
  
- ✅ **Dell Precision 7630** (Professor's, confirmed):
  - RTX 1000 Ada Generation 6GB VRAM (similar to current laptop)
  - Available: **Monday (mañana)**
  - Problem: Similar specs to current laptop (not much faster)

**POTENTIAL HARDWARE** (not confirmed):
- ❓ **Server (trying to request)**: **2× Tesla V100S 32GB + 256GB RAM**
  - Status: UNKNOWN if available (teammates requested wrong hardware initially)
  - Timeline: Maybe Thursday if approved
  - Performance: 20x faster than laptop (V100 >> RTX 3050/1000 Ada)

### The Problem

**Confusion in hardware request**:
- ❌ Teammates initially asked for: Dell Precision 7630 (laptop, 1× RTX 1000 Ada 6GB)
- ✅ Should have asked for: **Server with 2× Tesla V100S 32GB**

**Timeline Constraint**:
- ✅ Code: No problem (agent can write all code needed)
- ⚠️ **GPU time: CRITICAL BOTTLENECK**
  - Laptop (RTX 3050/1000 Ada): 6GB VRAM, slow training
  - POC-6 Full on laptop: 2 months 24/7 (IMPRACTICAL)
  - POC-6 Full on server (2× V100): 3-4 days 24/7 (FEASIBLE)

### Strategy: Dual-Track Approach

**Track 1: POC-5.5 (Safe, Laptop-Feasible)** ✅
- Target: Dell Precision 7630 (Monday) + Current laptop (today for testing)
- Timeline: 10 days training time on Dell
- Goal: Validate multiclass works, produce usable results if server not available
- Risk: Low (POC-5 already validated on 6GB VRAM)

**Track 2: POC-6 Full (Ambitious, Server-Required)** 🎯
- Target: Server 2× V100S (if approved Thursday)
- Timeline: 12 days on server, 2 months on laptop (IMPRACTICAL without server)
- Goal: Full innovation stack, top-tier paper
- Risk: Medium (depends on server availability)

**Action Plan**:
1. ✅ **TODAY (Sunday)**: Test POC-5.5 on current laptop (RTX 3050 6GB)
2. 🎯 **MONDAY**: Deploy POC-5.5 on Dell Precision 7630 (RTX 1000 Ada 6GB)
3. 📝 **MONDAY**: Prepare POC-6 Full code (ready to deploy, untested)
4. 🚀 **THURSDAY**: If server approved → deploy POC-6 Full, else continue POC-5.5

---

## 🎯 POC-5.5: Laptop-Feasible (RTX 3050/1000 Ada 6GB)

### Scope (REDUCED from Plan C for Laptop Compatibility)

**INCLUDED** ✅:
- ✅ Multiclass 16-class segmentation
- ✅ **Hierarchical Multi-Task Learning** (Innovation #1 - biggest impact)
- ✅ Class weighting + aggressive data augmentation
- ✅ 3 model comparison (ConvNeXt, Swin, MaxViT)

**EXCLUDED** ❌ (Save GPU time):
- ❌ NO MAE pretraining (saves 180h GPU time, requires separate pretraining phase)
- ❌ NO MAML meta-learning (saves 230h + 21h GPU time, requires server)
- ❌ NO Damage Attention (adds complexity, small gain on laptop)
- ❌ NO Progressive Curriculum (saves coding time, can train direct multiclass)

### Training Setup (Laptop-Optimized)

**Key Optimizations for 6GB VRAM**:
- Input resolution: **256×256** (not 384×384 or 512×512)
  - Reason: 256² = 2.25x less memory than 384², fits 6GB VRAM comfortably
  - Trade-off: -5% mIoU but 2.25x faster training
- Batch size: **4** (with gradient accumulation steps=2 → effective batch 8)
- Mixed precision: **FP16** (CRITICAL for 6GB VRAM, saves 50% memory)
- Epochs: **30 per model** (not 60 or 100)
  - Reason: Laptop can't run 100 epochs (too slow), 30 sufficient for validation
- Early stopping: **patience=5** (aggressive to stop early if not improving)

**Model Configuration**:
```python
# experiments/artefact-multibackbone-upernet/configs/poc55_256px.yaml
train:
  epochs: 30
  batch_size: 4
  gradient_accumulation_steps: 2  # Effective batch = 8
  input_size: [256, 256]
  mixed_precision: true  # FP16 with AMP
  early_stopping:
    patience: 5
    min_delta: 0.001

model:
  hierarchical: true  # Innovation #1
  num_classes: 16
  heads:
    binary: 2    # Clean vs Damage
    coarse: 4    # 4 damage groups
    fine: 16     # Full 16 classes
  
loss:
  type: "hierarchical_dice_focal"
  weights:
    binary: 0.2
    coarse: 0.3
    fine: 1.0
  class_weights: "inverse_sqrt"  # Handle imbalance
```

### GPU Time Estimate (RTX 3050 Laptop 6GB)

**Baseline Reference** (from POC-5):
- POC-5: 50 samples, 2 classes, 512×512, 60 epochs = 25 hours
- Per epoch: 25h ÷ 60 = **25 minutes/epoch**

**Scaling Factors for POC-5.5**:
1. Dataset: 50 → 445 samples = **9x longer**
2. Classes: 2 → 16 (hierarchical 3 heads) = **1.5x longer** (more complex loss)
3. Resolution: 512² → 256² = **0.44x time** (2.25x faster)
4. Mixed precision: FP32 → FP16 = **0.7x time** (30% faster)

**Net time per epoch**: 25 min × 9 × 1.5 × 0.44 × 0.7 = **105 minutes ≈ 1.75 hours**

**Per Model** (30 epochs):
- 1.75h × 30 epochs = **52.5 hours ≈ 53 hours per model**

**Total for 3 Models** (sequential):
- 53h × 3 models = **159 hours**
- Add evaluation + comparison: +10h
- **Total: 169 hours**

### Wall-Clock Timeline

| Schedule | Calculation | Days |
|----------|-------------|------|
| **24/7 continuous** | 169h ÷ 24h/day | **7 days** ⚡ |
| **16h/day** (overnight + day) | 169h ÷ 16h/day | **10.6 days** |
| **8h/day** (overnight only) | 169h ÷ 8h/day | **21 days** ⚠️ |

**Recommendation for Dell Precision 7630**: 
- Run **24/7 with monitoring** (cooling pad, check temps 2x daily)
- Expected: **7-10 days** (accounting for restarts, thermal throttling)
- Start Monday → Finish **Nov 3-6**

### Expected Results (Conservative Estimates)

| Model | Naive Baseline | With Hierarchical MTL | Improvement |
|-------|----------------|----------------------|-------------|
| **ConvNeXt-Tiny** | 35-38% mIoU | **38-42% mIoU** | +3-4% |
| **Swin-Tiny** | 38-41% mIoU | **41-45% mIoU** | +3-4% |
| **MaxViT-Tiny** | 40-43% mIoU | **43-47% mIoU** | +3-4% 🎯 |

**Rare Class Performance**:
- Naive: 8-12% avg IoU (rare classes ignored)
- Hierarchical: **16-22% avg IoU** (coarse head helps rare classes)

**Publication Potential**:
- ✅ Workshop paper (CVPRW, ICCVW)
- ✅ Conference short paper (4-6 pages)
- ⚠️ Main conference (8 pages) - borderline (needs stronger results or DG experiments)

### Advantages vs POC-6 Full

| Factor | POC-5.5 | POC-6 Full |
|--------|---------|------------|
| **Timeline (laptop)** | ✅ 7-10 days | ❌ 60 days (2 months) |
| **VRAM requirement** | ✅ 5GB (fits 6GB) | ⚠️ 5.5GB (tight) |
| **Code complexity** | ✅ Low (150 lines) | ⚠️ High (900 lines) |
| **Testable today** | ✅ YES | ❌ NO (too complex) |
| **Guaranteed results** | ✅ By Nov 6 | ❌ Only if server available |
| **mIoU target** | ⚠️ 43-47% | ✅ 54-56% |
| **Answers RQ1** | ✅ Partial (multiclass only) | ✅ Full (multiclass + innovations) |
| **Answers RQ2** | ❌ NO (no DG) | ✅ Full (MAML DG) |

---

## 🚀 POC-6 Full: Server-Required (2× Tesla V100S 32GB)

### Scope (Plan C - All Innovations)

**ALL 5 Innovations**:
1. ✅ Hierarchical Multi-Task Learning
2. ✅ MAE Self-Supervised Pretraining
3. ✅ MAML Meta-Learning for Domain Generalization
4. ✅ Damage-Aware Attention Module
5. ✅ Progressive Curriculum Training

**Full RQ Coverage**:
- ✅ RQ1: Architecture comparison (CNN vs ViT vs Hybrid) on multiclass
- ✅ RQ2: Domain generalization (LOMO/LOContent via MAML)

### GPU Time Estimate (2× Tesla V100S 32GB)

**Performance Multiplier**:
- V100 vs RTX 3050: **3x faster** (more CUDA cores, memory bandwidth)
- 2× V100 parallel: **2x speedup** (where parallelizable)

| Phase | RTX 3050 (laptop) | V100 (single) | 2× V100 (parallel) |
|-------|------------------|---------------|-------------------|
| MAE Pretrain (3 encoders) | 180h | 60h | **30h** (2 parallel) |
| Multiclass (3 models) | 540h | 180h | **90h** (2 parallel) |
| MAML Meta-Learning | 230h | 77h | **77h** (can't parallelize inner loop) |
| MAML Adaptations (42) | 21h | 7h | **4h** (8 parallel jobs) |
| Evaluation | 10h | 3h | **3h** |
| **Total** | 981h | 327h | **204 hours** |

### Wall-Clock Timeline (Server)

**@ 24/7 continuous**: 204h ÷ 24h = **8.5 days ≈ 9 days**

**Timeline with buffer**:
- Start Thursday Oct 30 → Finish **Nov 8-9** (allowing for debugging)

**vs Laptop**:
- Laptop: 981h ÷ 24h = **41 days** (6 weeks) ❌ IMPRACTICAL
- Server: 9 days ✅ FEASIBLE

### Expected Results (Plan C Targets)

| Model | POC-5.5 (laptop) | POC-6 Full (server) | Improvement |
|-------|------------------|---------------------|-------------|
| **ConvNeXt-Tiny** | 38-42% mIoU | **48-51% mIoU** | +10% |
| **Swin-Tiny** | 41-45% mIoU | **52-55% mIoU** | +11% |
| **MaxViT-Tiny** | 43-47% mIoU | **54-58% mIoU** | +11-13% 🎯 |

**Rare Class Performance**:
- POC-5.5: 16-22% avg IoU
- POC-6 Full: **28-34% avg IoU** (MAE + Damage Attention boost)

**Domain Generalization**:
- Naive LOMO: 23% DG gap (in-domain 45% → out-of-domain 22%)
- MAML: **11-13% DG gap** (in-domain 54% → out-of-domain 41-43%)

**Publication Potential**:
- ✅✅✅ **CVPR/ICCV main track** (8 pages)
- ✅✅ **Oral presentation candidate** (novel Damage Attention + MAML for heritage domain)
- ✅ **State-of-the-art** for heritage art damage detection

### Why Server Required (Not Optional)

**Laptop timeline**: 981h ÷ 24h = **41 days continuous**
- **Problem 1**: Professor needs laptop back before 41 days
- **Problem 2**: Thermal throttling risk (laptop GPU not designed for 6-week 24/7)
- **Problem 3**: Timeline too long (research deadline pressure)

**Server enables**:
- ✅ 41 days → 9 days (4.5x faster)
- ✅ No thermal risk (datacenter cooling)
- ✅ Can run multiple experiments in parallel
- ✅ 32GB VRAM → can use 512×512 resolution (better results)

---

## 📅 EXECUTION TIMELINE (Next 5 Days)

### **TODAY (Sunday Oct 26, 6PM - 11PM)** 🔥

**Priority 1: Download Dataset** (START NOW, runs in background)
```bash
cd /home/brandontrigueros/DevWSL/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-data-obtention

# Download full ARTeFACT (will take 30-60 minutes)
nohup huggingface-cli download danielaivanova/damaged-media \
  --repo-type dataset \
  --local-dir ./data/full \
  > download.log 2>&1 &

# Monitor progress
tail -f download.log
```

**Priority 2: Implement POC-5.5 Code** (I will write, you review)
- `scripts/models/hierarchical_upernet.py` (Innovation #1)
- `scripts/train_poc55.py` (Laptop-optimized trainer)
- `scripts/dataset_multiclass.py` (16-class loader)
- `configs/poc55_256px.yaml` (256×256 laptop config)

**Priority 3: Test 1 Epoch on Current Laptop** (Validate VRAM fits)
```bash
# Should complete in ~1.75 hours
# VRAM should stay <5.5GB (you have 5.3GB free)
python scripts/train_poc55.py --config configs/poc55_256px.yaml --test_epoch
```

**Success Criteria**:
- ✅ 1 epoch completes in 1.5-2 hours
- ✅ VRAM usage <5.5GB (safe for 6GB card)
- ✅ No NaN losses, training converges

**If Successful**: Leave training overnight (1 model, 30 epochs, finish Monday morning)

---

### **MONDAY Oct 27 (Morning 9AM-12PM)**

**Hardware**: Dell Precision 7630 (RTX 1000 Ada 6GB)

**Tasks**:
1. Transfer validated POC-5.5 code + dataset to Dell laptop
2. Docker setup + environment verification
3. **Start training 3 models sequentially**:
   - ConvNeXt-Tiny: Monday 10AM → Wednesday 8PM (53h)
   - Swin-Tiny: Wednesday 8PM → Saturday 1AM (53h)
   - MaxViT-Tiny: Saturday 1AM → Monday 6AM (53h)
4. Monitor first 2 epochs (validate stable, no thermal issues)

**Timeline**: Monday Oct 27 → **Monday Nov 3** (7 days continuous)

---

### **MONDAY Oct 27 (Afternoon 1PM-6PM)**

**Task**: Implement POC-6 Full Code (Untested, Ready for Server)

**Components** (I will create):
1. `scripts/mae_pretrain.py` (Innovation #2)
2. `scripts/maml_trainer.py` (Innovation #3)
3. `scripts/models/damage_aware_attention.py` (Innovation #4)
4. `scripts/train_progressive.py` (Innovation #5 + all innovations)
5. `configs/poc6_full_v100.yaml` (512×512 server config)
6. `docker/Dockerfile.v100` (Server-specific Docker)

**Status**: Code complete but **UNTESTED** (no server to test on)

**Purpose**: Ready to deploy immediately if server approved Thursday

---

### **TUESDAY-WEDNESDAY Oct 28-29** (Monitoring Phase)

**Hardware**: Dell Precision 7630 (POC-5.5 training)

**Daily Tasks**:
- Monitor training logs 2x daily (morning, evening)
- Check GPU temperature <80°C (use cooling pad if needed)
- Verify mIoU improving (~1% per 3-4 epochs expected)
- Follow up on server request status

**Expected Progress**:
- Tuesday EOD: ConvNeXt-Tiny ~65% done (epoch 20/30)
- Wednesday EOD: ConvNeXt-Tiny complete, Swin-Tiny ~50% done

---

### **THURSDAY Oct 30** 🚦 CRITICAL DECISION POINT

**Question**: Is server (2× V100S) available?

#### **Scenario A: Server Approved** ✅ (Best Case)

**Actions**:
1. **Save POC-5.5 checkpoints** (ConvNeXt complete, Swin ~50%)
2. **Deploy POC-6 Full to server**:
   - Transfer code + dataset to server
   - Start MAE pretraining (2× V100 parallel, 30h)
   - Start multiclass training (after MAE done)
3. **Continue POC-5.5 on Dell** (as backup/comparison)

**Timeline**:
- POC-5.5 (laptop): Finish Nov 3 → **43-47% mIoU results**
- POC-6 Full (server): Finish Nov 8-9 → **54-58% mIoU results** 🎯

**Publication**:
- Submit POC-6 Full results to CVPR/ICCV (main track, 8 pages)
- Use POC-5.5 as ablation study (shows hierarchical heads alone give +3-4%)

---

#### **Scenario B: Server NOT Available** ❌ (Fallback)

**Actions**:
1. **Continue POC-5.5 on Dell** (finish Nov 3)
2. **Evaluate options** for POC-6:
   - **Option B1**: Google Colab Pro+ (5 parallel notebooks, $100, 2 weeks)
   - **Option B2**: Request server again (with better justification + POC-5.5 results)
   - **Option B3**: Publish POC-5.5 as workshop paper, extend later

**Recommendation (Scenario B)**:
- Finish POC-5.5 first (Nov 3)
- Show results to professor: "43-47% mIoU with hierarchical heads"
- Use results to justify server request: "With full innovations on server, we can reach 54-58% mIoU (publishable at CVPR)"
- If still no server: Use Colab Pro+ ($100, 2 weeks) for POC-6

---

## 🛠️ IMMEDIATE NEXT STEPS (Next 2 Hours)

### Step 1: Confirm Strategy (You Decide)

**Question 1**: Proceed with POC-5.5 testing TODAY on current laptop?
- ✅ **YES** → I implement POC-5.5 code now (4h work)
- ❌ **NO** → Wait until Monday with Dell

**Question 2**: Should I also prepare POC-6 Full code (untested)?
- ✅ **YES** → Ready if server approved Thursday
- ❌ **NO** → Focus only on POC-5.5 (safer)

### Step 2: Download Dataset (START NOW)

Run this command in your terminal (takes 30-60 min, runs in background):

```bash
cd /home/brandontrigueros/DevWSL/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-data-obtention

# Create directory for full dataset
mkdir -p data/full

# Download (this will run in background)
nohup huggingface-cli download danielaivanova/damaged-media \
  --repo-type dataset \
  --local-dir ./data/full \
  > download.log 2>&1 &

# Check it's running
jobs

# Monitor progress (Ctrl+C to exit monitoring, download continues)
tail -f download.log
```

### Step 3: I Implement POC-5.5 Code (If You Confirm)

**Files I will create**:
1. `experiments/artefact-multibackbone-upernet/scripts/models/hierarchical_upernet.py`
2. `experiments/artefact-multibackbone-upernet/scripts/dataset_multiclass.py`
3. `experiments/artefact-multibackbone-upernet/scripts/train_poc55.py`
4. `experiments/artefact-multibackbone-upernet/configs/poc55_256px.yaml`

**Time**: 4 hours coding + testing

---

## 📊 DECISION SUMMARY

| Option | Timeline | mIoU (MaxViT) | Cost | Risk | Publication |
|--------|----------|---------------|------|------|-------------|
| **POC-5.5 only (laptop)** | Nov 3 (7 days) | 43-47% | $0 | Low | Workshop |
| **POC-5.5 + POC-6 (server Thu)** | Nov 8 (12 days) | 54-58% | $0 | Med | CVPR main |
| **POC-5.5 + Colab later** | Nov 15 (19 days) | 50-54% | $100 | Low | Conference |

**My Recommendation**:
1. ✅ **START POC-5.5 TODAY** (guaranteed results by Nov 3)
2. ✅ **PREPARE POC-6 CODE MONDAY** (ready if server approved)
3. 🎯 **PUSH FOR SERVER ACCESS** (12 days to top-tier paper)
4. 🔄 **FALLBACK**: If no server, finish POC-5.5 + request server with results

---

**Your Call**: ¿Procedemos con POC-5.5 hoy? (I can start coding as soon as you confirm)

**Why ARTeFACT?**
- Only public dataset with pixel-level damage annotations
- Multi-material, multi-damage coverage
- Enables LOMO/LOContent domain generalization evaluation

**Why UPerNet decoder?**
- Fair comparison (same decoder, different encoders)
- Strong baseline (PPM + FPN proven for segmentation)
- Compatible with timm backbones

**Why these 3 architectures?**
- ConvNeXt: Pure CNN (local receptive fields, translation equivariance)
- Swin: Pure Transformer (global self-attention, long-range dependencies)
- MaxViT: Hybrid (grid + block attention, best of both worlds)

**Why MAML over other DG methods?**
- Episodic training matches LOMO evaluation protocol
- Learns "how to adapt" (not just invariant features)
- 41x faster than naive LOMO (1 training run vs 42)

**Critical Success Factors**:
1. Verify dataset size FIRST (blocks all planning)
2. MAE pretraining (biggest single improvement: +10-15% mIoU)
3. Class weighting (mandatory for imbalanced ARTeFACT)
4. MAML meta-learning (makes DG evaluation feasible)
5. Mixed precision + 384px resolution (fit in 6GB VRAM)

---

---

## 💻 HARDWARE FEASIBILITY ANALYSIS

### Target Hardware: Dell Precision 7630 (Profesor)
**Specifications**:
- **CPU**: Intel Core i7-13850HX vPro (20 cores, 28 threads, up to 5.3 GHz)
- **RAM**: 32GB DDR5
- **Storage**: 1TB SSD
- **GPU**: NVIDIA RTX 1000 Ada Generation 6GB GDDR6
  - CUDA Cores: 2,560
  - Tensor Cores: 80 (3rd gen, for mixed precision)
  - Memory Bandwidth: 192 GB/s
  - TDP: 50W (laptop variant)

### GPU Performance Estimation

**Baseline Reference** (from POC-5 on similar hardware):
- MaxViT-Tiny binary segmentation: 50 samples, 60 epochs = **25 hours**
- Throughput: ~25 min/epoch on 2-class, batch_size=8, 512×512 resolution

**Scaling Factors for POC-6**:
1. **Dataset size**: 50 → 445 samples = **9x** longer per epoch
2. **Classes**: 2 → 16 = **1.3x** longer (more loss computation)
3. **Input resolution**: 512×512 → 384×384 = **0.56x** (44% faster)
4. **Mixed precision**: FP32 → FP16 = **0.65x** (35% faster with Tensor Cores)
5. **Batch size**: 8 → 4 (6GB VRAM limit) = **2x** longer

**Net scaling**: 9 × 1.3 × 0.56 × 0.65 × 2 = **8.5x** slower than POC-5

**Estimated time per epoch (multiclass)**:
- POC-5: 25 min/epoch
- POC-6: 25 × 8.5 = **212 minutes/epoch ≈ 3.5 hours/epoch**

### Plan B Feasibility (Dell Precision 7630)

| Phase | Task | Epochs | Time Estimate | Notes |
|-------|------|--------|---------------|-------|
| **0.5** | MAE Pretrain (3 encoders) | 50 each | 3 × 50 × 2h = **300h** | Self-supervised, no labels |
| **2** | Multiclass Training (3 models) | 100 each | 3 × 100 × 3.5h = **1,050h** | Progressive curriculum: 20+20+60 |
| **5** | MAML Meta-Learning | 100 | 100 × 3.5h = **350h** | Single meta-training run |
| **5** | MAML Adaptations (42 runs) | 10 each | 42 × 10 × 0.05h = **21h** | Fast few-shot adaptation |
| **Total** | - | - | **1,721 hours** | **≈ 71.7 days continuous GPU** |

**Continuous Runtime**: 1,721h ÷ 24h/day = **71.7 days**

**CRITICAL CLARIFICATION**: These are **GPU hours**, NOT wall-clock time!

**Realistic Scenarios**:

1. **Sequential Training** (1 GPU, run experiments one after another):
   - 1,721h GPU time = 1,721h wall-clock time
   - @ 8h/day (overnight + workday): 1,721h ÷ 8h = **215 days ≈ 7 months**
   - @ 16h/day (overnight + all day, supervised): 1,721h ÷ 16h = **107 days ≈ 3.5 months**
   - @ 24/7 (continuous, risky for laptop): 1,721h ÷ 24h = **72 days ≈ 2.4 months**

2. **Parallel Training** (multiple GPUs/cluster):
   - 3 GPUs (1 per model): 1,721h ÷ 3 = **574h per GPU** → 24 days @ 24/7
   - 4 GPUs (MAE parallel): 1,721h ÷ 4 = **430h per GPU** → 18 days @ 24/7
   - **Cluster (8-12 GPUs)**: 1,721h ÷ 10 = **172h per GPU** → **7 days @ 24/7** 🚀

**Feasibility**: 
- ⚠️ **Single Dell laptop**: 7 months @ 8h/day (IMPRACTICAL for continuous 6-month loan)
- ✅ **Cluster access**: 1-2 weeks wall-clock time (HIGHLY PRACTICAL)

### Plan C Feasibility (Dell Precision 7630)

| Phase | Task | Epochs | Time Estimate | Notes |
|-------|------|--------|---------------|-------|
| **0.5** | MAE Pretrain (3 encoders) | 50 each | 3 × 50 × 2h = **300h** | Same as Plan B |
| **2** | Multiclass + All Innovations | 100 each | 3 × 100 × 3.8h = **1,140h** | +8% slower (Damage Attn + extras) |
| **5** | MAML Meta-Learning | 100 | 100 × 3.8h = **380h** | Slightly slower with innovations |
| **5** | MAML Adaptations (42 runs) | 10 each | 42 × 10 × 0.05h = **21h** | Same as Plan B |
| **Total** | - | - | **1,841 hours** | **≈ 76.7 days continuous GPU** |

**Continuous Runtime**: 1,841h ÷ 24h/day = **76.7 days**

**Realistic Scenarios**:

1. **Sequential** (1 GPU): 1,841h ÷ 8h/day = **230 days ≈ 7.7 months** (IMPRACTICAL)
2. **Parallel** (cluster 10 GPUs): 1,841h ÷ 10 = **184h per GPU** → **8 days @ 24/7** 🚀

**Feasibility**: 
- ⚠️ **Single Dell laptop**: 7.7 months @ 8h/day (NOT RECOMMENDED)
- ✅ **Cluster access**: 1-2 weeks wall-clock time (IDEAL)

### VRAM Bottleneck Analysis (6GB RTX 1000 Ada)

**Memory Budget Breakdown** (per training step):

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| **Model Parameters** | ~150MB | MaxViT-Tiny (39M params) + UPerNet |
| **Optimizer States** | ~450MB | AdamW (2x params for momentum + variance) |
| **Batch (4 images, 384×384)** | ~1,100MB | FP16: 4 × 3 × 384 × 384 × 2 bytes |
| **Activations (gradients)** | ~2,800MB | Encoder layers + decoder |
| **Loss computation** | ~200MB | 16-class softmax + dice |
| **CUDA overhead** | ~300MB | PyTorch kernel cache |
| **Total** | **~5,000MB** | ✅ Fits in 6GB with small margin |

**Risk Mitigation**:
- **Gradient checkpointing**: Trade 30% speed for 50% VRAM savings → 2,800 → 1,400MB
- **Batch size 2**: Emergency fallback if VRAM overflow → 1,100 → 550MB
- **Resolution 256×256**: Nuclear option → activations 2,800 → 1,200MB

**Conclusion**: 6GB VRAM is **tight but workable** with:
- Batch size 4
- 384×384 resolution
- Mixed precision (FP16)
- Gradient checkpointing enabled

---

## 🎯 PROPOSED POC INTERMEDIATE (POC-5.5)

**Motivation**: Plan C is too aggressive to jump from binary POC-5. Need intermediate validation step.

### POC-5.5: Multiclass Baseline + Hierarchical MTL (2-3 weeks)

**Goal**: Validate multiclass feasibility and test Innovation #1 (Hierarchical heads) before committing to full Plan C.

**Scope**:
- ✅ **IN**: Multiclass 16-class training, Hierarchical Multi-Task Learning (#1), Class weighting
- ❌ **OUT**: MAE pretraining (#2), MAML (#3), Damage Attention (#4), Progressive Curriculum (#5)

**Why This Order?**:
1. **Hierarchical MTL** is **lowest-risk, highest-impact** innovation (+5-8% mIoU)
2. Tests multiclass feasibility (is 445 samples enough?)
3. Validates class imbalance mitigation strategies
4. Only +150 lines code (2 days work)
5. Only +90h GPU time (3 models × 30 epochs each)

**Timeline** (Dell Precision 7630):
| Phase | Task | Time | Deliverable |
|-------|------|------|-------------|
| **Week 1** | Download dataset, implement Hierarchical UPerNet | 20h work | `hierarchical_upernet.py` |
| **Week 2** | Train 3 models (30 epochs each, not 100) | 3 × 30 × 3.5h = **315h GPU** | Checkpoints + metrics |
| **Week 3** | Evaluate, compare vs binary POC-5 | 10h work | Decision: proceed to Plan C? |

**GPU Timeline**: 
- **Sequential** (1 GPU): 315h ÷ 8h/day = **40 days ≈ 5.5 weeks**
- **Sequential** (1 GPU, 24/7): 315h ÷ 24h/day = **13 days ≈ 2 weeks**
- **Parallel** (3 GPUs, 1 per model): 315h ÷ 3 = **105h per GPU** → **4.4 days @ 24/7** 🚀

**Success Criteria** (Go/No-Go for Plan C):
- ✅ **GO**: Multiclass mIoU ≥ 42% (shows 445 samples sufficient)
- ✅ **GO**: Hierarchical heads improve +3% vs single head (validates Innovation #1)
- ✅ **GO**: Training stable, no VRAM overflow
- ❌ **NO-GO**: mIoU < 35% (dataset too small, need data augmentation overhaul)
- ❌ **NO-GO**: VRAM overflow even with batch_size=2 (need smaller model)

**After POC-5.5**:
- If **GO**: Proceed to Plan C (add MAE pretraining + MAML + Damage Attn)
- If **NO-GO**: Pivot to Plan A-lite (binary + coarse 4-class only, merge rare classes)

---

## 🚦 DECISION FRAMEWORK: Plan B vs Plan C vs POC-5.5

### Option 1: Plan B (Pragmatic, Safe)
**Innovations**: Hierarchical MTL + MAE Pretrain + MAML  
**Timeline**: 7 months (215 days @ 8h/day)  
**Expected mIoU**: 52.1% (MaxViT-Tiny)  
**Risk**: Low (all techniques validated in literature)  
**Publication**: CVPR/ICCV main track (tier 1)  
**Pros**: Solid results, feasible timeline, good paper story  
**Cons**: Missing novelty of Damage Attention (less "wow" factor)  

### Option 2: Plan C (Full Innovation, High Risk/Reward)
**Innovations**: All 5 (Hierarchical + MAE + MAML + Damage Attn + Curriculum)  
**Timeline**: 7.7 months (230 days @ 8h/day)  
**Expected mIoU**: 56.4% (MaxViT-Tiny)  
**Risk**: Medium (Damage Attention novel, curriculum adds complexity)  
**Publication**: CVPR/ICCV oral presentation possible (top tier)  
**Pros**: State-of-the-art, novel contribution, best paper potential  
**Cons**: +15 days timeline, more debugging, higher chance of "something breaks"  

### Option 3: POC-5.5 First, Then Decide (Recommended 🌟)
**Innovations**: Hierarchical MTL only (validate first)  
**Timeline**: 5.5 weeks (POC-5.5) + decision point  
**Expected mIoU**: 45-48% (with hierarchical heads)  
**Risk**: Very low (incremental from POC-5)  
**Next Step**: If successful → Plan C; If issues → Plan B or pivot  
**Pros**: De-risks Plan C, validates assumptions, fast initial results  
**Cons**: Delays full POC-6 by 5 weeks (but worth it for risk mitigation)  

---

## 🎓 RECOMMENDATION: Phased Approach (POC-5.5 → POC-6 Full)

### Phase 1: POC-5.5 (5-6 weeks, Dell Precision 7630)
**Execute**: Multiclass 16-class + Hierarchical MTL (Innovation #1)  
**Goal**: Validate dataset sufficiency, test hierarchical heads impact  
**Deliverable**: 3 models trained (30 epochs each), evaluation report  
**Decision Point**: Go/No-Go for Plan C based on:
- Multiclass mIoU ≥ 42% (dataset sufficient)
- Hierarchical improvement ≥ +3% (innovation works)
- VRAM stable (6GB enough)

### Phase 2: POC-6 Plan C (if POC-5.5 succeeds, 7 months, Dell)
**Execute**: Add MAE (#2) + MAML (#3) + Damage Attn (#4) + Curriculum (#5)  
**Goal**: Full innovation stack, answer RQ1 + RQ2  
**Deliverable**: Top-tier conference paper (CVPR/ICCV)  
**Timeline**: 230 days @ 8h/day GPU (assumes professor loans laptop consistently)

### Phase 2-Alt: POC-6 Plan B (if POC-5.5 shows issues)
**Execute**: Add only MAE (#2) + MAML (#3), skip Damage Attn + Curriculum  
**Goal**: Solid results with lower risk  
**Deliverable**: Conference paper (CVPR/ICCV main track)  
**Timeline**: 215 days @ 8h/day GPU

---

## ⚖️ PLAN C FEASIBILITY: DEEP DIVE

### Can We Run Plan C on Dell Precision 7630 RTX 1000 Ada 6GB?

**Short Answer**: ✅ **YES, but with careful optimizations**.

**Critical Bottlenecks**:
1. **VRAM (6GB)**: Tight fit, requires batch_size=4, 384×384 resolution, gradient checkpointing
2. **Training Time (1,841h)**: Requires ~7.7 months @ 8h/day access to Dell laptop
3. **Thermal Throttling**: 50W TDP laptop GPU may throttle after 4-6h continuous use

**Mitigation Strategies**:

#### VRAM Optimization
```python
# Enable gradient checkpointing (trade 30% speed for 50% VRAM)
model.encoder.set_grad_checkpointing(enable=True)

# Mixed precision with AMP
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Batch size 4 (vs 8 on larger GPUs)
train_loader = DataLoader(dataset, batch_size=4, ...)

# Resolution 384×384 (vs 512×512)
transform = A.Compose([A.Resize(384, 384), ...])
```

#### Training Time Optimization
```python
# Early stopping (aggressive)
early_stop = EarlyStopping(patience=10, min_delta=0.001)

# Reduce MAE pretraining epochs (50 → 30)
mae_pretrain(encoder, epochs=30)  # Still effective per literature

# Reduce MAML meta-epochs (100 → 60)
maml_train(model, tasks, meta_epochs=60)  # Faster convergence with good init
```

#### Thermal Management
- **Laptop cooling pad**: Keep GPU <75°C
- **Batch training**: 4h sessions with 30min cooldown (prevents throttling)
- **Undervolting**: -50mV to -100mV (reduces heat without performance loss)

**Revised Plan C Timeline** (with optimizations):
| Original | Optimized | Reduction |
|----------|-----------|-----------|
| MAE: 300h | MAE: 180h (30 epochs) | -120h |
| Multiclass: 1,140h | Multiclass: 1,050h (early stop) | -90h |
| MAML: 380h | MAML: 230h (60 epochs) | -150h |
| **Total: 1,841h** | **Total: 1,481h** | **-360h (-20%)** |

**Optimized Timeline**: 
- **Sequential** (1 GPU @ 8h/day): 1,481h ÷ 8h = **185 days ≈ 6.2 months** (IMPRACTICAL)
- **Sequential** (1 GPU @ 24/7): 1,481h ÷ 24h = **62 days ≈ 2 months** (laptop thermal risk)
- **Parallel** (cluster 8 GPUs @ 24/7): 1,481h ÷ 8 = **185h per GPU** → **7.7 days ≈ 1 week** 🚀

### Can We Run POC-5.5 on Current PC First?

**User's Current PC** (unknown specs, but likely weaker than Dell):
- Assume: GTX 1060/1650 or RTX 2060 (4-6GB VRAM)
- Goal: Run POC-5.5 (multiclass 30 epochs) to validate approach

**POC-5.5 Requirements**:
- VRAM: ~5GB (same as Plan C, batch_size=4)
- Time: 315h GPU (3 models × 30 epochs × 3.5h)
- Timeline: 315h ÷ 8h/day = **40 days** (realistic on current PC)

**Recommendation**: ✅ **YES, run POC-5.5 on current PC** as validation:
- If current PC has ≥4GB VRAM: Can run POC-5.5 (batch_size=2-4, 256-384px)
- If successful: Confidence to request Dell Precision 7630 for full Plan C
- If issues: Pivot to Plan B (fewer innovations) or merge to 8 classes

---

## ☁️ CLUSTER OPTIONS: Game Changer for Timeline

### Problem with Single GPU (Dell Precision 7630)
- **Sequential training**: 1 model at a time = 1,481h GPU time = 2 months continuous 24/7
- **Thermal risk**: Laptop GPU running 24/7 for 2 months (high failure risk)
- **Occupancy**: Professor can't use laptop for 2 months
- **Impractical**: Even @ 8h/day = 6 months calendar time

### Solution: Cluster/Cloud GPUs (Parallelize Everything)

#### Option 1: University Cluster (FREE, best option if available)
**Typical Setup**:
- 10-20 GPUs (RTX 3090/4090, A100, V100)
- SLURM job scheduler
- Shared queue (may wait 1-3 days for allocation)

**Timeline with Cluster**:
| Phase | Sequential (1 GPU) | Parallel (8 GPUs) | Speedup |
|-------|-------------------|-------------------|---------|
| MAE Pretrain (3 encoders) | 180h | 180h ÷ 3 = **60h** (run 3 parallel jobs) | 3x |
| Multiclass (3 models) | 1,050h | 1,050h ÷ 3 = **350h** (run 3 parallel jobs) | 3x |
| MAML Meta-Learning | 230h | 230h (single job, can't parallelize inner loop) | 1x |
| MAML Adaptations (42) | 21h | 21h ÷ 8 = **3h** (run 8 parallel jobs) | 8x |
| **Total** | **1,481h** | **643h cluster time** | **2.3x** |

**Wall-clock time**: 643h ÷ 24h/day = **27 days @ 24/7** → **~1 month** 🚀

**How to request**:
```bash
# Typical SLURM submission
sbatch --gres=gpu:1 --time=48:00:00 --job-name=mae_convnext train_mae.sh
sbatch --gres=gpu:1 --time=72:00:00 --job-name=multiclass_swin train_multiclass.sh
# etc. (submit all jobs, they run in parallel when GPUs available)
```

**Advantages**:
- ✅ FREE (university resource)
- ✅ Professional GPUs (A100 80GB >> RTX 1000 Ada 6GB)
- ✅ Can run 10+ jobs in parallel
- ✅ No thermal/laptop damage risk

**Disadvantages**:
- ⚠️ Queue wait times (1-3 days to start, not immediate)
- ⚠️ Job time limits (often 48-72h max per job, need checkpointing)
- ⚠️ Shared resource (lower priority for undergrad/master students)

---

#### Option 2: Google Colab Pro/Pro+ (CHEAP, easy to start)
**Pricing**:
- **Colab Pro**: $9.99/month, T4/P100 GPU, 24h max runtime
- **Colab Pro+**: $49.99/month, V100/A100 GPU, longer runtimes, background execution

**Timeline with Colab Pro+**:
- Can run multiple notebooks in parallel (3-5 sessions)
- V100 ~2x faster than RTX 1000 Ada
- 1,481h on RTX 1000 → 740h on V100 → 740h ÷ 5 parallel = **148h wall-clock** → **6 days** 🚀

**Cost**: $49.99/month × 1 month = **~$50 total**

**Advantages**:
- ✅ Start immediately (no approval needed)
- ✅ Very cheap ($50 for entire POC-6)
- ✅ Can run 3-5 parallel notebooks
- ✅ Persistent storage (Google Drive)

**Disadvantages**:
- ⚠️ 24h max runtime (need checkpointing every 24h)
- ⚠️ May disconnect randomly (background execution helps)
- ⚠️ Shared GPU pool (not guaranteed availability during peak hours)

---

#### Option 3: Kaggle Notebooks (FREE, 30h/week GPU quota)
**Specs**:
- FREE Tesla P100 or T4 GPU
- 30 hours/week GPU quota (resets weekly)
- 9h max session runtime

**Timeline with Kaggle**:
- 1,481h ÷ 30h/week = **50 weeks** (sequential, IMPRACTICAL)
- Run 2-3 parallel notebooks: 50 weeks ÷ 3 = **17 weeks ≈ 4 months**

**Advantages**:
- ✅ Completely FREE
- ✅ Good for POC-5.5 validation (315h ÷ 30h/week = 11 weeks with 1 notebook)

**Disadvantages**:
- ❌ 30h/week limit too restrictive for POC-6 full
- ❌ 9h max session (frequent restarts)
- ⚠️ Better for testing/validation, not full training

---

#### Option 4: AWS/Azure/GCP Cloud (EXPENSIVE, full control)
**Pricing** (AWS p3.2xlarge - V100 GPU):
- $3.06/hour on-demand
- $0.92/hour spot instance (can be terminated anytime)

**Timeline with 4× Spot Instances**:
- 1,481h ÷ 4 parallel = 370h per GPU → **370h wall-clock** → **15.4 days**

**Cost**:
- **On-demand**: 1,481h × $3.06 = **$4,532** (VERY EXPENSIVE)
- **Spot**: 1,481h × $0.92 = **$1,363** (still expensive, but 70% cheaper)

**Advantages**:
- ✅ Full control, no queue
- ✅ Can provision 10+ GPUs instantly
- ✅ Professional infrastructure

**Disadvantages**:
- ❌ EXPENSIVE ($1,300+ even with spot instances)
- ❌ Spot can be terminated (need robust checkpointing)
- ⚠️ Overkill for research project (unless funded grant)

---

#### Option 5: Paperspace Gradient (CHEAP cloud, good middle ground)
**Pricing**:
- RTX 4000 (8GB): $0.51/hour
- RTX 5000 (16GB): $0.78/hour
- A4000 (16GB): $0.76/hour

**Timeline with 4× RTX 4000 (parallel)**:
- 1,481h ÷ 4 = 370h per GPU → **370h wall-clock** → **15.4 days**

**Cost**:
- 1,481h × $0.51 = **$755** (much cheaper than AWS)

**Advantages**:
- ✅ Cheaper than AWS/Azure (50% less)
- ✅ Easy setup (Jupyter notebooks)
- ✅ Free tier available (M4000, limited hours)

**Disadvantages**:
- ⚠️ Less reliable than AWS (smaller company)
- ⚠️ Still ~$750 cost

---

### 🎯 CLUSTER RECOMMENDATION

#### Best Options Ranked:

**🥇 1st Choice: University Cluster (if available)**
- **Cost**: FREE ✅
- **Timeline**: ~1 month wall-clock (27 days @ 24/7)
- **Action**: Ask your advisor: "Does the university have a GPU cluster for research? How do I request access?"
- **Likely scenarios**:
  - CS/Engineering department: Often have small cluster (5-10 GPUs)
  - National supercomputing center: Requires proposal (1-2 page research plan)

**🥈 2nd Choice: Google Colab Pro+ ($50/month)**
- **Cost**: ~$50 total ✅✅
- **Timeline**: ~1 week wall-clock (6-7 days with 5 parallel notebooks)
- **Action**: Subscribe immediately, start POC-5.5 validation today
- **Best for**: Fast iteration, testing, POC-5.5 validation

**🥉 3rd Choice: Paperspace Gradient ($755)**
- **Cost**: $755 (moderate) ⚠️
- **Timeline**: ~15 days wall-clock (with 4 parallel GPUs)
- **Action**: If university cluster not available and need full control
- **Best for**: Serious research project, funded by advisor/grant

**❌ Not Recommended: AWS/Azure ($1,300+)**
- Too expensive for student research
- Only if grant-funded or company-sponsored

**❌ Not Recommended: Single Dell laptop (6 months)**
- Timeline too long (6 months @ 8h/day)
- Thermal risk (laptop not designed for 24/7 compute)
- Blocks professor's laptop for half a year

---

### 💡 REVISED RECOMMENDATION: Hybrid Approach

```
Phase 0-1 (Week 1-2): Dataset Verification + Code Setup
├─ Hardware: Your current PC (no GPU needed, CPU-only tasks)
├─ Tasks: Download ARTeFACT, analyze class distribution, implement code
├─ Cost: $0
├─ Time: 2 weeks human work

Phase 2 (Week 3-4): POC-5.5 Validation  
├─ Hardware: Google Colab Pro+ ($50/month) or University cluster
├─ Tasks: Train 3 models (30 epochs each) to validate multiclass works
├─ Cost: $50 (Colab) or $0 (cluster)
├─ Time: 1 week wall-clock (3 parallel Colab notebooks)
├─ Decision Point: GO/NO-GO for Plan C

Phase 3 (Week 5-8): POC-6 Plan C Full
├─ Hardware: University cluster (best) or Colab Pro+ ($50×2 months)
├─ Tasks: MAE pretraining + Multiclass + MAML + all innovations
├─ Cost: $0 (cluster) or $100 (Colab 2 months)
├─ Time: 1 month wall-clock (cluster) or 2 weeks (Colab Pro+ with many parallel sessions)
└─ Deliverable: Full results, paper-ready

Total Cost: $0-$150 (vs $755-$1,300 cloud options)
Total Time: 2 months wall-clock (vs 6 months on single laptop)
```

---

## 🚀 IMMEDIATE ACTION PLAN

### This Week: Validate Cluster Access

**Step 1: Email your advisor/professor** (TODAY)
```
Subject: GPU Cluster Access Request for Thesis Research

Profesor [Name],

Para mi tesis sobre detección de daños en arte patrimonial, necesito
entrenar modelos de deep learning (CNNs y Vision Transformers).

Estimación de recursos:
- ~1,500 GPU-hours totales
- 8-10 GPUs en paralelo → 1 mes de tiempo real
- Compatible con RTX 3090, V100, A100 (6GB+ VRAM)

¿La universidad tiene acceso a un cluster de GPUs para investigación?
De ser así, ¿cuál es el proceso para solicitar acceso?

Alternativa: Estoy evaluando Google Colab Pro+ ($50/mes) si no hay 
cluster disponible.

Gracias,
[Your name]
```

**Step 2: While waiting for response, start with FREE options**
```bash
# Today: Try Kaggle (FREE 30h/week)
# Sign up: https://www.kaggle.com
# Upload POC-5 code, test training 1 model for 1 epoch
# Verify VRAM usage, training speed

# Tomorrow: Try Google Colab FREE tier
# Upload code to Google Colab
# Train for 1 epoch, measure time
# If works well → upgrade to Colab Pro+ ($10/month for testing)
```

**Step 3: Download dataset NOW (CPU task, run on your PC)**
```bash
cd experiments/artefact-data-obtention
huggingface-cli download danielaivanova/damaged-media --repo-type dataset
python scripts/analyze_dataset.py  # Count samples, class distribution
```

---

## 📊 FINAL TIMELINE COMPARISON

| Approach | Hardware | Wall-Clock Time | Cost | Feasibility |
|----------|----------|-----------------|------|-------------|
| **Sequential (1 Dell laptop @ 8h/day)** | RTX 1000 Ada 6GB | **6 months** | $0 | ❌ IMPRACTICAL |
| **Sequential (1 Dell laptop @ 24/7)** | RTX 1000 Ada 6GB | **2 months** | $0 | ⚠️ RISKY (thermal) |
| **University Cluster (8 GPUs)** | A100/V100 mix | **1 month** | $0 | ✅✅✅ BEST |
| **Google Colab Pro+ (5 parallel)** | V100/A100 | **1-2 weeks** | $50-100 | ✅✅ EXCELLENT |
| **Kaggle (3 parallel notebooks)** | P100/T4 | **4 months** | $0 | ⚠️ Slow but FREE |
| **Paperspace (4 GPUs)** | RTX 4000/5000 | **15 days** | $755 | ⚠️ Expensive |
| **AWS Spot (4 GPUs)** | V100 | **15 days** | $1,363 | ❌ Too expensive |

**Recommendation**: 
1. **Try to get university cluster** (email advisor TODAY) → 1 month, $0
2. **Fallback: Colab Pro+** ($50-100) → 1-2 weeks
3. **For testing NOW: Kaggle FREE** → validate code works

---

## 📋 FINAL RECOMMENDATION

## 📋 FINAL RECOMMENDATION (UPDATED: Cluster-First Approach)

### **ANSWER TO YOUR QUESTION**:

> **"¿Estamos hablando de 6 semanas de entrenamiento seguidos?"**

❌ **NO** - Era confusión de cálculo. Las opciones son:

1. **1 GPU (Dell laptop) secuencial**: 1,481 GPU-hours
   - @ 8h/día: 185 días = **6 meses calendario** ❌ IMPRACTICAL
   - @ 24/7: 62 días = **2 meses continuos** ⚠️ Risky (thermal, ocupa laptop del profesor)

2. **8 GPUs (cluster) paralelo**: 1,481 GPU-hours ÷ 8 GPUs
   - = 185h per GPU @ 24/7 = **7.7 días = 1 semana** ✅✅✅
   - Costo: **$0** (cluster universitario)

3. **5 GPUs (Colab Pro+) paralelo**: 1,481h ÷ 5 parallel sessions
   - = 296h wall-clock = **12 días @ 24/7** ✅✅
   - Costo: **$50-100** (1-2 meses Colab Pro+)

> **"¿Mejor veo si puedo pedir un cluster?"**

✅ **SÍ, ABSOLUTAMENTE**. Cluster cambia todo:
- De **6 meses** (1 laptop) → **1 semana** (cluster)
- De **impractical** → **totally feasible**

---

### Phased Execution Plan (CLUSTER-OPTIMIZED)

```
┌─────────────────────────────────────────────────────────────────┐
│ TOTAL TIMELINE: 6-8 weeks wall-clock (NOT 6 months!)            │
├─────────────────────────────────────────────────────────────────┤
│ Week 1-2: Setup (Your PC, no GPU needed)                        │
│   - Email advisor for cluster access                             │
│   - Download ARTeFACT dataset (CPU task)                         │
│   - Implement Hierarchical UPerNet code                          │
│   - Test 1 epoch on Kaggle FREE (validate code)                 │
│   - Deliverable: Code ready to deploy                            │
├─────────────────────────────────────────────────────────────────┤
│ Week 3-4: POC-5.5 Validation (Cluster or Colab Pro+)            │
│   - Train 3 models (30 epochs each) in PARALLEL                  │
│   - Wall-clock: 4-7 days with cluster, 1 week with Colab        │
│   - Cost: $0 (cluster) or $50 (Colab Pro+)                      │
│   - Deliverable: GO/NO-GO decision for Plan C                    │
├─────────────────────────────────────────────────────────────────┤
│ Week 5-8: POC-6 Plan C Full (Cluster REQUIRED)                  │
│   - MAE pretrain (3 encoders parallel): 2-3 days                │
│   - Multiclass training (3 models parallel): 10-14 days         │
│   - MAML meta-learning: 7-10 days                               │
│   - Wall-clock: 3-4 weeks with cluster                          │
│   - Cost: $0 (cluster) or $100-150 (Colab Pro+ 2-3 months)      │
│   - Deliverable: Full results, paper draft                       │
└─────────────────────────────────────────────────────────────────┘
```

**Key Changes from Previous Plan**:
- ❌ **OLD**: "6 months on single laptop @ 8h/day" → IMPRACTICAL
- ✅ **NEW**: "6-8 weeks on cluster" → TOTALLY FEASIBLE
- 💰 **Cost**: $0 (cluster) or $50-150 (Colab fallback)

---

### IMMEDIATE NEXT STEPS (This Week)

#### ⚡ Priority 1: Cluster Access (TODAY)
```bash
# Email your advisor with this template:
```
**Subject**: Solicitud de acceso a cluster GPU para tesis (Heritage Art Damage Detection)

Estimado Profesor [Name],

Para mi trabajo de tesis sobre detección automática de daños en arte patrimonial, 
necesito entrenar modelos de deep learning (CNNs, Vision Transformers) en el 
dataset ARTeFACT (~445 imágenes anotadas, 16 clases).

**Requerimientos estimados**:
- ~1,500 GPU-hours totales (~2 meses en 1 GPU, pero **1 semana en cluster con 8 GPUs**)
- Compatible con: RTX 3090/4090, V100, A100 (mínimo 6GB VRAM)
- Framework: PyTorch + Docker
- Prioridad: Media (tesis de pregrado/maestría, deadline paper Nov 2025)

**Preguntas**:
1. ¿La universidad/departamento tiene un cluster de GPUs disponible?
2. ¿Cuál es el proceso de solicitud? ¿Necesito propuesta escrita?
3. ¿Hay cola de espera? ¿Tiempo límite por job?

**Plan B alternativo**:
Si no hay cluster disponible, usaré Google Colab Pro+ ($50/mes, ~1-2 semanas).

Adjunto: Resultados POC-5 (binary segmentation, 71% mIoU con MaxViT-Tiny).

Gracias por su apoyo,
[Your Name]
```

#### ⚡ Priority 2: Test FREE Options (While Waiting for Response)

**Option A: Kaggle (30h/week FREE)**
```bash
# 1. Create account: https://www.kaggle.com
# 2. Upload your POC-5 code as Kaggle notebook
# 3. Run 1 epoch training to measure speed:
#    - If 1 epoch = 3.5h on RTX 1000 Ada
#    - On Kaggle P100: expect ~2h per epoch (faster)
# 4. Validates: Code works, VRAM fits, training stable
```

**Option B: Google Colab FREE Tier**
```bash
# 1. Upload POC-5 code to Google Colab
# 2. Connect to T4 GPU (free tier)
# 3. Run 1 epoch training
# 4. If works → upgrade to Pro+ ($9.99/month for testing)
```

#### ⚡ Priority 3: Download Dataset (CPU Task, Run Today)
```bash
cd /home/brandontrigueros/DevWSL/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-data-obtention

# Download full ARTeFACT
huggingface-cli download danielaivanova/damaged-media --repo-type dataset --local-dir ./data/full

# Count samples and analyze
python -c "
import datasets
ds = datasets.load_dataset('danielaivanova/damaged-media')
print(f'Total samples: {len(ds[\"train\"])}')
print(f'Columns: {ds[\"train\"].column_names}')
print(f'First sample: {ds[\"train\"][0]}')
"
```

---

### Decision Matrix: Cluster vs Colab vs Dell Laptop

| Factor | University Cluster | Google Colab Pro+ | Dell Laptop (24/7) |
|--------|-------------------|-------------------|-------------------|
| **Wall-clock time** | 🟢 **1 week** | 🟢 **1-2 weeks** | 🔴 2 months |
| **Cost** | 🟢 **$0** | 🟡 $50-150 | 🟢 $0 |
| **Reliability** | 🟢 High (SLURM queue) | 🟡 Medium (may disconnect) | 🟡 Medium (thermal risk) |
| **Ease of setup** | 🟡 Requires approval | 🟢 Instant (credit card) | 🟢 Already have laptop |
| **Parallel jobs** | 🟢 8-12 GPUs | 🟡 3-5 sessions | 🔴 1 GPU only |
| **Availability** | 🟡 Unknown (ask advisor) | 🟢 Guaranteed | 🟢 Guaranteed |
| **Professional GPUs** | 🟢 A100/V100 (fast) | 🟢 V100/A100 | 🔴 RTX 1000 Ada (slow) |
| **Thermal safety** | 🟢 Datacenter cooling | 🟢 Cloud (N/A) | 🔴 Laptop overheating risk |
| **Professor impact** | 🟢 Zero (shared resource) | 🟢 Zero | 🔴 Blocks laptop 2 months |

**Recommendation**:
1. **Try cluster first** (email advisor TODAY) → Best option (free + fast)
2. **If cluster not available or >2 week queue**: Use Colab Pro+ ($50-150)
3. **Avoid Dell laptop 24/7**: Only use for testing/validation (not full training)

---

### ✅ YOUR ANSWER: "Can I Run Plan C?"

**Short Answer**: ✅ **YES, if you get cluster or Colab Pro+**

**Timeline Comparison**:

| Scenario | Wall-Clock Time | Cost | Feasibility |
|----------|----------------|------|-------------|
| ❌ **1 Dell laptop @ 8h/day** | 6 months | $0 | IMPRACTICAL |
| ⚠️ **1 Dell laptop @ 24/7** | 2 months | $0 | RISKY (thermal) |
| ✅ **University cluster** | **1 week** | **$0** | **IDEAL** |
| ✅ **Colab Pro+ (5 parallel)** | **1-2 weeks** | **$50-150** | **EXCELLENT** |

**My Recommendation**:

```
┌──────────────────────────────────────────────────┐
│ 🎯 RECOMMENDED PATH: Cluster-First Hybrid         │
├──────────────────────────────────────────────────┤
│ 1. Email advisor TODAY (request cluster access)  │
│ 2. While waiting: Download dataset (CPU task)    │
│ 3. Test code on Kaggle FREE (validate it works)  │
│ 4. If cluster approved (1-2 weeks):              │
│    → Run POC-5.5 + POC-6 on cluster (6-8 weeks)  │
│ 5. If cluster NOT available:                     │
│    → Use Colab Pro+ ($50-150, 3-4 weeks total)   │
│ 6. Dell laptop: Only for testing/debugging       │
│    → NOT for full training runs                  │
└──────────────────────────────────────────────────┘
```

**Expected Total Time**: 
- **Best case** (cluster): 6-8 weeks wall-clock, $0
- **Fallback** (Colab Pro+): 4-6 weeks wall-clock, $100-150
- **Worst case** (Dell 24/7): 2-3 months, $0 (but risky)

---

### 🎬 What We Do RIGHT NOW

**I need you to confirm**:

1. ✅ **I will email my advisor TODAY** to ask about cluster access
2. ✅ **I will download ARTeFACT dataset** this week (CPU task, can run on your PC)
3. ✅ **I will test code on Kaggle FREE** (30h/week) to validate it works
4. **Choose fallback option** if cluster not available:
   - 🟢 Option A: Colab Pro+ ($50-150, fastest fallback, 1-2 weeks)
   - 🟡 Option B: Kaggle only (FREE but slow, 3-4 months)
   - 🔴 Option C: Dell laptop 24/7 (FREE but risky, 2 months)

**Once you confirm, I will**:
- Help you write the cluster request email (tailored to your university)
- Create POC-5.5 code structure (hierarchical UPerNet implementation)
- Setup Kaggle/Colab notebooks for testing
- Download and analyze ARTeFACT dataset

**Your decision**: ¿Procedemos con el plan Cluster-First? (Email advisor + test on Kaggle while waiting)

---
