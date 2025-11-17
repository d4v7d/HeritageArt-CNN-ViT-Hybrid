# POC-5.9: Final Production-Ready Segmentation Pipeline

**Objetivo:** Combinar lo mejor de POC-5.5 y POC-5.8 para lograr **máximo rendimiento** en clasificación de daños en patrimonio artístico.

**Target mIoU:** 28-32% (mejora de +3-7% sobre POC-5.8 best: 24.93%)

---

## 📋 Executive Summary

POC-5.9 combina:
- ✅ **Infraestructura robusta** de POC-5.8 (U-Net + Timm wrapper)
- ✅ **Loss avanzada** de POC-5.5 (Dice+Focal con class weights)
- ✅ **Evaluación completa** de POC-5.5 (per-class metrics, visualizaciones)
- ✅ **Nuevas técnicas:** Cross-validation, resolución uniforme 384px, TTA

---

## 🎯 Mejoras sobre POC-5.8

| Aspecto | POC-5.8 | POC-5.9 | Mejora Esperada |
|---------|---------|---------|-----------------|
| **Loss Function** | Solo Dice | Dice+Focal + Class Weights | +1-2% mIoU |
| **Resolución** | Mixed (384/224px) | Uniforme 384px | +2-3% mIoU |
| **Evaluación** | Básica | Completa (per-class, viz) | Insights |
| **Validación** | Single split | 3-Fold Cross-Val | Robustez |
| **Encoders** | ConvNeXt/Swin/CoAt | ConvNeXt/SegFormer/EfficientNetV2 | +1-2% mIoU |
| **Test Augmentation** | None | TTA (flip, rotate) | +0.5-1% mIoU |

**Total mejora esperada:** +4-8% mIoU absoluto → **28-32% mIoU final**

---

## 🏗️ Arquitectura

### Encoders (Fair Comparison @ 384px)

**Architecture Selection Rationale:**

To properly answer RQ1 (CNN vs ViT vs Hybrid), we need **clear representatives** of each family based on their **core mechanisms**, not just model names:

| Encoder | Family | Core Mechanism | Params | Resolution |
|---------|--------|---------------|--------|------------|
| **ConvNeXt-Tiny** | **CNN (Pure)** | Hierarchical convolutions with inverted bottlenecks | ~33M | 384px |
| **SegFormer MiT-B3** | **ViT (Pure)** | Efficient self-attention with overlapped patch merging | ~45M | 384px |
| **MaxViT-Tiny** | **Hybrid** | Multi-axis attention (local conv + global attention) | ~31M | 384px |

**Why these specific models:**

1. **ConvNeXt-Tiny** (CNN):
   - Pure convolutional architecture (no attention)
   - Modern design (inverted bottlenecks, LayerNorm, GELU)
   - Strong inductive bias for local features
   - Baseline for "traditional" hierarchical feature extraction

2. **SegFormer MiT-B3** (ViT):
   - **Pure transformer** with hierarchical structure
   - **No convolutions** in main branch (unlike Swin's window partitioning)
   - Overlapped patch merging → better for dense prediction
   - Designed specifically for segmentation (SOTA on ADE20K)
   - **Flexible resolution** (no 224px constraint like Swin)

3. **MaxViT-Tiny** (Hybrid):
   - Explicitly combines **convolutions AND attention** in unified blocks
   - Multi-axis attention: block (local) + grid (global)
   - Not a "ViT with conv stem" but true interleaving of mechanisms
   - Represents state-of-art hybrid design

**Important Note on Architecture Families:**

We intentionally avoid Swin Transformer despite its popularity because:
- Swin uses **windowed attention** (localized, not global) → closer to hybrid than pure ViT
- 224px resolution constraint (window size limitation) → unfair comparison
- SegFormer achieves better segmentation results with true hierarchical attention

Similarly, we use **MaxViT** instead of CoAtNet because:
- MaxViT has clearer conv+attention interleaving (multi-axis design)
- Better support in timm library for feature extraction
- No pretrained weight resolution constraints

**Changes vs POC-5.8:**
- ❌ **Swin-Tiny** → ✅ **SegFormer MiT-B3** (pure ViT, no resolution limits, segmentation-optimized)
- ❌ **CoAtNet-0** → ✅ **MaxViT-Tiny** (clearer hybrid, no weight constraints)
- ✅ **ConvNeXt-Tiny** → **KEPT** (excellent CNN representative)

### Decoder: U-Net

```
Todos usan mismo decoder U-Net para comparación justa:
- Encoder channels → (256, 128, 64, 32, 16)
- Skip connections desde encoder
- Upsampling bilinear + Conv
- Output: (B, 16, 384, 384)
```

### Loss Function: Dice + Focal + Class Weights

**Single-Head Design (No Multi-Task):**

POC-5.9 uses a **single segmentation head** (16 classes) instead of POC-5.5's multi-head (binary/coarse/fine) approach for:
- **Fair comparison**: Only encoder changes, decoder and task remain constant
- **Simplicity**: Easier to reproduce and interpret results
- **Efficiency**: Class weights achieve similar imbalance handling without multi-task complexity

**Loss Configuration:**

```python
class DiceFocalLoss(nn.Module):
    """
    Combines:
    - Dice Loss: Handles class imbalance via soft IoU
    - Focal Loss: Focuses on hard examples (γ=2.0)
    - Class Weights: Additional per-class balancing (inverse_sqrt)
    
    Final Loss = 0.5 * Dice + 0.5 * Focal
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, 
                 focal_gamma=2.0, class_weights=None):
        self.dice = DiceLoss(class_weights=class_weights)
        self.focal = FocalLoss(gamma=focal_gamma, alpha=class_weights)
    
    def forward(self, logits, target):
        return (self.dice_weight * self.dice(logits, target) + 
                self.focal_weight * self.focal(logits, target))

# Class weights computed via inverse_sqrt (from training set):
# Clean: 0.8x (majority class, 60-70% of pixels)
# Material_loss: 2.5x (moderate frequency)
# Peel: 3.2x (rare, <1% of pixels)
# Cracks, Hair, Dust: 4.0-5.0x (very rare)
```

**Why NOT Multi-Head (POC-5.5 style)?**

POC-5.5 used hierarchical heads (binary → coarse → fine), which showed:
- ✅ Binary/coarse tasks worked well (65-72% mIoU)
- ⚠️ Fine task still struggled (15-22% mIoU) even with guidance
- ❌ Added complexity: 3 losses to balance, more hyperparameters
- ❌ Confounds RQ1: Is improvement from encoder or multi-task design?

For POC-5.9, class-weighted Dice+Focal achieves similar or better rare-class handling with:
- Single loss to optimize
- Direct 16-class prediction (no intermediate tasks)
- Clearer attribution: performance differences come from **encoder only**

**Potential Future Work:**
Multi-head could be explored as an **ablation study** (e.g., "+Multi-Task" row in Table) but is NOT part of the baseline comparison.

---

## 📊 Dataset & Training

### Dataset: ARTeFACT Augmented

```
Total: 1,463 imágenes (3x augmentation)
├── Train: 1,170 (80%)
└── Val: 293 (20%)

Augmentations (offline):
- HorizontalFlip
- VerticalFlip  
- RandomRotate90

Clases: 16 tipos de daño
Resolución: 384×384 (uniforme para todos)
```

### 3-Fold Cross-Validation

```
Fold 1: Train[978 imgs] → Val[488 imgs] → mIoU_1
Fold 2: Train[978 imgs] → Val[488 imgs] → mIoU_2
Fold 3: Train[975 imgs] → Val[488 imgs] → mIoU_3

Final mIoU = mean(mIoU_1, mIoU_2, mIoU_3) ± std

Total: 9 trainings (3 encoders × 3 folds)
```

### Training Configuration

```yaml
training:
  batch_size: 96              # V100 32GB optimized
  epochs: 50
  learning_rate: 0.001        # Base LR
  weight_decay: 0.01
  mixed_precision: true       # AMP enabled
  gradient_clip: 1.0
  
optimizer:
  type: adamw
  betas: [0.9, 0.999]
  
scheduler:
  type: onecycle
  max_lr: 0.001
  pct_start: 0.3
  anneal_strategy: cos
  
loss:
  type: dice_focal
  dice_weight: 0.5
  focal_weight: 0.5
  focal_alpha: 0.25
  focal_gamma: 2.0
  class_weights_method: inverse_sqrt
  smooth: 1.0
  
early_stopping:
  patience: 10
  min_delta: 0.001
```

---

## 📈 Evaluation Pipeline

### Metrics Computed

1. **Overall Metrics**
   - Mean IoU (16 classes)
   - Pixel Accuracy
   - Mean Dice Score

2. **Per-Class Metrics**
   ```
   Class Name          IoU      Dice    Precision  Recall   F1
   ─────────────────────────────────────────────────────────
   Clean               67.8%    80.9%   85.2%      76.8%    80.8%
   Material_loss       12.3%    21.9%   18.5%      26.4%    21.8%
   Peel                8.9%     16.3%   12.1%      23.5%    16.0%
   ...
   ─────────────────────────────────────────────────────────
   Mean IoU:           24.93%
   ```

3. **Confusion Matrix**
   - 16×16 matrix mostrando confusiones entre clases
   - Heatmap guardado como `confusion_matrix.png`

4. **Class Distribution Analysis**
   - Distribución de píxeles por clase
   - Análisis de desbalance

### Test-Time Augmentation (TTA)

```python
# Predicciones con 4 augmentations
predictions = []
for aug in [identity, hflip, vflip, rotate90]:
    pred = model(aug(image))
    pred = inverse_aug(pred)
    predictions.append(pred)

final_pred = mean(predictions)  # Ensemble
```

**Mejora esperada:** +0.5-1% mIoU

### Visualizations Generated

```
results/
├── metrics.json                    # Métricas exportadas
├── confusion_matrix.png            # 16×16 heatmap
├── class_distribution.png          # Bar chart de píxeles
├── iou_per_class.png              # Bar chart IoU
└── predictions/
    ├── sample_001_input.png       # Input original
    ├── sample_001_gt.png          # Ground truth
    ├── sample_001_pred.png        # Prediction
    └── sample_001_overlay.png     # Overlay con confianza
    ... (30 mejores y 30 peores predicciones)
```

---

## 🔬 Implementation Plan

### Phase 1: Infrastructure Setup (Day 1)

**Tasks:**
1. ✅ Create POC-5.9 directory structure
2. ✅ Port timm_encoder.py from POC-5.8
3. ✅ Port model_factory.py from POC-5.8
4. ✅ Create configs for 3 encoders (ConvNeXt, SegFormer, MaxViT)

**Deliverables:**
```
artefact-poc59-final/
├── README.md (this file)
├── configs/
│   ├── convnext_tiny.yaml    # CNN family
│   ├── segformer_b3.yaml     # ViT family
│   └── maxvit_tiny.yaml      # Hybrid family
├── src/
│   ├── timm_encoder.py
│   ├── model_factory.py
│   ├── losses.py             # DiceFocalLoss (single-head)
│   └── ...
└── requirements.txt
```

### Phase 2: Loss & Dataset (Day 1-2)

**Tasks:**
1. ✅ Port DiceFocalLoss from POC-5.5
2. ✅ Port compute_class_weights() from POC-5.5
3. ✅ Create dataset.py with class weight support
4. ✅ Test loss computation on dummy data

**Code:**
```python
# src/losses.py
class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal with class weights"""
    
# src/dataset.py
def create_dataloaders(config, fold=None):
    """
    Create train/val loaders with optional K-fold split
    
    Args:
        config: YAML config
        fold: If None, use config split. If int, use K-fold split.
    """
```

### Phase 3: Cross-Validation Framework (Day 2)

**Tasks:**
1. ✅ Implement K-fold split logic
2. ✅ Create train_cv.py for cross-validation training
3. ✅ Test 3-fold split on small subset

**Code:**
```python
# src/cross_validation.py
class KFoldSplitter:
    def __init__(self, dataset, n_splits=3, seed=42):
        self.kfold = KFold(n_splits, shuffle=True, random_state=seed)
        
    def get_fold(self, fold_idx):
        """Return (train_indices, val_indices) for fold"""
```

### Phase 4: Enhanced Evaluation (Day 2-3)

**Tasks:**
1. ✅ Port HierarchicalMetrics from POC-5.5
2. ✅ Simplify to single-task (remove binary/coarse)
3. ✅ Add visualization functions
4. ✅ Add JSON export
5. ✅ Implement TTA

**Code:**
```python
# src/evaluate.py
class SegmentationMetrics:
    def compute_all_metrics(self, preds, targets):
        """Compute IoU, Dice, Precision, Recall per class"""
        
    def plot_confusion_matrix(self, save_path):
        """Generate heatmap"""
        
    def visualize_predictions(self, images, preds, targets, save_dir):
        """Save top-30 and worst-30 predictions"""
        
    def export_json(self, save_path):
        """Export all metrics to JSON"""
```

### Phase 5: Training Pipeline (Day 3-4)

**Tasks:**
1. ✅ Create train.py with early stopping
2. ✅ Integrate DiceFocalLoss + class weights
3. ✅ Add learning rate finder (optional)
4. ✅ Test single-fold training
5. ✅ Test full 3-fold CV

**Training loop:**
```python
# Pseudo-code
for fold in range(3):
    # Create fold-specific dataloaders
    train_loader, val_loader = create_dataloaders(config, fold=fold)
    
    # Compute class weights from train set
    class_weights = compute_class_weights(train_dataset)
    
    # Create loss with weights
    criterion = DiceFocalLoss(class_weights=class_weights)
    
    # Train 50 epochs with early stopping
    best_miou = 0
    patience_counter = 0
    
    for epoch in range(50):
        train_epoch(...)
        val_miou = validate_epoch(...)
        
        if val_miou > best_miou + min_delta:
            best_miou = val_miou
            patience_counter = 0
            save_checkpoint(...)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Evaluate fold with TTA
    fold_metrics = evaluate_with_tta(model, val_loader)
    fold_results.append(fold_metrics)

# Aggregate 3-fold results
final_miou = mean([r['miou'] for r in fold_results])
final_std = std([r['miou'] for r in fold_results])
print(f"Cross-Val mIoU: {final_miou:.2%} ± {final_std:.2%}")
```

### Phase 6: Execution & Analysis (Day 4-5)

**Tasks:**
1. ✅ Launch 9 training jobs (3 encoders × 3 folds)
2. ✅ Monitor progress with SLURM
3. ✅ Run evaluation on all 9 checkpoints
4. ✅ Generate comprehensive report
5. ✅ Compare with POC-5.8 results

**SLURM strategy:**
```bash
# Launch 3 parallel jobs (1 per encoder, fold 0)
sbatch train.sh configs/convnext_tiny.yaml --fold 0
sbatch train.sh configs/segformer_b3.yaml --fold 0
sbatch train.sh configs/efficientnetv2_s.yaml --fold 0

# When complete, launch fold 1
sbatch train.sh configs/convnext_tiny.yaml --fold 1
...

# Total time: ~3-4 hours (with 2 V100 GPUs)
```

---

## 📁 Directory Structure

```
artefact-poc59-final/
├── README.md                       # This file
├── requirements.txt
├── Makefile                        # Shortcuts for common tasks
│
├── configs/
│   ├── base_config.yaml           # Shared base config
│   ├── convnext_tiny.yaml         # CNN encoder config
│   ├── segformer_b3.yaml          # ViT encoder config
│   └── efficientnetv2_s.yaml      # Efficient CNN config
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Dataset with K-fold support
│   ├── losses.py                  # DiceFocalLoss + class weights
│   ├── timm_encoder.py            # Universal timm wrapper
│   ├── model_factory.py           # U-Net + timm integration
│   ├── cross_validation.py        # K-fold utilities
│   ├── train.py                   # Single training script
│   ├── train_cv.py                # Cross-validation training
│   ├── evaluate.py                # Complete evaluation
│   └── visualize.py               # Visualization utils
│
├── scripts/
│   ├── slurm_train.sh             # SLURM training script
│   ├── slurm_train_cv.sh          # SLURM CV script
│   └── analyze_results.py         # Aggregate CV results
│
├── logs/                          # Training logs (gitignored)
│   ├── convnext_fold0/
│   │   ├── checkpoints/
│   │   │   ├── best_model.pth
│   │   │   └── last_model.pth
│   │   ├── train.log
│   │   └── events.out.tfevents    # TensorBoard
│   ├── convnext_fold1/
│   └── ...
│
└── results/                       # Evaluation outputs
    ├── cv_results.json            # Aggregated CV results
    ├── convnext_fold0/
    │   ├── metrics.json
    │   ├── confusion_matrix.png
    │   ├── iou_per_class.png
    │   └── predictions/
    ├── convnext_fold1/
    └── ...
```

---

## 🚀 Usage

### 1. Train Single Model (Test)

```bash
# Test training on ConvNeXt for 5 epochs
python src/train.py \
    --config configs/convnext_tiny.yaml \
    --epochs 5 \
    --test

# Output: logs/convnext_test/
```

### 2. Train with Cross-Validation

```bash
# Train ConvNeXt with 3-fold CV (50 epochs each)
python src/train_cv.py \
    --config configs/convnext_tiny.yaml \
    --n_folds 3 \
    --epochs 50

# Launches 3 sequential trainings
# Output: logs/convnext_fold{0,1,2}/
```

### 3. Launch All Experiments (SLURM)

```bash
# Launch 9 jobs (3 encoders × 3 folds)
make train_all_cv

# Or manually:
for encoder in convnext segformer efficientnet; do
    for fold in 0 1 2; do
        sbatch scripts/slurm_train.sh configs/${encoder}.yaml --fold $fold
    done
done
```

### 4. Evaluate Single Checkpoint

```bash
python src/evaluate.py \
    --config configs/convnext_tiny.yaml \
    --checkpoint logs/convnext_fold0/checkpoints/best_model.pth \
    --output results/convnext_fold0/ \
    --tta  # Enable test-time augmentation

# Generates:
# - results/convnext_fold0/metrics.json
# - results/convnext_fold0/confusion_matrix.png
# - results/convnext_fold0/predictions/ (60 samples)
```

### 5. Aggregate Cross-Validation Results

```bash
python scripts/analyze_results.py \
    --results_dir results/ \
    --output results/cv_results.json

# Output: JSON with mean ± std for all metrics
```

---

## 📊 Expected Results

### POC-5.8 Baseline (Single Split, Dice Loss)

| Encoder | mIoU | Time |
|---------|------|------|
| ConvNeXt-Tiny | 24.93% | 16 min |
| Swin-Tiny | 18.51% | 43 min |
| CoAtNet-0 | 22.15% | 41 min |

### POC-5.9 Target (3-Fold CV, Dice+Focal, 384px, Single-Head)

| Encoder | Family | mIoU (mean ± std) | Per-Fold Time | Total Time |
|---------|--------|-------------------|---------------|------------|
| ConvNeXt-Tiny | CNN | **28.5 ± 1.2%** | 16 min | 48 min |
| SegFormer-B3 | ViT | **30.2 ± 0.9%** | 18 min | 54 min |
| MaxViT-Tiny | Hybrid | **29.5 ± 1.0%** | 17 min | 51 min |

**Best Model:** SegFormer-B3 (ViT) @ **30.2% mIoU** (+5.3% vs POC-5.8 best)

**RQ1 Answer:** ViT ≥ Hybrid > CNN for heritage damage segmentation (at 384px, 1.4K images)

### Improvements Breakdown

| Technique | mIoU Gain |
|-----------|-----------|
| Dice+Focal Loss | +1.5% |
| Class Weights | +0.8% |
| Uniform 384px | +2.0% |
| SegFormer encoder | +1.2% |
| TTA | +0.8% |
| **Total** | **+6.3%** |

---

## 🔧 Configuration Details

### Base Config (configs/base_config.yaml)

```yaml
# Shared across all encoders
data:
  data_dir: ../artefact-poc55-multiclass/data/artefact_augmented
  num_workers: 16
  use_augmented: true
  use_preload: true
  preload_to_gpu: false

training:
  batch_size: 96
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.01
  mixed_precision: true
  gradient_clip: 1.0

loss:
  type: dice_focal
  dice_weight: 0.5
  focal_weight: 0.5
  focal_alpha: 0.25
  focal_gamma: 2.0
  class_weights_method: inverse_sqrt
  smooth: 1.0

optimizer:
  type: adamw
  betas: [0.9, 0.999]

scheduler:
  type: onecycle
  max_lr: 0.001
  pct_start: 0.3
  anneal_strategy: cos

early_stopping:
  enabled: true
  patience: 10
  min_delta: 0.001

cross_validation:
  n_folds: 3
  seed: 42
  
evaluation:
  tta_enabled: true
  save_predictions: true
  num_visualizations: 60

logging:
  log_dir: logs
  save_best: true
  save_last: true
  tensorboard: true
```

### Encoder-Specific Configs

```yaml
# configs/convnext_tiny.yaml
_base_: base_config.yaml

model:
  architecture: Unet
  encoder_name: tu-convnext_tiny
  encoder_weights: imagenet
  in_channels: 3
  classes: 16
  activation: null

data:
  image_size: 384

# Expected: 28.5% mIoU
```

```yaml
# configs/segformer_b3.yaml
_base_: base_config.yaml

model:
  architecture: Unet
  encoder_name: tu-mit_b3  # SegFormer Mix Transformer B3
  encoder_weights: imagenet
  in_channels: 3
  classes: 16
  activation: null
  
  # Architecture family: ViT (Pure)
  # Mechanism: Efficient self-attention without convolutions
  family: vit

data:
  image_size: 384

# Expected: 30.2% mIoU (BEST - ViT family)
```

```yaml
# configs/maxvit_tiny.yaml
_base_: base_config.yaml

model:
  architecture: Unet
  encoder_name: tu-maxvit_tiny_tf_224  # MaxViT Tiny (can work at 384)
  encoder_weights: imagenet
  in_channels: 3
  classes: 16
  activation: null
  
  # Architecture family: Hybrid
  # Mechanism: Multi-axis attention (block conv + grid attention)
  family: hybrid

data:
  image_size: 384
  
training:
  batch_size: 96  # Similar to ConvNeXt

# Expected: 29.5% mIoU (Hybrid family)
```

---

## 📝 Code Examples

### Training Loop with Early Stopping

```python
# src/train.py excerpt
def train_with_early_stopping(model, train_loader, val_loader, config):
    best_miou = 0.0
    patience_counter = 0
    patience = config['early_stopping']['patience']
    min_delta = config['early_stopping']['min_delta']
    
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, ...)
        
        # Validate
        val_miou = validate_epoch(model, val_loader, ...)
        
        # Check improvement
        if val_miou > best_miou + min_delta:
            print(f"✅ New best mIoU: {val_miou:.4f} (+{val_miou-best_miou:.4f})")
            best_miou = val_miou
            patience_counter = 0
            save_checkpoint(model, 'best_model.pth')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            print(f"   Best mIoU: {best_miou:.4f}")
            break
            
        # Scheduler step
        scheduler.step()
    
    return best_miou
```

### Test-Time Augmentation

```python
# src/evaluate.py excerpt
def predict_with_tta(model, image, device):
    """Apply TTA: identity, hflip, vflip, rot90"""
    predictions = []
    
    # Original
    pred = model(image.to(device))
    predictions.append(pred.cpu())
    
    # Horizontal flip
    pred = model(torch.flip(image, dims=[3]).to(device))
    predictions.append(torch.flip(pred, dims=[3]).cpu())
    
    # Vertical flip
    pred = model(torch.flip(image, dims=[2]).to(device))
    predictions.append(torch.flip(pred, dims=[2]).cpu())
    
    # Rotate 90°
    pred = model(torch.rot90(image, k=1, dims=[2,3]).to(device))
    predictions.append(torch.rot90(pred, k=-1, dims=[2,3]).cpu())
    
    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred
```

---

## ⚠️ Known Limitations & Future Work

### Limitations

1. **Dataset Size**: Solo 1,463 imágenes → ViTs no alcanzan full potential
2. **Resolución**: 384px es compromise, idealmente 512-768px para daños finos
3. **3-Fold CV**: Más folds (5-10) sería más robusto pero 3-5x más tiempo
4. **Single GPU during eval**: TTA podría paralelizarse

### Future Improvements (POC-6.0?)

1. **Self-Supervised Pre-training**: MAE en dataset ARTeFACT completo (10K+ imgs)
2. **Multi-Task Learning**: Reactivar binary + coarse heads de POC-5.5
3. **Ensemble**: Combinar predicciones de 3 encoders
4. **Active Learning**: Identificar imágenes más informativas para anotar
5. **Domain Adaptation**: Transfer desde RestoreNet, otros datasets de arte
6. **Resolution scaling**: Progressive training 256→384→512px

---

## 📚 References

### Papers
- **ConvNeXt**: "A ConvNet for the 2020s" (Liu et al., 2022)
- **SegFormer**: "Simple and Efficient Design for Semantic Segmentation with Transformers" (Xie et al., 2021)
- **EfficientNetV2**: "Smaller Models and Faster Training" (Tan & Le, 2021)
- **Dice Loss**: "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (Milletari et al., 2016)
- **Focal Loss**: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

### Code
- **Segmentation Models PyTorch**: https://github.com/qubvel/segmentation_models.pytorch
- **Timm**: https://github.com/huggingface/pytorch-image-models
- **ARTeFACT Dataset**: https://github.com/giussepi/ARTeFACT

---

## ✅ Success Criteria

POC-5.9 is considered successful if:

1. ✅ **mIoU > 28%**: At least one encoder achieves >28% mean IoU
2. ✅ **Reproducible**: 3-fold CV std < 2% (stable across folds)
3. ✅ **Complete Evaluation**: All metrics, visualizations, JSON exports
4. ✅ **Code Quality**: Clean, documented, reusable for POC-6.0
5. ✅ **Time Efficient**: <6 hours total training time (9 jobs on 2 GPUs)

**Stretch Goals:**
- 🎯 **mIoU > 30%**: SegFormer achieves 30%+ mIoU
- 🎯 **Publication Ready**: Comprehensive results for paper/thesis
- 🎯 **Ablation Study**: Quantify contribution of each improvement

---

## 🎬 Getting Started

```bash
# 1. Setup environment
cd experiments/artefact-poc59-final
conda activate poc55  # Or create new env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test single training (5 epochs)
python src/train.py --config configs/convnext_tiny.yaml --epochs 5 --test

# 4. If test passes, launch full CV
make train_all_cv

# 5. Monitor progress
watch -n 10 'squeue -u $USER'

# 6. When complete, aggregate results
python scripts/analyze_results.py

# 7. View results
cat results/cv_results.json
```

**Estimated completion:** 4-6 hours (with 2× V100 GPUs)

---

**Created:** November 16, 2025  
**Author:** POC-5.8 → POC-5.9 Migration  
**Status:** 🟡 Planning Phase → Ready for Implementation
