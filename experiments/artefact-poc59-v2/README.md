# POC-5.9-v2: Production Segmentation Pipeline

**Simple & Fast**: Train/val split (80/20) with RAM preloading for maximum throughput.

**Target mIoU:** 28-32% (improvement over POC-5.8 best: 24.93%)

---

## 📋 Quick Summary

- ✅ **Fast infrastructure** from POC-5.8 (83 imgs/s @ 384px)
- ✅ **Advanced loss**: DiceFocalLoss with pre-computed class weights
- ✅ **Modern encoders**: ConvNeXt (CNN) / SegFormer (ViT) / MaxViT (Hybrid)
- ✅ **Uniform resolution**: 384px for fair comparison
- ✅ **RAM preloading**: 30GB → 83 imgs/s throughput

---

## 🏗️ Architecture

### Encoders (Fair Comparison @ 384px)

| Encoder | Family | Core Mechanism | Params | Expected mIoU |
|---------|--------|---------------|--------|---------------|
| **ConvNeXt-Tiny** | **CNN** | Hierarchical convolutions | ~33M | 28-29% |
| **SegFormer MiT-B3** | **ViT** | Hierarchical self-attention | ~45M | **30-32%** ✨ |
| **MaxViT-Tiny** | **Hybrid** | Conv + Multi-axis attention | ~31M | 29-31% |

### Loss Function

```python
DiceFocalLoss = 0.5 × DiceLoss + 0.5 × FocalLoss
```

**With class weights** (inverse sqrt method) computed from 1,458 augmented images.

---

## 🚀 Quick Start

### 1. Test (1 epoch, ~45 sec per encoder)

```bash
# Test single encoder
sbatch scripts/slurm_test.sh configs/convnext_tiny.yaml

# Test all 3 encoders in parallel (if 3 GPUs available)
for encoder in convnext_tiny segformer_b3 maxvit_tiny; do
    sbatch scripts/slurm_test.sh configs/${encoder}.yaml
done
```

**Expected:**
- Preload time: ~30 sec
- Training time: ~13 sec (1 epoch)
- Throughput: ~83 imgs/s
- VRAM: ~0.5GB / 32GB (1.6%)

### 2. Full Training (50 epochs, ~15 min per encoder)

```bash
# Train single encoder
sbatch scripts/slurm_train.sh configs/convnext_tiny.yaml

# Train all 3 encoders in parallel
for encoder in convnext_tiny segformer_b3 maxvit_tiny; do
    sbatch scripts/slurm_train.sh configs/${encoder}.yaml
done
```

**Expected per encoder:**
- Preload time: ~30 min (one-time RAM loading)
- Training time: ~15 min (50 epochs @ 83 imgs/s)
- Total time: **~45 min**
- Total for 3 encoders (parallel): **~45 min** if 3 GPUs available

---

## 📁 Directory Structure

```
artefact-poc59-v2/
├── README.md                       # This file
├── configs/
│   ├── convnext_tiny.yaml         # CNN encoder
│   ├── segformer_b3.yaml          # ViT encoder  
│   └── maxvit_tiny.yaml           # Hybrid encoder
├── scripts/
│   ├── slurm_test.sh              # 1-epoch test
│   ├── slurm_train.sh             # 50-epoch training
│   └── evaluate_all.sh            # Batch evaluation
├── src/
│   ├── train.py                   # Training script
│   ├── losses.py                  # DiceFocalLoss
│   ├── dataset.py                 # Standard DataLoader
│   ├── preload_dataset.py         # RAM preloading
│   ├── model_factory.py           # Model creation
│   ├── timm_encoder.py            # Timm encoder wrapper
│   └── evaluate.py                # Evaluation script
└── logs/                          # Training outputs
```

---

## 🔧 Training Configuration

```yaml
training:
  batch_size: 96              # V100 32GB optimized
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.01
  mixed_precision: true       # AMP enabled
  
loss:
  type: dice_focal
  dice_weight: 0.5
  focal_weight: 0.5
  focal_gamma: 2.0
  focal_alpha: 0.25
  
optimizer:
  type: adamw
  
scheduler:
  type: onecycle
  max_lr: 0.001
```

---

## 📈 Expected Results

| Encoder | Family | mIoU | Throughput | Time (50 epochs) |
|---------|--------|------|------------|------------------|
| ConvNeXt-Tiny | CNN | 28-29% | 83 imgs/s | ~45 min |
| SegFormer-B3 | ViT | **30-32%** ✨ | ~80 imgs/s | ~48 min |
| MaxViT-Tiny | Hybrid | 29-31% | ~83 imgs/s | ~47 min |

**Validated Metrics (1-epoch test):**
- Throughput: 83.8 imgs/s (ConvNeXt-Tiny)
- VRAM: 0.52GB / 31.75GB (1.6%)
- RAM preload: 30.33GB (24.59 train + 5.74 val)

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
| **Loss Function** | Dice only | Dice+Focal + Weights |
| **Resolution** | Mixed (384/224px) | Uniform 384px |
| **Encoders** | ConvNeXt/Swin/CoAtNet | ConvNeXt/SegFormer/MaxViT |
| **Throughput** | 368 imgs/s (224px) | 83 imgs/s (384px) |
| **Expected mIoU** | 24.93% | **28-32%** |
| **Focus** | Speed test | Production accuracy |

---

## 📝 Notes

- **No K-fold CV**: Simplified to single train/val split for faster iteration
- **No shared memory**: RAM preload is optimal for sequential training
- **Batch size**: Can be increased (only using 1.6% VRAM), but 96 already achieves good throughput
- **Class weights**: Pre-computed using inverse sqrt method over full augmented dataset

---

## 🚦 Next Steps

1. ✅ **Test** all 3 encoders (1 epoch) - validate pipeline works
2. 🔄 **Train** all 3 encoders (50 epochs) - get production results  
3. 📊 **Compare** mIoU across encoders - identify best architecture
4. 🎯 **Analyze** per-class performance - find strengths/weaknesses
5. 🚀 **Deploy** best model for production use

---

*Last updated: November 16, 2025*  
*Status: Ready for production training*
