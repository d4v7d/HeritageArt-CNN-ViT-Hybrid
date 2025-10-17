# POC-4: ARTeFACT Training Minimal

**Objective**: Validate end-to-end training with real ARTeFACT data using a simple CNN baseline.

**Status**: ✅ COMPLETED - Binary segmentation (Damage vs Clean) with ResNet50-UNet on 50 samples.

## Results Summary

| Configuration | Samples | Best mIoU | Best mF1 | Accuracy | Epoch | Training Time |
|--------------|---------|-----------|----------|----------|-------|---------------|
| Baseline | 10 | 0.4487 | 0.5007 | 79.82% | 10 | ~5 min |
| Extended | 10 | 0.5500 | 0.6178 | 86.45% | 15 | ~10 min |
| **Optimized** | **50** | **0.5721** | **0.6851** | **84.68%** | **37** | **~20 min** |

**Key Achievements:**
- ✅ **Best mIoU: 0.5721** (exceeds target > 0.5)
- ✅ **27.5% improvement** over baseline
- ✅ **Dataset scaling validated**: 10 → 50 samples
- ✅ **Transfer learning confirmed**: ImageNet pretrained encoder effective
- ✅ **Production-ready pipeline**: Docker + automated eval + visualizations

## What This POC Does

1. **Loads real ARTeFACT data** from POC-1 (50 validated samples)
2. **Trains a ResNet50-UNet** model for binary segmentation (32.5M parameters)
3. **Binary task**: Damage (class 1) vs Clean (class 0)
4. **512x512 input size** with optimized data augmentations
5. **Train/Val split**: 80/20 (40 train, 10 val)
6. **Metrics**: mIoU, F1, Accuracy with per-epoch tracking
7. **Checkpointing**: Saves best model and periodic checkpoints (every 5 epochs)
8. **Evaluation**: Generates comprehensive metrics and 4-panel visualizations
9. **Inference**: Demo script for single image prediction
10. **Transfer Learning**: Leverages ImageNet pretrained ResNet50 encoder

## Project Structure

```
artefact-training-minimal/
├── docker/
│   ├── Dockerfile           # CUDA 12.6 + PyTorch environment
│   ├── docker-compose.yml   # Container orchestration
│   └── requirements.txt     # Python dependencies
├── scripts/
│   ├── dataset.py          # ARTeFACT dataset loader
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation + visualization
│   └── infer_demo.py       # Single image inference
├── configs/
│   └── train_config.yaml   # Training configuration
├── data/
│   └── artefact/           # Symlink to POC-1 demo data
├── logs/
│   ├── checkpoints/        # Model checkpoints
│   ├── training/           # Training logs
│   ├── evaluation/         # Evaluation results
│   └── visualizations/     # Output visualizations
├── Makefile                # Convenience commands
├── .gitignore
└── README.md               # This file
```

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support
- POC-1 completed with demo data available

### 1. Build Docker Image

```bash
make build
```

This builds a container with:
- CUDA 12.6 runtime
- PyTorch 2.1+
- segmentation-models-pytorch
- albumentations
- All required dependencies

### 2. Setup Data

```bash
make setup-data
```

Creates a symlink: `data/artefact` → `../artefact-data-obtention/data/demo/`

This reuses the 10 validated samples from POC-1.

### 3. Train Model

**Baseline (10 epochs):**
```bash
make train
```

**Extended (50 epochs):**
```bash
make train-extended
```

**Optimized (60 epochs, recommended):**
```bash
make train-optimized
```

**Optimized configuration**:
- **Model**: ResNet50-UNet (pretrained ImageNet encoder, 32.5M params)
- **Loss**: Dice (0.7) + Focal (0.3) - optimized for mIoU
- **Optimizer**: AdamW (lr=3e-4, wd=5e-5)
- **Scheduler**: CosineAnnealing with 8-epoch warmup, min_lr=5e-7
- **Batch size**: 4
- **Augmentations**: Moderate (rotate 20°, brightness/contrast 0.25, blur, flips)

**Expected output**:
```
Epoch 37 Summary:
  Train - Loss: 0.2035, mIoU: 0.7057, mF1: 0.8043
  Val   - Loss: 0.2957, mIoU: 0.5721, mF1: 0.6851
  LR: 0.000097

Saved checkpoint: logs/checkpoints/checkpoint_epoch_37.pth
Saved best model: logs/checkpoints/best_model.pth (mIoU: 0.5721)
```

**Training time**: ~20 minutes on RTX 3090 (60 epochs, 50 samples)

### 4. Evaluate Model

```bash
make evaluate
```

Generates:
- **Metrics JSON**: `logs/evaluation/metrics.json`
- **Visualizations**: `logs/evaluation/visualizations/` (10 validation samples)
  - 4-panel plots: Input Image | Ground Truth | Prediction | Overlay

**Actual results (50 samples, optimized config)**:
```
EVALUATION RESULTS
==================
  mIoU:     0.5721 ± 0.0658
  mF1:      0.6851 ± 0.0693
  Accuracy: 0.8468
  Samples:  10
```

**Success criteria**: **mIoU > 0.5** ✅ ACHIEVED

**Visualization panels explained:**
- **Input Image**: Original artefact photograph
- **Ground Truth**: Expert-labeled damage masks (Red = Damage, Black = Clean)
- **Prediction**: Model's pixel-wise prediction (Red = Damage, Black = Clean)
- **Overlay**: Input with predicted damage overlaid in red for context

### 5. Run Inference Demo

```bash
make infer
# Enter image path when prompted
```

Example:
```bash
Enter image path: data/artefact/images/sample_001.png

Results:
  Damage pixels: 12,345 / 262,144 (4.71%)

Saved visualization to: logs/demo_inference.png
```

## Configuration

Three configurations available:

**1. `train_config.yaml` - Baseline (10 epochs)**
```yaml
model:
  encoder: "resnet50"
  classes: 2           # Binary mode

training:
  batch_size: 4
  epochs: 10
  learning_rate: 0.0001
  
  loss:
    dice_weight: 0.6
    focal_weight: 0.4

data:
  tile_size: 512
  train_val_split: 0.8
  binary_mode: true
```

**2. `train_config_extended.yaml` - Extended (50 epochs)**

Same as baseline but with 50 epochs for better convergence.

**3. `train_config_optimized.yaml` - Optimized (60 epochs) ⭐ RECOMMENDED**
```yaml
model:
  encoder: "resnet50"
  classes: 2

training:
  batch_size: 4
  epochs: 60
  learning_rate: 0.0003      # Reduced for stability
  weight_decay: 0.00005      # Reduced for small dataset
  
  loss:
    dice_weight: 0.7         # Increased (favors mIoU)
    focal_weight: 0.3        # Decreased
  
  scheduler:
    type: "cosine"
    warmup_epochs: 8         # Increased warmup
    min_lr: 0.0000005

augmentation:
  train:
    rotate_limit: 20         # Moderate rotation
    brightness_limit: 0.25   # Moderate brightness
    contrast_limit: 0.25     # Moderate contrast
    blur_limit: 4
    horizontal_flip: 0.5
    vertical_flip: 0.5

data:
  tile_size: 512
  train_val_split: 0.8
  binary_mode: true
```

**Rationale for optimized config:**
- Lower LR (0.0003) prevents overshooting with small dataset
- Dice weight 0.7 prioritizes IoU metric alignment
- Extended warmup (8 epochs) stabilizes early training
- Moderate augmentations prevent unrealistic distortions
- 60 epochs allow full convergence without overfitting

## Binary Mode

ARTeFACT has 16 classes (0=Clean, 1-15=Damage types, 255=Background).

In **binary mode**:
- **Class 0**: Clean (preserved)
- **Classes 1-15**: → **Class 1** (Damage)
- **Class 255**: Ignored in loss/metrics

This simplifies the task for initial validation.

## Dataset

Uses **POC-1 demo data** (expanded):
- **50 samples** total (downloaded via HuggingFace streaming)
- **40 train** / **10 validation** (80/20 split)
- **Materials**: Tesserae (34%), Parchment (16%), Paper (14%), Film emulsion (10%), Glass (10%)
- **Damage distribution**: Clean (85%), Peel (8.3%), Material loss (1.6%), Staining (1.35%), Cracks (1.3%)
- **Image sizes**: Variable (resized to 512×512 for training)

Each sample:
- RGB image (PNG, variable size: 341×512 to 1024×1024)
- Annotation mask (PNG, same size, pixel-wise labels)
- Metadata (material, content, damage types, bounding boxes)

## Output Files

### Checkpoints
- `logs/checkpoints/best_model.pth` - Best validation mIoU
- `logs/checkpoints/checkpoint_epoch_N.pth` - Every 5 epochs

### Metrics
- `logs/evaluation/metrics.json` - Final evaluation metrics

### Visualizations
- `logs/evaluation/visualizations/sample_*.png` - 4-panel comparisons
- `logs/demo_inference.png` - Single image demo output

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size: 2  # or 1
```

### Data Not Found

Ensure POC-1 demo data exists:
```bash
ls ../artefact-data-obtention/data/demo/images/
# Should show 10 PNG files
```

### Poor Performance (mIoU < 0.5)

Possible causes:
1. **Insufficient data**: Only 10 samples (expected for POC)
2. **Class imbalance**: Adjust loss weights
3. **Augmentation too aggressive**: Reduce in config
4. **Learning rate**: Try 5e-5 or 2e-4

## Advanced Usage

### Resume Training

```bash
make train-resume
# Enter checkpoint path when prompted
```

### Custom Training

```bash
cd docker
docker-compose run --rm artefact-training \
  python3 scripts/train.py \
  --config configs/custom_config.yaml \
  --resume logs/checkpoints/checkpoint_epoch_5.pth
```

### Inspect Data

```bash
make shell
python3 -c "
from scripts.dataset import verify_dataset
verify_dataset('data/artefact')
"
```

## Key Findings

### Transfer Learning Effectiveness
- **ImageNet pretraining** enables learning with limited data (50 samples)
- **Encoder frozen initially** not required - fine-tuning end-to-end works well
- **Train-val gap** (~0.19 mIoU) indicates slight overfitting but acceptable for dataset size

### Hyperparameter Impact
- **Dice/Focal ratio**: 0.7/0.3 optimal for mIoU metric (vs 0.6/0.4 baseline)
- **Learning rate**: 3e-4 balances speed and stability (5e-4 too aggressive)
- **Warmup**: 8 epochs critical for stable convergence with small datasets
- **Augmentation**: Moderate transforms (rotate 20°) prevent unrealistic samples

### Data Scaling Results
| Samples | Train/Val | mIoU | Improvement | Validation Stability |
|---------|-----------|------|-------------|---------------------|
| 10 | 8/2 | 0.4487-0.5500 | Baseline | High variance (2 val samples) |
| 50 | 40/10 | **0.5721** | +27.5% | Stable (10 val samples) |

**Recommendation**: 50+ samples minimum for reliable metrics. Next step: 100-200 samples for production.

## Next Steps (POC-5)

This POC validates:
- ✅ ARTeFACT data is trainable with CNNs
- ✅ Training loop is stable and reproducible
- ✅ Metrics (mIoU, F1) computed correctly
- ✅ Docker environment works on CUDA 12.6
- ✅ Transfer learning from ImageNet effective
- ✅ Binary segmentation baseline established (mIoU 0.5721)

**POC-5 will add**:
1. **UPerNet decoder** (common for all backbones)
2. **Multiple backbones**: ResNet50 (CNN), Swin-Tiny (ViT), CoaT-Lite (Hybrid)
3. **Architectural comparison** (CNN vs ViT vs Hybrid on same task)
4. **Multiclass mode** (16 damage classes vs binary)
5. **Attention visualization** (understand what models focus on)

## References

- **ARTeFACT Dataset**: [Ivanova et al., WACV 2025](https://doi.org/10.1109/WACV48630.2025.00104)
- **Segmentation Models PyTorch**: [qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- **Albumentations**: [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)

## License

This POC is part of the HeritageArt-CNN-ViT-Hybrid research project.
See repository root for license information.

---

**POC-4 Status**: Ready for execution ✅
**Target**: Validate ARTeFACT training end-to-end
**Success Metric**: mIoU > 0.5 on validation set
