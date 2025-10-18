# POC-5: Multi-backbone UPerNet Comparison

**Objective**: Compare modern architectures (CNN, ViT, Hybrid) for heritage art damage detection using unified UPerNet decoder.

## 🏗️ Architecture Comparison

| Model | Type | Encoder | Decoder | Params | ImageNet Top-1 |
|-------|------|---------|---------|--------|----------------|
| **ConvNeXt-Tiny** | CNN | ConvNeXt-Tiny | UPerNet | ~50M | 82.1% |
| **Swin-Tiny** | ViT | Swin Transformer | UPerNet | ~60M | 81.3% |
| **CoaT-Lite-Small** | Hybrid | CoaT (Conv+Attn) | UPerNet | ~45M | 77.5% |

### UPerNet Decoder (Unified)
- **PPM (Pyramid Pooling Module)**: Multi-scale context at scales [1, 2, 3, 6]
- **FPN (Feature Pyramid Network)**: Multi-level feature fusion (256 channels)
- **Purpose**: Fair comparison by isolating encoder differences

## 📁 Directory Structure

```
artefact-multibackbone-upernet/
├── configs/
│   ├── base_config.yaml              # Common training settings
│   ├── convnext_tiny_upernet.yaml    # ConvNeXt-Tiny config
│   ├── swin_tiny_upernet.yaml        # Swin-Tiny config
│   └── coat_lite_small_upernet.yaml  # CoaT-Lite-Small config
├── docker/
│   ├── Dockerfile                    # CUDA 12.6 + PyTorch 2.0+
│   ├── requirements.txt              # timm, einops, albumentations, etc.
│   └── docker-compose.yml            # GPU-enabled container
├── scripts/
│   ├── dataset.py                    # ARTeFACT dataloader (symlink to POC-4)
│   ├── train.py                      # Training script (supports all 3 models)
│   ├── evaluate.py                   # Evaluation script (TODO)
│   ├── compare.py                    # Cross-model comparison (TODO)
│   ├── visualize_attention.py        # Attention visualization (TODO)
│   └── models/
│       ├── __init__.py               # Package exports
│       ├── upernet_custom.py         # Custom UPerNet (PPM + FPN)
│       └── model_factory.py          # Build models from timm encoders
├── data/                             # Symlink to POC-1 demo (50 samples)
├── logs/                             # Training logs and checkpoints
│   ├── convnext_tiny_upernet/
│   ├── swin_tiny_upernet/
│   └── coat_lite_small_upernet/
├── Makefile                          # Automation commands
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Build Docker Image

```bash
make build
```

### 2. Train Individual Models

```bash
# Train ConvNeXt-Tiny (~20 min, 60 epochs)
make train-convnext

# Train Swin-Tiny (~25 min, 60 epochs)
make train-swin

# Train CoaT-Lite-Small (~22 min, 60 epochs)
make train-coat

# Or train all sequentially (~1h 7min)
make train-all
```

### 3. Evaluate Models

```bash
# Evaluate each model
make evaluate-convnext
make evaluate-swin
make evaluate-coat

# Or evaluate all
make evaluate-all
```

### 4. Compare Results

```bash
# Generate comparison table and visualizations
make compare

# Generate attention maps (Swin + CoaT only)
make visualize-attention
```

## 📊 Dataset

- **Source**: ARTeFACT (10K heritage art damage annotations)
- **Samples**: 50 images (40 train / 10 val)
- **Classes**: 2 (Clean, Damage) - binary segmentation
- **Resolution**: 512×512
- **Format**: RGB images + binary masks

**Data path**: `data/artefact/` → symlinked to `../../artefact-data-obtention/data/demo/`

## ⚙️ Training Configuration

**Inherited from POC-4 optimizations** (ResNet50-UNet baseline: mIoU 0.5721):

- **Epochs**: 60
- **Batch size**: 4
- **Learning rate**: 0.0003 (AdamW)
- **Warmup**: 8 epochs (linear)
- **Scheduler**: Cosine annealing (min_lr=0.0000005)
- **Loss**: DiceFocalLoss (dice=0.7, focal=0.3, alpha=0.25, gamma=2.0)
- **Augmentation**: 
  - Horizontal/vertical flip (50%)
  - Rotate ±20°
  - Brightness/contrast ±0.25
  - Gaussian blur (kernel 4)
- **Mixed precision**: Enabled (AMP)
- **Checkpointing**: Every 10 epochs + best model

## 🔬 Technical Details

### Custom UPerNet Implementation

**Why custom?**: `segmentation-models-pytorch` doesn't support timm encoders with UPerNet natively.

**Components**:

1. **PPM (Pyramid Pooling Module)**:
   - Adaptive pooling at scales [1, 2, 3, 6]
   - Upsamples and concatenates multi-scale features
   - Bottleneck fusion (3×3 conv)

2. **FPN (Feature Pyramid Network)**:
   - Lateral connections (1×1 conv) to unify channels
   - Top-down pathway with bilinear upsampling
   - Skip connections from encoder stages
   - Output conv (3×3 conv) per level

3. **Decoder Flow**:
   ```
   encoder_features (4 stages) → PPM on stage4 → FPN fusion → Concat all levels → Classifier
   ```

### Model Factory

**Purpose**: Load timm encoders and attach custom UPerNet decoder.

**Key features**:
- Auto-detects feature channels from encoder
- Supports any timm model with `features_only=True`
- Pre-configured for ConvNeXt, Swin, CoaT channel dimensions

## 📈 Expected Results

Based on POC-4 baseline (ResNet50-UNet: mIoU 0.5721) and literature:

| Model | Expected mIoU | Rationale |
|-------|---------------|-----------|
| **ConvNeXt-Tiny** | 0.63-0.66 | Modern CNN, +6% ImageNet over ResNet50 |
| **Swin-Tiny** | 0.65-0.68 | Hierarchical ViT, global context |
| **CoaT-Lite-Small** | 0.64-0.67 | Hybrid, parallel conv+attention |

**Hypothesis**: Swin or CoaT will outperform ConvNeXt due to better global context modeling for damage patterns.

## 🎯 Deliverables

- [x] Docker environment (CUDA 12.6 + timm)
- [x] Custom UPerNet implementation (PPM + FPN)
- [x] Model factory for timm encoders
- [x] Training script with mixed precision
- [ ] Evaluation script with per-class metrics
- [ ] Comparison script (side-by-side predictions)
- [ ] Attention visualization (Swin + CoaT)
- [ ] Results analysis and findings

## 📖 References

1. **UPerNet**: [Unified Perceptual Parsing for Scene Understanding](https://arxiv.org/abs/1807.10221)
2. **ConvNeXt**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
3. **Swin Transformer**: [Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
4. **CoaT**: [Co-Scale Conv-Attentional Image Transformers](https://arxiv.org/abs/2104.06399)
5. **ARTeFACT**: [A Large-Scale Dataset for Benchmark on Art Damage Detection](https://arxiv.org/abs/2305.12549)

## 🔧 Development Commands

```bash
# Build Docker image
make build

# Train specific model
docker-compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
    python scripts/train.py --config configs/convnext_tiny_upernet.yaml

# Resume training from checkpoint
docker-compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
    python scripts/train.py --config configs/swin_tiny_upernet.yaml \
    --resume logs/swin_tiny_upernet/checkpoints/checkpoint_epoch_30.pth

# Test UPerNet decoder
docker-compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
    python scripts/models/upernet_custom.py

# Test model factory
docker-compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
    python scripts/models/model_factory.py

# Clean logs and checkpoints
make clean
```

## 📝 TODO

- [ ] Create `evaluate.py` (metrics + visualizations)
- [ ] Create `compare.py` (cross-model comparison)
- [ ] Create `visualize_attention.py` (attention maps)
- [ ] Run all 3 training sessions
- [ ] Generate comparison results
- [ ] Analyze findings and document insights
- [ ] Commit and push to GitHub

---

**Status**: 🚧 In Development (Phase 1: Setup Complete)  
**POC-4 Baseline**: ResNet50-UNet mIoU 0.5721  
**Next**: Test UPerNet implementation, then start training
