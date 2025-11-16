# POC-5.8: Standard Segmentation Benchmark (Server-Optimized)

**Created**: November 2025  
**Status**: ✅ Production Ready  
**Environment**: CITIC Server (Tesla V100S-32GB × 2)

---

## 🎯 Objective

Benchmark **CNN vs ViT vs Hybrid** architectures for heritage art damage segmentation using:
- ✅ Industry-standard library (Segmentation Models PyTorch)
- ✅ Proven UNet decoder (simple, efficient)
- ✅ Modern encoders (ConvNeXt, Swin, CoAtNet)
- ✅ Mixed Precision (AMP) for 2x speedup
- ✅ Optimized for Tesla V100 GPUs

**Why POC-5.8?**  
POC-5.5 validated multi-task hierarchical learning on laptop. POC-5.8 focuses on **fair encoder comparison** with simple architecture on server hardware.

---

## 📊 Architecture Comparison (50 epochs)

| Model | Type | Params | Image Size | mIoU (expected) | Throughput |
|-------|------|--------|------------|-----------------|------------|
| **ConvNeXt-Tiny** | CNN | 33.1M | 384×384 | ~28-30% | ~24 imgs/s |
| **Swin-Tiny** | ViT | 32.8M | 224×224 | ~29-31% | ~25 imgs/s |
| **CoAtNet-0** | Hybrid | 30.8M | 224×224 | ~30-32% | ~23 imgs/s |

**Hardware:**
- GPU: Tesla V100S-PCIE-32GB (×2 for parallel training)
- VRAM Usage: ~1.6% (0.52GB per model)
- Training Time: ~15 min per model (50 epochs)

---

## 🚀 Quick Start

### Test Single Model (1 epoch)

```bash
cd ~/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standard

# Test ConvNeXt
sbatch --gres=gpu:1 scripts/slurm_train.sh configs/convnext_tiny.yaml --test-epoch

# Monitor progress
tail -f logs/train_*.out
```

### Train All Models (50 epochs, parallel)

```bash
# Launch all 3 models (2 parallel + 1 sequential)
bash scripts/train_all_parallel.sh

# Monitor all jobs
squeue -u $USER
```

### Evaluate Results

```bash
# Evaluate all trained models
bash scripts/evaluate_all.sh

# Results in: logs/Unet_tu-{model}/evaluation/
```

---

## 📁 Project Structure

```
artefact-poc58-standard/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── configs/
│   ├── convnext_tiny.yaml      # CNN baseline
│   ├── swin_tiny.yaml          # ViT baseline
│   └── coatnet_0.yaml          # Hybrid CNN+ViT
├── scripts/
│   ├── slurm_train.sh          # SLURM job script
│   ├── train_all_parallel.sh   # Parallel training (recommended)
│   └── evaluate_all.sh         # Batch evaluation
├── src/
│   ├── train.py                # Training loop (AMP + DataParallel)
│   ├── evaluate.py             # Evaluation script
│   ├── dataset.py              # Standard dataloader
│   ├── preload_dataset.py      # RAM pre-loading (optional)
│   ├── model_factory.py        # Model creation
│   └── timm_encoder.py         # Custom timm wrapper
└── logs/
    └── Unet_tu-{model}/        # Training logs + checkpoints
```

---

## 🏗️ Technical Architecture

### Encoder-Decoder Pipeline

```
Input: (B, 3, H, W)
    ↓
[Encoder: ConvNeXt/Swin/CoAtNet from timm]
    ├─ Stage 1: 96 channels
    ├─ Stage 2: 192 channels  
    ├─ Stage 3: 384 channels
    └─ Stage 4: 768 channels
    ↓
[UNet Decoder: Skip Connections]
    ├─ Up 1: 768 → 384 (+ skip)
    ├─ Up 2: 384 → 192 (+ skip)
    ├─ Up 3: 192 → 96 (+ skip)
    └─ Up 4: 96 → 16 (output)
    ↓
Output: (B, 16, H, W) - 16 damage classes
```

### Training Configuration

- **Loss**: DiceLoss (multiclass, smooth=1.0)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: OneCycleLR (30% warmup, cosine annealing)
- **AMP**: Automatic Mixed Precision (FP16/FP32)
- **Augmentation**: HFlip, VFlip, Rotate90, Normalize

---

## 🔧 Dataset

- **Source**: ARTeFACT (Desmoulins et al., 2024)
- **Augmented**: 1,463 images (from 418 original)
- **Classes**: 16 damage types
- **Split**: 80/20 train/val (1,170 / 293)
- **Resolution**: 384×384 (ConvNeXt), 224×224 (Swin, CoAtNet)

---

## 📈 Monitoring Training

### Check Job Status

```bash
# All your jobs
squeue -u btrigueros

# Detailed status
scontrol show job <JOB_ID>
```

### View Progress (Multiple Jobs)

```bash
# Option 1: Two terminals
tail -f logs/train_XXXX.out  # Job 1
tail -f logs/train_YYYY.out  # Job 2

# Option 2: Watch both
watch -n 2 'tail -20 logs/train_*.out | grep -E "Epoch|mIoU|Throughput"'
```

### Expected Output

```
Epoch 25/50 (12.4s)
  Train - Loss: 0.7245
  Val   - Loss: 0.6892
         mIoU: 0.2847
  Throughput: 24.1 imgs/s
  VRAM: 0.52GB / 31.75GB (1.6%)
  ✅ New best mIoU: 0.2847
```

---

## 🔄 Comparison with POC-5.5

| Aspect | POC-5.5 (Laptop) | POC-5.8 (Server) |
|--------|------------------|------------------|
| **Hardware** | RTX 3050 6GB | Tesla V100S 32GB |
| **Objective** | Hierarchical multi-task | Encoder comparison |
| **Architecture** | Custom UPerNet | SMP UNet |
| **Encoders** | ConvNeXt, Swin, MaxViT | ConvNeXt, Swin, CoAtNet |
| **Tasks** | Binary + Coarse + Fine | Fine only (16 classes) |
| **Dataset** | 334 images | 1,463 images |
| **Batch Size** | 8-16 | 96 |
| **Training Time** | ~4 hours | ~15 min |
| **mIoU Target** | 22% | 28-32% |
| **Innovation** | Multi-task learning | Fair benchmark |

---

## 🚨 Troubleshooting

### OOM Errors

Reduce batch size in configs:
```yaml
training:
  batch_size: 64  # from 96
```

### DataParallel Issues

Already fixed with `CUDA_VISIBLE_DEVICES=0` in `slurm_train.sh`

### Low Throughput

Enable RAM pre-loading for long training:
```yaml
data:
  use_preload: true
```

---

## 📚 References

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [ConvNeXt](https://arxiv.org/abs/2201.03545)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [CoAtNet](https://arxiv.org/abs/2106.04803)

---

**Status**: ✅ Ready for benchmarking  
**Next**: Run 50-epoch training, compare results
