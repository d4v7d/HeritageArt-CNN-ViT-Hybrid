# HeritageArt-CNN-ViT-Hybrid

A hybrid CNN-ViT architecture for heritage artifact damage segmentation using the ARTeFACT dataset.

##  Repository Structure

```
HeritageArt-CNN-ViT-Hybrid/
├── pipeline/                           # Main training pipeline (planned)
│   ├── src/                           # Source code (models, datasets, training)
│   ├── configs/                       # Configuration files
│   ├── tests/                         # Unit tests
│   ├── requirements.txt               # Python dependencies
│   └── pyproject.toml                # Project metadata
│
├── experiments/                        # Research experiments and POCs
│   ├── artefact-data-obtention/      # ARTeFACT dataset downloader
│   │   └── data/demo/                # 50 samples (binary segmentation)
│   ├── artefact-multibackbone-upernet/ # **POC-5: Multi-backbone comparison ✅**
│   │   ├── scripts/                  # Training, evaluation, comparison
│   │   ├── configs/                  # Model configs (ConvNeXt, Swin, MaxViT)
│   │   ├── logs/                     # Training logs + evaluation results
│   │   └── README.md                 # POC-5 documentation
│   └── utils/                         # Utility scripts (CUDA test, etc.)
│
└── documentation/                      # Project documentation
    ├── main.tex                       # Research paper (LaTeX)
    └── From-Paper-to-Plan.md         # Implementation roadmap
```

---

## 🏆 POC-5 Results: Multi-Backbone Comparison

**Objective**: Compare CNN, Vision Transformer, and Hybrid architectures for heritage art damage detection.

### Architecture & Methodology

All models use:
- **Decoder**: UPerNet (Unified Perceptual Parsing Network)
  - PPM (Pyramid Pooling Module) at scales [1, 2, 3, 6]
  - FPN (Feature Pyramid Network) with 256 channels
  - Purpose: Fair comparison by isolating encoder differences
- **Dataset**: ARTeFACT 50 samples (40 train / 10 val)
- **Task**: Binary segmentation (Clean vs Damage)
- **Resolution**: 512×512
- **Training**: 60 epochs, batch size 4, AdamW optimizer (lr=0.0003)
- **Loss**: DiceFocalLoss (dice=0.7, focal=0.3)

### Results Summary

| 🏅 | Model | Type | mIoU | mF1 | Accuracy | Damage IoU | Params | Best Epoch |
|---|-------|------|------|-----|----------|------------|--------|------------|
| 🥇 | **MaxViT-Tiny** | **Hybrid** | **71.64%** | **82.13%** | **91.01%** | **53.29%** | 39.2M | 38 |
| 🥈 | Swin-Tiny | Transformer | 64.23% | 76.07% | 87.21% | 42.59% | 37.4M | 34 |
| 🥉 | ConvNeXt-Tiny | CNN | 63.67% | 75.52% | 87.16% | 41.48% | 37.7M | 38 |

### Key Findings

**🚀 MaxViT (Hybrid CNN+Multi-axis Attention) is the winner:**
- **+12.5% mIoU** vs ConvNeXt (CNN)
- **+11.5% mIoU** vs Swin (Transformer)
- **+28.5% Damage IoU** vs ConvNeXt
- **+25.1% Damage IoU** vs Swin

**Architecture Insights:**
1. **Hybrid > Transformer > CNN** for heritage art damage detection
2. **MaxViT** excels at detecting damage (53.29% Damage IoU)
3. All models converge around epoch 34-38
4. Small dataset (50 samples) → Hybrid better leverages ImageNet pretraining

**Per-Class Performance (MaxViT):**
- **Clean**: 89.99% IoU, 94.73% F1
- **Damage**: 53.29% IoU, 69.53% F1
- **Clean Precision**: 92.29% | **Damage Precision**: 82.09%
- **Clean Recall**: 97.30% | **Damage Recall**: 60.30%

**Detailed Results**: See `experiments/artefact-multibackbone-upernet/logs/comparison/`
- Training curves, metrics comparison, per-class IoU, convergence analysis
- Confusion matrices and prediction visualizations

---

## Prerequisites

- **Python**: 3.10 (tested on 3.10.18)
- **CUDA**: 11.8+ (for GPU training)
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended (RTX 3060 Ti or better)

---

## Installation Steps

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 2. Update pip tooling

```bash
python -m pip install -U pip setuptools wheel
```

### 3. **IMPORTANT: Install PyTorch FIRST** (before requirements.txt)

**Check your CUDA version:**
```bash
nvidia-smi  # Look for "CUDA Version: X.X"
```

**Then install the matching PyTorch build:**

```bash
# For CUDA 12.8+ (RTX 40 series, A100, etc.)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# For CUDA 12.6
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# For CUDA 11.8 (older GPUs: RTX 30 series, V100, etc.)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

# For ROCm 6.3 (AMD GPUs, Linux only)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/rocm6.3

# For CPU-only (no GPU)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

### 4. Verify PyTorch Installation

```bash
python experiments/utils/cuda-test.py
```

**Expected output:**
```
Torch: 2.7.1+cu128
torchvision: 0.22.1+cu128
CUDA available: True
Device count: 1
Device name: NVIDIA GeForce RTX 4060
```

 Verify:
- `CUDA available: True`
- Your GPU name appears correctly

### 5. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 6. Install OpenMMLab Components (via MIM)

```bash
pip install -U openmim
mim install mmengine==0.10.4
mim install mmcv==2.1.0
mim install mmsegmentation==1.2.2
```

> **Note:** MIM automatically selects the correct `mmcv` wheel for your PyTorch/CUDA version.

### 7. Install Pre-commit Hooks (for development)

```bash
pre-commit install
```

This enables automatic code formatting and linting before each commit.

---

## Download Model Checkpoints

Use `mim` to download pre-trained models and their configs:

### PSPNet + ResNet50 (ADE20K, demo purposes)

```bash
mim download mmsegmentation \
  --config pspnet_r50-d8_4xb4-80k_ade20k-512x512 \
  --dest ./experiments/mmseg_demos/_mmseg_demo
```

This downloads a lightweight model for testing MMSegmentation installation.

### Swin-Base + UPerNet (ADE20K, 512×512)

```bash
mim download mmsegmentation \
  --config swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512 \
  --dest checkpoints
```

- **Pretrain**: ImageNet-22k
- **Finetune**: ADE20K (160k iterations)
- **Resolution**: 512×512

### ConvNeXt-Large + UPerNet (ADE20K, 640×640, AMP)

```bash
mim download mmsegmentation \
  --config convnext-large_upernet_8xb2-amp-160k_ade20k-640x640 \
  --dest checkpoints
```

- **Pretrain**: ImageNet-22k
- **Finetune**: ADE20K (160k iterations)
- **Resolution**: 640×640
- **Training**: Mixed precision (AMP)

---

## Quick Start: Run Demos

### 1. Verify CUDA Setup

```bash
python experiments/utils/cuda-test.py
```

### 2. Test MMSegmentation Installation

```bash
python experiments/mmseg_demos/MMSeg-test.py
```

### 3. Run Segmentation Demo (Swin + ConvNeXt)

```bash
# Requires checkpoints downloaded above
python experiments/mmseg_demos/image_demo.py
```

This will:
1. Load Swin-Base + UPerNet model
2. Perform inference on `demo/demo.png`
3. Display results
4. Repeat with ConvNeXt-Large + UPerNet

---

## Project Structure

```
HeritageArt-CNN-ViT-Hybrid/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config (orchestrator)
│   ├── model/                 # Model architectures
│   │   ├── cnn.yaml          # ConvNeXt + UPerNet
│   │   ├── vit.yaml          # Swin + UPerNet
│   │   └── hybrid.yaml       # CoaT + UPerNet
│   ├── data/                  # Dataset configs
│   │   └── artefact.yaml     # ARTeFACT dataset
│   └── train/                 # Training hyperparameters
│       └── default.yaml
├── src/                       # Main pipeline (follows From-Paper-to-Plan.md)
│   ├── datasets/              # Dataset loaders
│   ├── models/                # Model definitions
│   └── infer/                 # Inference scripts
├── experiments/               # Quick experiments & demos (not part of main pipeline)
│   ├── utils/
│   │   └── cuda-test.py      # CUDA verification
│   └── mmseg_demos/
│       ├── MMSeg-test.py     # MMSeg installation test
│       └── image_demo.py     # Segmentation demo
├── tests/                     # Unit tests for src/
├── checkpoints/               # Downloaded model weights (gitignored)
├── data/                      # ARTeFACT dataset (gitignored)
├── outputs/                   # Training outputs (gitignored)
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Tool configurations (black, isort, pytest)
├── .pre-commit-config.yaml   # Pre-commit hooks
└── .flake8                   # Linting rules
```

---

## Development Workflow

1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Make code changes** in `src/`

3. **Run pre-commit checks manually** (optional):
   ```bash
   pre-commit run --all-files
   ```

4. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```
   Pre-commit hooks will automatically:
   - Format code with Black
   - Lint with Flake8
   - Sort imports with isort
   - Fix trailing whitespace
   - Validate YAML/JSON/TOML

---

## Generate Requirements Lock File

To create a strict, machine-specific dependency snapshot:

```bash
pip freeze --exclude-editable > requirements.lock.txt
```

Use this for exact reproducibility on the same machine/environment.

---

## Troubleshooting

### PyTorch not using GPU

```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision -y
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

### MMSegmentation import errors

```bash
# Reinstall OpenMMLab stack
pip uninstall mmengine mmcv mmsegmentation -y
pip install -U openmim
mim install mmengine==0.10.4
mim install mmcv==2.1.0
mim install mmsegmentation==1.2.2
```

### Pre-commit hooks failing

```bash
# Reinstall pre-commit
pre-commit clean
pre-commit install
pre-commit run --all-files
```

---

## License

[Add your license here]

## Citation

If you use this code, please cite:

```bibtex
[Add citation here]
```

---

## Acknowledgments

- **ARTeFACT Dataset**: [Link to dataset paper]
- **MMSegmentation**: OpenMMLab semantic segmentation toolbox
- **PyTorch**: Deep learning framework
