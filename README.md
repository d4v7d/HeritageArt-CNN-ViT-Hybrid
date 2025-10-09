# HeritageArt-CNN-ViT-Hybrid

A hybrid CNN-ViT architecture for heritage artifact damage segmentation using the ARTeFACT dataset.

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

✅ Verify:
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
