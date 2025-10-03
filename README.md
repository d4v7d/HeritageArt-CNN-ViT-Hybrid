# HeritageArt-CNN-ViT-Hybrid

# Setup
Versión de Python: `3.10.18`

Updatear pip tooling:
```
python -m pip install -U pip setuptools wheel
```

Según la versión de CUDA (depende del GPU) escoger UNICAMENTE UNO de los siguientes a instalar:
```
# CUDA 11.8
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```
```
# CUDA 12.6
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```
```
# CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```
```
# ROCm 6.3 (AMD, Linux)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/rocm6.3
```
```
# CPU-only
pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu
```
Verifique funcionalidad y versiones corriendo cuda-test.py

Ahora use `OpenMIM` para escoger el mmcv wheel correcto según la versión de Torch/CUDA:

```
# install base packages 
pip install -r requirements.txt

# let MIM resolve the correct mmcv for Torch/CUDA
pip install -U openmim
mim install mmengine==0.10.4
# mim install "mmcv>=2.0.0"
mim install "mmcv==2.1.0"
# mim install "mmsegmentation>=1.2.0"
mim install "mmsegmentation==1.2.2"


# download the config (and optionally the checkpoint) locally
mim download mmsegmentation \
  --config pspnet_r50-d8_4xb4-80k_ade20k-512x512 \
  --dest ./_mmseg_demo
``` 

Para uso posterior, específico a la máquina:
```
# strict, exact pins
pip freeze --exclude-editable > requirements.lock.txt
```

```
Hay que unificar formatos lol
## Setup

### Prerequisites
- Python 3.10+ (tested on 3.12)
- CUDA 11.8+ (for GPU training)

### Installation

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install pre-commit hooks:
```bash
pre-commit install
```
