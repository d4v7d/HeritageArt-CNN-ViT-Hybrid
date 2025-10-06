# Experiments & Proof-of-Concept Tests

This folder contains quick experiments and proof-of-concept code that **doesn't follow the main pipeline** defined in `From-Paper-to-Plan.md`.

## Structure

- `utils/` - System checks and diagnostics
  - `cuda-test.py` - Verifica instalación de PyTorch + CUDA
  
- `mmseg_demos/` - Pruebas con MMSegmentation
  - `MMSeg-test.py` - Test de carga de modelo MMSeg
  - `image_demo.py` - Demo de inferencia con Swin y ConvNeXt

## Usage

Estos scripts son **standalone** y no afectan el pipeline principal en `src/`.

### Verificar CUDA
```bash
python experiments/utils/cuda-test.py
```

### Probar MMSegmentation
```bash
# 1. Test básico de carga
python experiments/mmseg_demos/MMSeg-test.py

# 2. Demo de segmentación (requiere checkpoints descargados)
python experiments/mmseg_demos/image_demo.py
```

## Notes

- Pre-commit hooks **NO** corren en esta carpeta (ver `.pre-commit-config.yaml`)
- Estos experimentos pueden usar cualquier estructura de código
- Para desarrollo formal, seguir el plan en `From-Paper-to-Plan.md` → implementar en `src/`