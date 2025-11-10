# POC-5.8: Standard Segmentation Pipeline# POC-5.8: Standard Segmentation Pipeline (Server-Optimized)



## 🎯 Objetivo## Objetivo



Implementar pipeline de segmentación **estándar y robusto** usando librerías probadas industrialmente, tras múltiples fallos con código custom (POC5.5-5.7).Validar que el servidor V100 puede correr segmentación semántica **correctamente** usando arquitecturas y técnicas estándar probadas. Abandonar custom implementations problemáticas y usar **best practices** de la industria.



## 📊 Resultados---



### Test 1 Época (Job 2060)## Filosofía: Keep It Simple, Stupid (KISS)

- **Throughput**: 97.0 imgs/s (3.7x mejora vs baseline 26 imgs/s)

- **Tiempo/época**: 12.1s (train) + 3.0s (val) = 15.1s totalPOC5.7 fracasó porque:

- **RAM usage**: 32.8 GB (27GB train + 5.8GB val)- ❌ UPerNet custom con 3 heads → 30GB activations intermedias

- **VRAM usage**: 1.3% (0.41 GB / 31.75 GB)- ❌ Sin mixed precision → 2x VRAM desperdiciada

- **mIoU**: 0.0588 (6% - normal en 1 época)- ❌ Arquitectura compleja para 418 imágenes → overkill

- ❌ Debugging infinito de código custom

### Comparativa POC5.5 (Laptop Baseline)

| Métrica | POC5.5 Laptop | POC5.8 Server | Mejora |POC5.8 solución:

|---------|--------------|---------------|---------|- ✅ Usar `segmentation-models-pytorch` (SMP) - librería estándar, probada

| Arquitectura | Custom UPerNet | DeepLabV3+ (SMP) | ✅ Standard |- ✅ Mixed Precision (AMP) desde día 1 → 50% menos VRAM

| Throughput | 4 imgs/s | 97 imgs/s | **24x** |- ✅ Arquitectura simple primero (U-Net) → complejidad incremental

| mIoU (final) | 22% | TBD (50 epochs) | - |- ✅ Código mínimo, máximo aprovechamiento de librería

| Dataset | 334 images | 1,464 images | **4.4x** |

---

## 🔧 Stack Tecnológico

## Arquitectura

### Core Libraries

- **PyTorch**: 2.0.1+cu118### Fase 1: Baseline U-Net (30 min - ESTA FASE)

- **SMP**: segmentation-models-pytorch 0.3.3

- **Albumentations**: Data augmentation```

- **AMP**: Automatic Mixed Precision (torch.cuda.amp)Input (3, 384, 384)

    ↓

### ArquitecturaEncoder: ConvNeXt-Tiny (pretrained)

- **Model**: DeepLabV3Plus (ASPP decoder, memory efficient)    ├─ Stage 1: 96 channels

- **Encoder**: ResNet50 (26.7M params, ImageNet pretrained)    ├─ Stage 2: 192 channels

- **Loss**: DiceLoss multiclass    ├─ Stage 3: 384 channels

- **Optimizer**: AdamW + OneCycleLR    └─ Stage 4: 768 channels (bottleneck)

- **Batch size**: 64 (26.7M params × 64 = ~1.7GB activations)    ↓

Decoder: U-Net Skip Connections

### Dataset    ├─ Up 1: 768 → 384 (+ skip)

- **Original**: ARTeFACT 418 images, 16 damage classes    ├─ Up 2: 384 → 192 (+ skip)

- **Augmented**: 1,464 images (3x offline multiplier)    ├─ Up 3: 192 → 96 (+ skip)

- **Augmentations**: HFlip, VFlip, Rotate90/180/270    └─ Up 4: 96 → 16 (final)

- **Split**: 80/20 train/val (1,171 / 293)    ↓

Output: (16, 384, 384) - Fine classes

### Innovación: RAM Pre-loading```

- **Problema**: 80% tiempo en CPU I/O (decode PNG + augmentations)

- **Solución**: Pre-cargar TODAS las imágenes en RAM al inicio**Estimación:**

- **Implementación**: `PreloadedArtefactDataset` en `src/preload_dataset.py`- Parámetros: ~30M

- **Resultado**: I/O → 0%, throughput × 3.7- VRAM con batch=128 + AMP: 12-15GB (40-50%)

- Throughput esperado: >100 imgs/s

## 📁 Estructura

### Fase 2: Hierarchical (OPCIONAL - si Fase 1 funciona)

```

artefact-poc58-standard/```

├── README.md                    # Este archivoShared Encoder (ConvNeXt-Tiny)

├── configs/    ↓

│   └── unet_convnext_batch128.yaml  # Config principal (DeepLabV3+, batch=64)U-Net Decoder → Fine (16 classes)

├── data/    ├─ Conv1x1 → Binary (2 classes)

│   ├── artefact/                # Dataset original (418 images)    └─ Conv1x1 → Coarse (4 classes)

│   └── artefact_augmented/      # Dataset augmentado (1,464 images)```

├── logs/

│   ├── train_2060.out          # Test exitoso 1 épocaLightweight heads → solo +2-3GB VRAM vs +20GB en UPerNet

│   └── old_tests/              # Tests fallidos archivados

├── scripts/---

│   └── slurm_train.sh          # SLURM job script

└── src/## Stack Tecnológico

    ├── train.py                # Script principal de training

    ├── dataset.py              # Dataset estándar (CPU I/O)### Core Dependencies

    ├── preload_dataset.py      # Dataset con RAM pre-loading ⚡

    └── dali_dataset.py.bak     # DALI fallido (backup)```bash

```# Librería estándar para segmentación

segmentation-models-pytorch==0.3.3

## 🚀 Uso

# Ya instaladas

### Test 1 Épocatorch==2.0.1+cu118

```bashalbumentations

cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc58-standardtimm

sbatch scripts/slurm_train.sh --test-epoch```

```

### Herramientas Clave

### Training Completo (50 Épocas)

```bash1. **SMP (Segmentation Models PyTorch)**

sbatch scripts/slurm_train.sh   - 500+ combinaciones encoder-decoder pre-configuradas

```   - Encoders: ResNet, EfficientNet, ConvNeXt, Swin, etc.

   - Decoders: U-Net, U-Net++, DeepLabV3+, FPN, etc.

### Configuración

2. **PyTorch AMP (Automatic Mixed Precision)**

Editar `configs/unet_convnext_batch128.yaml`:   - FP16 automático en operaciones seguras

   - FP32 en operaciones sensibles (loss, normalización)

```yaml   - 50% menos VRAM, 2-3x más rápido

# Activar/desactivar RAM pre-loading

data:3. **OneCycleLR Scheduler**

  use_preload: true              # RAM pre-loading (97 imgs/s)   - Better que CosineAnnealing para datasets pequeños

  use_augmented: true            # Dataset augmentado (1,464 imgs)   - Learning rate warmup automático

  

training:---

  batch_size: 64                 # Máximo sin OOM

  epochs: 50## Pipeline de Entrenamiento

  mixed_precision: true          # AMP enabled

```### Data



## 📈 Métricas de Entrenamiento```python

# Augmentations mínimas (rápidas)

El script reporta cada época:train_transform = A.Compose([

- **Loss**: DiceLoss train/val    A.Resize(384, 384),

- **mIoU**: Mean IoU across 16 classes    A.HorizontalFlip(p=0.5),

- **Throughput**: imgs/s    A.VerticalFlip(p=0.3),

- **VRAM**: % utilización GPU    A.RandomRotate90(p=0.3),

- **Time**: Tiempo por época    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ToTensorV2()

## 🔍 Troubleshooting])



### OOM (Out of Memory)# DataLoader optimizado

- Reducir `batch_size` de 64 → 32DataLoader(

- Problema: batch=128 requiere ~28.5GB (OOM en V100 32GB)    dataset,

    batch_size=128,      # Grande con AMP

### Bajo throughput    num_workers=8,       # Max CPUs

- Verificar `use_preload: true` en config    pin_memory=True,     # GPU transfer rápido

- Sin pre-loading: ~26 imgs/s (80% I/O wait)    persistent_workers=True,

- Con pre-loading: ~97 imgs/s (I/O eliminado)    prefetch_factor=4    # Prefetch 4 batches

)

### Dataset incompleto```

- Verificar que existan 1,464 images y 1,464 annotations en `artefact_augmented/`

- Si falta alguna: genera con augmentation script### Training Loop con AMP



## 🧪 Experimentos Previos```python

from torch.cuda.amp import autocast, GradScaler

### POC5.5 (Laptop)

- ✅ Funcionó: 22% mIoU, 4 imgs/sscaler = GradScaler()

- ❌ Problema: Código custom, lento

for images, masks in train_loader:

### POC5.6 (Server Port)    # Forward en FP16

- ❌ Falló: 1.8% VRAM, GPU underutilization    with autocast():

        predictions = model(images)

### POC5.7 (Server Native)        loss = criterion(predictions, masks)

- ❌ Falló: OOM batch≥128, 1.9% VRAM batch=64    

    # Backward con gradient scaling

### POC5.8 (Standard Pipeline) ← ACTUAL    scaler.scale(loss).backward()

- ✅ Funcionó: SMP + AMP + RAM pre-loading    scaler.step(optimizer)

- Esperando: mIoU en 50 épocas    scaler.update()

    optimizer.zero_grad()

## 📝 Notas Técnicas```



### ¿Por qué DeepLabV3+ vs U-Net?### Loss Function

- ASPP decoder más eficiente en memoria

- Mejor mIoU (0.0794 vs 0.0494 en 1 época)```python

- 21% menos parámetros (26.7M vs 34.4M)# SMP tiene losses optimizadas

from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

### ¿Por qué ResNet50 vs ResNeSt50?

- Soporta dilated convolutions (mejor para segmentación)# Fase 1: Single task

- Más estándar, mejor documentadocriterion = DiceLoss(mode='multiclass')

- Similar performance, menor complejidad

# Fase 2: Multi-task (si se implementa)

### ¿Por qué RAM pre-loading?criterion_binary = DiceLoss(mode='binary')

- V100 procesamiento: ~0.5s/batchcriterion_coarse = DiceLoss(mode='multiclass')

- CPU I/O (PNG decode): ~1.9s/batchcriterion_fine = DiceLoss(mode='multiclass')

- **Bottleneck identificado**: 80% tiempo en I/O

- **Solución**: Cargar todo en RAM (32GB disponibles)total_loss = 0.2*L_binary + 0.3*L_coarse + 0.5*L_fine

- **Resultado**: I/O → 0s, throughput × 3.7```



### ¿Por qué no DALI?### Optimizer & Scheduler

- Intentado: 10+ iteraciones, todos fallaron

- Problema: ExternalSource decode muy complejo para PNGs custom```python

- Decisión: RAM pre-loading más simple y efectivo# AdamW con weight decay

optimizer = torch.optim.AdamW(

## 🎯 Próximos Pasos    model.parameters(),

    lr=1e-3,          # Higher initial LR (OneCycle bajará)

1. ✅ Test 1 época exitoso (Job 2060)    weight_decay=0.01

2. 🔜 Training 50 épocas)

3. 🔜 Evaluar mIoU final vs POC5.5 (target: ≥22%)

4. 🔜 Si funciona: base para POC6 (ViT integration)# OneCycleLR (mejor que Cosine para pocos datos)

scheduler = torch.optim.lr_scheduler.OneCycleLR(

## 💾 Recursos    optimizer,

    max_lr=1e-3,

- **GPU**: Tesla V100S-PCIE-32GB    total_steps=len(train_loader) * epochs,

- **RAM**: 32-48GB (necesita ~33GB para pre-loading)    pct_start=0.3,    # 30% warmup

- **CPUs**: 8-10 cores    anneal_strategy='cos'

- **Tiempo estimado (50 epochs)**: ~12-15 minutos)

```

---

---

**Autor**: POC5.8 Standard Pipeline  

**Fecha**: Noviembre 2025  ## Métricas Objetivo

**Status**: ✅ Test exitoso, listo para training completo

### GPU Utilization
- ✅ VRAM: **40-60%** (12-18GB de 32GB)
- ✅ Throughput: **>100 imgs/s** (vs POC5.7: 23.9 imgs/s)
- ✅ Time/epoch: **<10s** (vs POC5.7: 17.4s)

### Model Performance
- 🎯 Target mIoU (fine): **>25%** en 50 épocas
  - Baseline POC5.5 laptop: 22% mIoU
  - Con servidor + AMP + mejor arquitectura: debe superar

### Training Time
- 📊 1 época: <10s
- 📊 50 épocas: <10 min
- 📊 Total (con validación): <15 min

**Comparación:**
- POC5.5 laptop: ~4 horas
- POC5.8 servidor: ~15 min
- **Speedup: 16x** 🚀

---

## Estructura del Proyecto

```
artefact-poc58-standard/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencies
├── configs/
│   ├── unet_convnext_batch128.yaml
│   ├── unet_swin_batch128.yaml
│   └── unet_maxvit_batch128.yaml
├── src/
│   ├── dataset.py              # ARTeFACT dataset loader
│   ├── model.py                # SMP model wrapper
│   ├── train.py                # Training script con AMP
│   └── evaluate.py             # Evaluation script
├── scripts/
│   ├── slurm_train.sh          # SLURM script (1 GPU)
│   └── slurm_compare.sh        # Train 3 encoders en paralelo
└── logs/
    └── (training logs)
```

---

## Plan de Ejecución

### Step 1: Setup (5 min)
```bash
cd artefact-poc58-standard
pip install segmentation-models-pytorch==0.3.3
ln -s ../artefact-poc55-multiclass/data data
```

### Step 2: Test 1 Epoch (5 min)
```bash
sbatch scripts/slurm_train.sh --test-epoch
# Validar: VRAM >40%, throughput >100 imgs/s
```

### Step 3: Full Training (15 min)
```bash
sbatch scripts/slurm_train.sh
# 50 épocas ConvNeXt-Tiny
```

### Step 4: Multi-Encoder Comparison (30 min)
```bash
sbatch scripts/slurm_compare.sh
# Train ConvNeXt, Swin, MaxViT en paralelo (3 GPUs)
# Compare mIoU
```

---

## Decisiones de Diseño

### ¿Por qué U-Net y no UPerNet?

| Característica | U-Net | UPerNet (POC5.7) |
|----------------|-------|------------------|
| Parámetros | ~30M | ~38M |
| VRAM @ batch=128 | 12-15GB | 30GB+ (OOM) |
| Throughput | >100 imgs/s | 23.9 imgs/s |
| Complejidad | Baja | Alta |
| Debugging | Mínimo | Días |

**Veredicto:** U-Net es 90% tan bueno con 1/10 del dolor de cabeza.

### ¿Por qué Mixed Precision?

- V100 tiene Tensor Cores optimizados para FP16
- FP16 = 2x menos VRAM, 2-3x más rápido
- Pérdida numérica negligible (<0.1% mIoU)
- **No hay razón para NO usarlo** en 2025

### ¿Por qué SMP y no custom?

- 200k+ usuarios, battle-tested
- Optimizado para V100/A100
- Documentación extensa
- Debugging = issue en GitHub, no días perdidos

---

## Contingencias

### Si batch=128 da OOM (poco probable):
1. Reducir a batch=96
2. Reducir image_size a 320px
3. Usar gradient accumulation

### Si mIoU <20%:
1. Aumentar augmentations (ColorJitter, etc.)
2. Aumentar épocas a 100
3. Probar DeepLabV3+ en vez de U-Net

### Si throughput <100 imgs/s:
1. Verificar num_workers=8
2. Verificar persistent_workers=True
3. Verificar AMP habilitado

---

## Éxito Definido

POC5.8 es **exitoso** si en <2 horas:

1. ✅ 1 época corre sin errores
2. ✅ VRAM >40% y throughput >100 imgs/s
3. ✅ 50 épocas completan en <15 min
4. ✅ mIoU ≥ 22% (POC5.5 baseline)

Si esto falla, **el problema NO es el código**, es el servidor o PyTorch installation.

---

## Próximos Pasos (POC6)

Una vez POC5.8 valida que el servidor funciona:

**POC6 puede agregar innovations SOBRE arquitectura probada:**
- MAE pretraining (sobre U-Net)
- MAML meta-learning (sobre U-Net)
- Domain adaptation (sobre U-Net)
- Attention mechanisms (sobre U-Net)

**No reinventar la rueda de segmentación básica.**

---

## Referencias

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch AMP Tutorial](https://pytorch.org/docs/stable/amp.html)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [OneCycleLR Paper](https://arxiv.org/abs/1708.07120)

---

**Filosofía final:** "Make it work, make it right, make it fast" - en ese orden. POC5.7 trató de hacer todo a la vez. POC5.8 hace una cosa bien.
