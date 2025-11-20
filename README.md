# HeritageArt-CNN-ViT-Hybrid

## Resumen del proyecto
HeritageArt-CNN-ViT-Hybrid reúne todos los pipelines, scripts y reportes usados para segmentar automáticamente daños en artefactos patrimoniales usando el dataset **ARTeFACT**. El repositorio consolida tres pruebas de concepto (POC-5.5, POC-5.8 y POC-5.9) que comparan arquitecturas CNN, Vision Transformer e híbridas, documentan la evolución del desempeño y preparan el lanzamiento de POC-6 orientado a domain generalization.

Objetivo principal: obtener un flujo reproducible de preparación de datos → entrenamiento → evaluación → visualización que produzca modelos desplegables (SegFormer-B3 alcanzó **37.63 % mIoU** en POC-5.9-v2) y deje trazabilidad completa para futuras iteraciones.

---

## Estructura del repositorio
| Ruta | Propósito |
| --- | --- |
| `pipeline/` | Código planificado del pipeline genérico (módulos en `src/`, configuraciones en `configs/`, pruebas en `tests/`). |
| `experiments/common-data/` | Descarga y augmentación del dataset ARTeFACT. Incluye Makefile, scripts Docker y el dataset aumentado (`artefact_augmented/`). |
| `experiments/artefact-poc55-multiclass/` | POC-5.5: entrenamiento jerárquico multinivel (Docker o SLURM). Configuraciones y scripts dedicados. |
| `experiments/artefact-poc58-standard/` | POC-5.8: benchmark estándar con Segmentation Models PyTorch (ConvNeXt, Swin, CoAtNet). |
| `experiments/artefact-poc59-multiarch-benchmark/` | POC-5.9-v2: benchmark de producción (ConvNeXt-Tiny, MaxViT-Tiny, SegFormer-B3) con scripts SLURM, evaluaciones y visualizaciones completas. |
| `presentation_data/` | CSV y scripts para gráficos (tablas 1–6). Sustentan resultados y proyecciones POC-6. |
| `documentation/` | Reportes técnicos: historia completa (`POC_Full_History_Analysis.md`), comparativas (`POC_Evolution_Comparison.md`), estado (`PROJECT_STATUS.md`), reporte de producción (`POC59_PRODUCTION_REPORT.md`), planes POC-6. |

---

## Modelos implementados

### POC-5.5 – Hierarchical UPerNet
- **Descripción:** Entrenamiento curricular con tres cabezas (binaria, coarse de 4 grupos y fine de 16 clases) sobre UPerNet para permitir aprendizaje estable en una GPU de 6 GB.
- **Archivos clave:**
  - `experiments/artefact-poc55-multiclass/scripts/train_poc55.py` y `scripts/models/hierarchical_upernet.py`.
  - Configuraciones en `experiments/artefact-poc55-multiclass/configs/*.yaml`.
- **Hiperparámetros:** imágenes 256×256, batch 4, AdamW, pérdidas ponderadas por nivel (Dice + CrossEntropy). Uso intensivo de augmentaciones y warmup largo.
- **Notas:** único modelo multitarea; ideal para iterar en laptop via Docker (`make train-convnext`, `make train-swin`, `make train-maxvit`).

### POC-5.8 – DeepLabV3+ (SMP)
- **Descripción:** Benchmark server-optimized usando Segmentation Models PyTorch con backbones ConvNeXt-Tiny, Swin-Tiny y CoAtNet-0 para medir throughput y comparabilidad.
- **Archivos clave:**
  - `experiments/artefact-poc58-standard/src/train.py`, `src/dataset.py`, `src/evaluate.py`.
  - Configuraciones en `experiments/artefact-poc58-standard/configs/*.yaml`.
- **Hiperparámetros:** batch 32–96 (según encoder), Mixed Precision (AMP), OneCycleLR, preloading opcional (`preload_dataset.py`).
- **Notas:** scripts SLURM (`scripts/slurm_train.sh`, `scripts/train_all_parallel.sh`) ejecutan múltiples modelos en paralelo para aprovechar 2 GPUs.

### POC-5.9 – Multi-Architecture Benchmark (Producción)
- **Modelos:** ConvNeXt-Tiny (CNN), MaxViT-Tiny (Hybrid) y SegFormer-B3 (Vision Transformer).
- **Archivos clave:**
  - `experiments/artefact-poc59-multiarch-benchmark/src/train.py`, `src/evaluate.py`, `src/visualize.py`.
  - Fábrica de modelos en `src/model_factory.py` y adaptadores TIMM en `src/timm_encoder.py`.
  - Configs YAML en `experiments/artefact-poc59-multiarch-benchmark/configs/`.
  - Scripts SLURM/monitorización en `experiments/artefact-poc59-multiarch-benchmark/scripts/` (con README propio).
- **Hiperparámetros:** resolución 384×384, 50 épocas, batch adaptativo (32–96), `use_preload: true`, AMP, pérdidas Dice + CrossEntropy con pesos de clase (`experiments/common-data/artefact_augmented/class_weights_balanced.json`).
- **Notas:** única POC con pipeline Train → Evaluate → Visualize automatizado, almacenamiento organizado en `logs/models/<encoder>/` y reporte completo en `documentation/POC59_PRODUCTION_REPORT.md`.

---

## Dataset
- **Origen:** Hugging Face `danielaivanova/damaged-media` (ARTeFACT). Documentado en `experiments/common-data/README.md`.
- **Contenido:** imágenes RGB de artefactos y máscaras de 16 clases (0–15 + 255 ignore). 417 muestras originales (~2–3 GB) + 1,458 aumentadas (~9.7 GB) ya versionadas en `experiments/common-data/artefact_augmented/`.
- **Estructura base:**
  - `images/`, `annotations/`, `annotations_rgb/`, `visualizations/`.
  - `metadata.csv`, `statistics.json`, `class_weights_balanced.json`.
- **Preparación completa (opcional, si se desea recrear datos):**
  ```bash
  cd experiments/common-data/artefact-data-obtention
  make build                     # Construye imagen Docker
  make download-full             # Genera ../artefact/ con 417 muestras
  make generate-augmentations    # Crea ../artefact_augmented/ con 1,458 ejemplos
  ```
- **Uso en POCs:** todas las configuraciones apuntan a `experiments/common-data/artefact_augmented/`. No es necesario volver a descargar a menos que se quiera regenerar.

---

## Pipeline general del proyecto
1. **Obtención y augmentación:** scripts en `experiments/common-data/artefact-data-obtention/scripts/` descargan ARTeFACT y generan aumentos equilibrados.
2. **Preprocesamiento:** `preload_dataset.py` (POC-5.8) y la lógica de `ArtefactDataset` (POC-5.9) normalizan imágenes, aplican transforms Albumentations y construyen pesos de clase.
3. **Entrenamiento:** se lanza la POC deseada (`train_poc55.py`, `src/train.py` de POC-5.8 o `src/train.py` de POC-5.9) usando configuraciones YAML específicas.
4. **Evaluación:** scripts (`scripts/evaluate.py`, `src/evaluate.py`) generan métricas (IoU, precisión, recall, F1) y guardan resultados en `logs/models/<encoder>/evaluation/`.
5. **Visualización y reporting:** `src/visualize.py` (POC-5.9) produce grids, distribuciones y análisis per-class; los CSV/plots finales viven en `presentation_data/`.
6. **Documentación:** cada ciclo se registra en `documentation/` (reportes, planes, comparativas), finalizando con `POC59_PRODUCTION_REPORT.md` y `PROJECT_STATUS.md`.

Diagrama textual del pipeline:
```
Datos → Augmentations → Dataloaders → Entrenamiento (por modelo) → Checkpoints → Evaluaciones → Visualizaciones → Reportes/Presentaciones
```

---

## Requisitos y dependencias
- **Lenguaje:** Python ≥ 3.10 (servidor validado con 3.10.13).
- **Frameworks principales:** PyTorch, torchvision, timm, segmentation-models-pytorch, albumentations, Hydra, PyYAML, pandas, seaborn, matplotlib, tqdm.
- **Componentes OpenMMLab (para demos):** mmengine 0.10.4, mmcv 2.1.0, mmsegmentation 1.2.2 (instalación vía `openmim`).
- **Archivos de dependencias relevantes:**
  - `pipeline/requirements.txt` – dependencias del pipeline general.
  - `experiments/artefact-poc55-multiclass/docker/requirements.txt` – entorno Docker.
  - `experiments/artefact-poc58-standard/requirements.txt` y `experiments/artefact-poc59-multiarch-benchmark/requirements.txt` – listas específicas de cada POC.
- **Instalación típica (POC-5.9):**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate        # PowerShell
  python -m pip install -U pip
  pip install -r experiments/artefact-poc59-multiarch-benchmark/requirements.txt
  ```
- **Verificación GPU:** `python experiments/utils/cuda-test.py` (confirma versión de torch y dispositivos).

---

## Cómo ejecutar el proyecto

### 1. Validar entorno
```bash
python experiments/utils/cuda-test.py
```

### 2. POC-5.5 (Docker o SLURM autodetectado)
```bash
cd experiments/artefact-poc55-multiclass
make env              # confirma entorno
make train-convnext   # u otras dianas: train-swin, train-maxvit, train-all
make evaluate         # produce métricas jerárquicas
```

### 3. POC-5.8 (cluster SLURM)
```bash
cd experiments/artefact-poc58-standard
sbatch scripts/slurm_train.sh configs/convnext_tiny.yaml
sbatch scripts/slurm_train.sh configs/swin_tiny.yaml
sbatch scripts/slurm_train.sh configs/coatnet_0.yaml
bash scripts/evaluate_all.sh
```
También existen atajos como `scripts/train_all_parallel.sh` para lanzar los tres modelos en dos GPUs.

### 4. POC-5.9 (pipeline producción)
```bash
cd experiments/artefact-poc59-multiarch-benchmark
sbatch scripts/slurm_train.sh configs/segformer_b3.yaml      # entrenamiento 50 épocas
sbatch scripts/slurm_evaluate.sh                             # evalúa todos los checkpoints
sbatch scripts/slurm_visualize.sh                            # genera 27 PNG por modelo
```

### 5. Inferencia rápida con un checkpoint existente
```bash
python src/evaluate.py \
  --config configs/segformer_b3.yaml \
  --checkpoint logs/models/segformer_b3/checkpoint/best_model.pth \
  --input path/a/imagen.jpg \
  --output path/a/prediccion.png
```

---

## Resultados y métricas
| Modelo | Familia | mIoU | Dice | Accuracy | Comentarios |
| --- | --- | --- | --- | --- | --- |
| SegFormer-B3 | Vision Transformer | **37.63 %** | 46.29 % | 88.15 % | Ganador POC-5.9, 81 img/s, checkpoint 543 MB. |
| MaxViT-Tiny | Hybrid CNN+ViT | 34.58 % | 43.89 % | 87.82 % | Mejor compromiso generalización/tiempo. |
| ConvNeXt-Tiny | CNN | 25.47 % | 35.24 % | 86.71 % | Baseline rápido (122 img/s). |

Resumen adicional:
- Top-3 clases por IoU en POC-5.9: Clean (95 %), Material Loss (81 %), Peel Loss (66 %).
- Visualizaciones: 9 PNG por modelo (`logs/models/<encoder>/visualizations/`).
- Reportes completos en `documentation/POC59_PRODUCTION_REPORT.md` y evolución histórica en `documentation/POC_Evolution_Comparison.md`.

---

## Ejemplos de uso

### Cargar dataset preprocesado (POC-5.9)
```python
from experiments.artefact_poc59_multiarch_benchmark.src.dataset import ArtefactDataset, get_transforms

dataset = ArtefactDataset(
    root="../common-data/artefact_augmented",
    split="val",
    transforms=get_transforms("val", image_size=384)
)
image, mask = dataset[0]
print(image.shape, mask.shape)
```

### Crear un modelo desde la fábrica
```python
import yaml
from experiments.artefact_poc59_multiarch_benchmark.src.model_factory import create_model

config = yaml.safe_load(open("configs/segformer_b3.yaml", mode="r", encoding="utf-8"))
model = create_model(config["model"])
model.eval()
```

### Visualizar métricas guardadas
```python
import json

with open("logs/models/segformer_b3/evaluation/metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

print("mIoU:", metrics["miou"])           # 0.3763
print("Per-class IoU keys:", metrics["per_class_iou"].keys())
```

---

## Limitaciones y trabajo futuro
- **Dataset incompleto:** solo 417 muestras reales frente a las 11k anunciadas. Las clases raras (scratches, structural defects, dirt) siguen con IoU < 1 %.
- **Generalización pendiente:** RQ2 (domain generalization) se documenta en `documentation/POC6_EXECUTION_PLAN.md`; planea usar splits LOMO/LOContent y técnicas Fishr, Deep CORAL, IRM y TENT.
- **Despliegue POC-6:** integrar UPerNet jerárquico en pipeline producción, aumentar resolución uniforme y explorar data-centric fixes.

---

## Licencia y créditos
- **Licencia:** el repositorio no incluye archivo de licencia; confirma requisitos antes de redistribuir modelos o datos.
- **Créditos:** dataset ARTeFACT cortesía de ICOMUSEF / danielaivanova. Código basado en PyTorch, timm, Segmentation Models PyTorch y herramientas OpenMMLab. Documentación elaborada a partir de `documentation/From-Paper-to-Plan.md`, `POC_Full_History_Analysis.md` y reportes asociados.

---
