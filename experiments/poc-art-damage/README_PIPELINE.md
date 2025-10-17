# ARTeFACT End-to-End Pipeline

Este pipeline completo te permite entrenar y evaluar tres modelos (CNN, ViT, Híbrido) en el dataset ARTeFACT de forma automatizada.

## 🎯 Lo que hace el pipeline

1. **Descarga ARTeFACT**: Descarga el dataset una sola vez y lo guarda localmente
2. **Entrena 3 modelos**:
   - **CNN**: ConvNeXt-Tiny + FPN
   - **ViT**: Swin-Base + UPerNet
   - **Híbrido**: MaxViT-Tiny + FPN
3. **Evalúa cada modelo**: Calcula métricas F1 y mIoU
4. **Genera visualizaciones**: Por cada imagen evaluada, crea:
   - Máscara de segmentación predicha
   - Ground truth
   - Overlay sobre la imagen original
   - Visualización comparativa (4 paneles)
   - Archivo JSON con métricas F1 y mIoU

## 📁 Estructura de salida

```
logs/
├── data/
│   └── artefact_real/          # Dataset descargado (se mantiene entre ejecuciones)
│       ├── image/
│       ├── annotation/
│       └── metadata.csv
└── pipeline_results/
    ├── checkpoints/             # Modelos entrenados
    │   ├── convnext_tiny_fpn_artefact.pth
    │   ├── upernet_swin_base_artefact.pth
    │   └── maxvit_tiny_fpn_artefact.pth
    ├── convnext_tiny_fpn/      # Resultados del modelo CNN
    │   ├── {image_id}_visualization.png  # 4 paneles comparativos
    │   ├── {image_id}_pred.png          # Predicción coloreada
    │   ├── {image_id}_gt.png            # Ground truth coloreado
    │   ├── {image_id}_overlay.png       # Overlay
    │   ├── {image_id}_metrics.json      # Métricas por imagen
    │   └── overall_metrics.json         # Métricas generales del modelo
    ├── upernet_swin_base/      # Resultados del modelo ViT
    │   └── ... (mismo formato)
    ├── maxvit_tiny_fpn/        # Resultados del modelo Híbrido
    │   └── ... (mismo formato)
    └── summary_results.json    # Comparación de los 3 modelos
```

## 🚀 Uso

### Opción 1: Pipeline rápido (recomendado para pruebas)
```bash
make build              # Solo la primera vez
make pipeline-quick     # 30 muestras, 10 epochs, 10 imágenes de evaluación
```

### Opción 2: Pipeline personalizado
```bash
make build  # Solo la primera vez
make pipeline MAX_SAMPLES=100 EPOCHS=20 MAX_EVAL=20
```

Parámetros:
- `MAX_SAMPLES`: Cantidad de muestras a descargar de ARTeFACT (None = todas, ~400)
- `EPOCHS`: Número de epochs para entrenar cada modelo
- `MAX_EVAL`: Máximo de imágenes para evaluar y visualizar

### Opción 3: Solo evaluación (usa checkpoints existentes)
```bash
make pipeline-eval-only  # Salta el entrenamiento, usa modelos ya entrenados
```

## ⏱️ Tiempos estimados

Con GPU NVIDIA (ejemplo RTX 3090):
- **Pipeline rápido**: ~30-45 minutos
  - Descarga: ~5 min (30 muestras)
  - Entrenamiento: ~5-8 min por modelo (10 epochs)
  - Evaluación: ~2 min por modelo (10 imágenes)

- **Pipeline completo** (100 muestras, 20 epochs):
  - Descarga: ~15-20 min
  - Entrenamiento: ~10-15 min por modelo
  - Evaluación: ~4 min por modelo (20 imágenes)

## 📊 Métricas generadas

Para cada imagen evaluada (`{image_id}_metrics.json`):
```json
{
  "image_id": "0001",
  "mIoU": 0.6234,
  "mF1": 0.7123,
  "per_class_iou": [0.89, 0.45, ...],
  "per_class_f1": [0.94, 0.62, ...]
}
```

Resumen general (`overall_metrics.json`):
```json
{
  "model_name": "convnext_tiny_fpn",
  "mean_mIoU": 0.6234,
  "std_mIoU": 0.0523,
  "mean_mF1": 0.7123,
  "std_mF1": 0.0412,
  "num_images": 20
}
```

Comparación de modelos (`summary_results.json`):
- Métricas de los 3 modelos para comparación directa

## 🔍 Visualización de resultados

Cada imagen genera 4 archivos:
1. **`{id}_visualization.png`**: Panel de 2x2 con original, GT, predicción y overlay
2. **`{id}_pred.png`**: Máscara de segmentación predicha (coloreada)
3. **`{id}_gt.png`**: Ground truth (coloreada)
4. **`{id}_overlay.png`**: Predicción superpuesta sobre la imagen original

## 🔄 Ejecuciones subsecuentes

El dataset ARTeFACT se descarga **una sola vez** y se guarda en `logs/data/artefact_real/`. En ejecuciones posteriores:
- Si el directorio existe, **no se descarga de nuevo**
- Solo se re-entrena y evalúa según los parámetros especificados

Para forzar una nueva descarga:
```bash
rm -rf logs/data/artefact_real
make pipeline-quick
```

## 🎨 Clases de daño (16 clases)

0. Clean (limpio)
1. Material loss (pérdida de material)
2. Peel (desprendimiento)
3. Dust (polvo)
4. Scratch (rayadura)
5. Hair (pelo)
6. Dirt (suciedad)
7. Fold (pliegue)
8. Writing (escritura)
9. Cracks (grietas)
10. Staining (manchas)
11. Stamp (sello)
12. Sticker (adhesivo)
13. Puncture (perforación)
14. Burn marks (marcas de quemadura)
15. Lightleak (fuga de luz)

## 🐛 Troubleshooting

### Error de memoria compartida (shared memory)
Si ves `Bus error` o `shared memory` error:
- El pipeline ya usa `num_workers=0` para evitar esto
- Si persiste, reduce `--batch-size 2` a `--batch-size 1`

### Descarga muy lenta
- Usa `MAX_SAMPLES=30` o `MAX_SAMPLES=50` para pruebas rápidas
- El dataset completo (~400 imágenes) puede tardar 20-30 min

### GPU sin memoria
- Reduce batch size en el código o usa menos samples
- El pipeline está configurado para batch_size=2 que funciona en GPUs de 8GB

## 📝 Notas

- Los modelos vienen con **pesos pre-entrenados** de ImageNet/similares
- El **fine-tuning** ajusta estos pesos al dataset ARTeFACT
- Los checkpoints se guardan y pueden reutilizarse con `pipeline-eval-only`
