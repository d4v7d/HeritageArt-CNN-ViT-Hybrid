# Detection of Conservation-Relevant Features in Cultural Heritage Artworks Using Vision Transformers and CNNs

## Abstract
Preserving the integrity and cultural authenticity of artworks is a fundamental challenge in heritage conservation. Traditionally, conservators rely on visual inspection and scientific imaging techniques such as X-ray, infrared photography, and microscopy. Despite significant advances in computer vision, its application to cultural heritage remains scarce and predominantly focused on digital restoration rather than preventive conservation.

Building on the ARTeFACT benchmark, we adopt its multiclass damage taxonomy and evaluation protocol to compare three architecture families for semantic damage segmentation: Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and Hybrid CNN-ViTs.

Our study aims to reproduce and validate strong baselines reported on ARTeFACT under a unified implementation. We report region-level mIoU, Dice/F1, and per-class IoU, and analyze error modes to characterize when each family is preferable (e.g., high-frequency versus context-dependent degradations).

The results establish reproducible baselines and illuminate the comparative behavior of CNNs, Transformers, and hybrids for artwork damage segmentation. Our experiments demonstrate that the Hybrid architecture achieves robust performance, effectively combining local inductive biases with global attention to excel on both broad material losses and complex structural defects.

## Repository Structure

This repository contains the complete implementation of the segmentation pipeline, including data preparation, model training, evaluation, and visualization tools.

| Path | Description |
| --- | --- |
| `pipeline/` | Core production pipeline containing source code, configurations, and scripts (based on POC-5.9). |
| `experiments/` | Historical experiments and proofs of concept (POC-5.5, POC-5.8, POC-5.9). |
| `documentation/` | Technical reports, LaTeX sources, and project documentation. |

## Methodology

### Dataset
The project utilizes the **ARTeFACT** dataset, which includes high-resolution images of artworks annotated with 15 pixel-level damage classes plus a clean label. The dataset is stratified by material and content type to ensure diverse representation.

### Architectures
We evaluate three distinct architectures to represent different inductive biases. All models utilize a U-Net decoder for consistent segmentation output.

1.  **CNN (ConvNeXt-Tiny):** A modernized CNN architecture that incorporates transformer-inspired design elements while maintaining the efficiency of convolutions.
2.  **Vision Transformer (SegFormer-B3):** A hierarchical transformer designed for semantic segmentation, capable of modeling global context and long-range dependencies.
3.  **Hybrid (MaxViT-Tiny):** A multi-axis vision transformer that combines local convolution blocks with global attention mechanisms.

### Training Configuration
*   **Resolution:** 384x384 pixels
*   **Optimizer:** AdamW
*   **Loss Function:** Dice Loss with class weights to handle imbalance.
*   **Augmentation:** Geometric (flips, rotations, elastic deformation) and photometric (brightness, contrast) transformations.

## Getting Started

### Prerequisites
The project requires Python 3.10+ and PyTorch. The dependencies are listed in `pipeline/requirements.txt`.

```bash
pip install -r pipeline/requirements.txt
```

### Usage

The pipeline supports training, evaluation, and visualization modes. Configuration is managed via YAML files located in `pipeline/configs/`.

#### Training

To train a model (e.g., ConvNeXt-Tiny):

```bash
python pipeline/src/train.py --config pipeline/configs/convnext_tiny.yaml
```

#### Evaluation

To evaluate a trained model and generate metrics:

```bash
python pipeline/src/evaluate.py --config pipeline/configs/convnext_tiny.yaml
```

#### Visualization

To generate visual predictions:

```bash
python pipeline/src/visualize.py --config pipeline/configs/convnext_tiny.yaml
```

## Authors
*   Brandon Trigueros-Lara
*   Valentino Vidaurre-Rodríguez
*   David González-Villanueva
*   Rubén González-Villanueva
*   Christian Quesada-López
*   Jeisson Hidalgo-Céspedes

*Universidad de Costa Rica*
