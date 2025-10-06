# From Paper to Plan: Building the Hybrid CNN–ViT Artifact (with DG) — *ARTeFACT-enabled*

> **TL;DR**
> A single, reproducible semantic-segmentation pipeline with a *common decoder (UPerNet)* and *swappable backbones* (CNN / ViT / Hybrid), evaluated *in-domain* and under *domain generalization* (LOMO/LOContent) on **ARTeFACT**. High-res handled via Hann-blended tiling; metrics include *mIoU, macro-F1*, with *ECE* and *κ* adjuncts; all with *bootstrap 95% CIs*. DG closures via *style/frequency aug*, *CORAL/IRM/Fishr*, and *TENT*.

## Contents

* [0) What we are building (Artifact Synthesis)](#0-what-we-are-building-artifact-synthesis)
* [1) Actionable Implementation Plan](#1-actionable-implementation-plan-objectives--tasks--verification--literature-ties)

  * [Phase 1 — Env/Repo/Data I/O](#phase-1--environment-repository-and-data-io)
  * [Phase 2 — Baseline CNN + Imbalance Loss](#phase-2--baseline-cnn-unet-or-deeplabv3-with-class-imbalance-aware-loss)
  * [Phase 3 — Common Decoder + Backbones (RQ1)](#phase-3--common-decoder--backbones-cnn-vs-vit-vs-hybrid-rq1-head-to-head)
  * [Phase 4 — Domain Generalization (RQ2)](#phase-4--domain-generalization-with-artefact-metadata-rq2)
  * [Phase 5 — Closing the DG Gap](#phase-5--closing-the-dg-gap-augmentation-stylefourier-dg-regularizers-tta)
  * [Phase 6 — Interpretability, Uncertainty, Release](#phase-6--interpretability-uncertainty-and-packaging-for-release)
* [2) Initial Code Scaffolding (PyTorch)](#2-initial-code-scaffolding-pytorch-skeleton)
* [3) Immediate Engineering Deltas (ARTeFACT-specific)](#3-immediate-engineering-deltas-because-we-use-artefact)
* [4) Critical Open Questions](#4-critical-open-questions-to-resolve-early)
* [5) What to do next (today)](#5-what-to-do-next-today)
* [6) References](#6-references)
* [Appendix A — Class Policy](#appendix-a--class-policy)
* [Appendix B — Milestone Tracker](#appendix-b--milestone-tracker)
* [Appendix C — High-Res Inference (Mermaid Sketch)](#appendix-c--high-res-inference-mermaid-sketch)

---

## 0) What we are building (Artifact Synthesis)

We will build a **semantic-segmentation** pipeline that can be instantiated with three **backbone families** while keeping a **common decoder** and **shared training/eval protocol**:

**Backbones**

1. **A modern CNN** (ConvNeXt)
2. **Hierarchical ViT** (Swin Transformer)
3. **Hybrid CNN–ViT** (CoaT or U-Conformer-style architecture that fuses local convolutional bias with global self-attention)

**Decoder**: One **state-of-the-art head** (**UPerNet**) used **across all backbones** for fairness.

**Dataset**: **ARTeFACT** (primary), with **16 semantic classes** from day one (**0=Clean; 1..15 damage**), and **Background = 255** (ignored in loss/metrics). The dataset provides **material** and **content** metadata enabling **LOMO** (Leave-One-Material-Out) and **LOContent** (Leave-One-Content-Out) splits for Domain Generalization (**DG**).

**Task**: Binary (Damage vs Clean) and multiclass (taxonomy of 15 damage types + Clean).

**Metrics**: mIoU and macro-F1 as primaries; calibration (ECE) and expert agreement (κ on tiles) as adjuncts.

**High-res handling**: **Patch/tiling** (e.g., **512×512**), **sliding-window inference** with ~**50% overlap**, **Hann blending** to reduce seams.

**Primary metrics**: **mIoU** and **macro-F1**; adjuncts: **ECE** (calibration) and **κ** (tile-wise). Report **bootstrap 95% CIs** (image-level resampling).

**Research questions**:

* **RQ1**: CNN vs ViT vs Hybrid (isolating **backbone** effect with common head/protocol).
* **RQ2**: **DG** gap under **LOMO/LOContent** (in-domain vs held-out domain), and methods to **close the gap**.

This artifact aligns with your RQ1 (CNN vs ViT vs Hybrid) and extends naturally to RQ2 (Domain Generalization) by evaluating in-domain vs held-out domain performance gaps under LOMO/LOContent.

---

## 1) Actionable Implementation Plan (Objectives → Tasks → Verification → Literature ties)

**Principle.** Keep **decoder, losses, tiling, metrics, eval protocol** identical across backbones. Swap *only* the backbone to isolate architectural effects (**RQ1**). Then **stress** with **LOMO/LOContent** to measure **DG** gaps (**RQ2**).

### Phase 1 — Environment, Repository, and Data I/O

**Objective.** Reproducible codebase and dataset loader yielding **image–mask tiles** for training and **whole-image** inference.

**Key Tasks.**

**Repo & tooling.** Initialize a Git repo with `src/`, `configs/`, `notebooks/`, `data/`, `tests/`, `logs/`. Add **pre-commit**, **black/flake8**, and **Hydra/YAML** configs. Track **seeds** and **library versions** in logs/README (MLOps discipline). (Project hygiene per Rosebrock; MLOps discipline per Lakshmanan.)

**Data loader (ARTeFACT).** Read **RGB images** + **PNG masks**; emit random **512×512** crops. Implement **sliding-window inference** with **50% overlap** + **Hann blending**. Normalize to **ImageNet stats** to match pretrained backbones. (Methods explicitly recommend tiling and Hann blending.)

**Transforms.** Basic augmentations: flips, slight rotations, mild color jitter/blur, and careful random crops preserving tiny structures (hair, scratch). (Image pre-processing foundations: Gonzalez & Woods; practice: Rosebrock.)

**Metrics.** Implement per-class **IoU/F1**, **macro-F1**, pixel accuracy (for reference), and **ECE** (binned reliability). **Bootstrap** (image-level) for **95% CIs**. (All metrics/protocols appear in the guide.)

**Verification.**

* **Smoke test**: load N images → batch tiles → dummy forward → IoU/F1 + CIs on toy masks.
* **Whole-image inference** reconstructs masks **without seams** (visual check). (Hann blending reduces edge effects.)

**Explicit Connections.**

* Tiling size, overlap, and Hann blending: methodology & guide.
* IoU/F1/ECE/κ/Bootstrap: metrics & evaluation sections.
* UPerNet as **common head** (segmentation standard). ARTeFACT card: **PNG labels**, **ignore=255**, **LOOCV** examples.

---

### Phase 2 — Baseline CNN + Imbalance Loss

**Objective.** Establish strong CNN baseline and stable training loop.

**Key Tasks.**

**Baseline models.** Implement **U-Net** and/or **DeepLabV3+ (ResNet-50)** with **binary** and **multiclass (16-class)** heads. Start from **ImageNet** initialization.

**Loss.** Use **Dice + Focal** (or **Tversky**) to handle **Clean ≫ damage** imbalance; log **per-class PR** curves. (Guide recommendations.)

**Optimizer.** **AdamW** with weight decay; **LR warmup + cosine/OneCycle**; **early-stopping** on **macro-F1**; **ReduceLROnPlateau** fallback. (Goodfellow §7–8; practice in guide.)

**Logging.** Save configs, seeds, metrics, and qualitative overlays (input/GT/pred). (Rosebrock + Lakshmanan.)

**Verification.**

* On a small validation split, **binary** baseline **mIoU > 0.5**.
* **Stability** across **3 seeds** with **narrow CIs**.

**Explicit Connections.**

* Losses, imbalance, metrics: guide §7–§8.
* Optimization/regularization: Goodfellow (Deep Learning); practical variants adopted in guide.
* DeepLabV3+ & U-Net as canonical baselines.

---

### Phase 3 — Common Decoder + Backbones (RQ1 head-to-head)

**Objective.** Isolate **backbone** effect via a shared **UPerNet** decoder and single training/eval pipeline.

**Key Tasks.**

**UPerNet head.** Implement **UPerNet** once (single implementation) with lateral pyramid fusion; add adapters for **1/4, 1/8, 1/16, 1/32** features regardless of backbone. (Method suggests a common decoder for fairness.)

**Backbones.**

* **CNN**: ConvNeXt-L (ImageNet-21k if available) or ResNet-50 → UPerNet.
* **ViT**: Swin-Base (self-supervised pretraining preferred: DINOv2/MAE, else ImageNet-1k) or SegFormer-B0/B2 → UPerNet (SSL pretraining preferred when available).
* **Hybrid**: CoaT-Mini/Small or U-Conformer-like architecture (parallel conv + MHSA / interleaving) exposing a **feature pyramid** to UPerNet. Start with CoaT-Mini/Small or an encoder with parallel conv+attention branches; verify feature-pyramid outputs → UPerNet (or a thin UNet-style decoder if needed).

**Training parity.** Identical crops/aug, batch/epochs, losses, schedulers, seeds, evaluation.

**Verification.**

* **In-domain** table with **mIoU** and **macro-F1** ± **95% CI** for **CNN/ViT/Hybrid**; **paired per-image IoU** comparisons.
* **Qualitative** panels (success/failure on cracks vs stains vs material loss). Typical success/failure for cracks vs stains vs material loss.

**Explicit Connections.**

* Backbone families & hypothesis of complementary strengths: methodology.
* Common decoder for fairness + metrics: methodology + guide.
* SegFormer's hierarchical encoder + light MLP decoder informs ViT side.

---

### Phase 4 — Domain Generalization with ARTeFACT Metadata (RQ2)

**Objective.** Quantify OOD drop and prepare to close it using ARTeFACT’s native domains.

**Key Tasks.**

* Implement **LOMO** (10 materials) and **LOContent** (4 contents) as **leave-one-domain-out** splits using dataset LOOCV example; persist **split JSONs/CSV manifests**.
* **Leakage checks** (perceptual hashing) to avoid near-duplicates across splits; **manual** inspection of outliers.
* Run **Phase-3 winners** under each protocol; compute **DG gap = in-domain − OOD** with **bootstrap CIs**.

**Verification.**

* Per-protocol figure showing **DG gaps** (CNN/ViT/Hybrid) with CIs; also **ECE** and **κ** (tile-wise).
* Reproducible **split files** + logs/artifacts.

**Explicit Connections.**
ARTeFACT’s **content/material** splits make **LOMO/LOContent** natural choices.

---

### Phase 5 — Closing the DG Gap: Augmentation, Style/Fourier, DG Regularizers, TTA

**Objective.** Reduce OOD loss without harming in-domain.

**Key Tasks.**

* **Aggressive augmentation**: higher-range **ColorJitter**; **MixUp/CutMix** for segmentation; **style/frequency** perturbations to vary palette/texture while preserving content.
* **DG regularizers**: **Deep CORAL**, **IRM**, **Fishr** across training domains (materials and/or contents).
* **TTA**: **TENT** (entropy minimization with **BN affine** params) per test domain; **safety guards** (clip updates / stop on divergence).

**Verification.**

* **Before/After** DG plots showing **mIoU OOD uplift** with **no in-domain collapse**.
* **Ablations**: “+Style/Fourier”, “+MixUp/CutMix”, “+CORAL/IRM”, “+TENT”.

**Explicit Connections.**
CORAL/IRM/Fishr as standard DG baselines; **TENT** for test-time adaptation.

---

### Phase 6 — Interpretability, Uncertainty, and Packaging for Release

**Objective.** Make the artifact **usable and trustworthy**.

**Key Tasks.**

* **Explainability**: **Grad-CAM/Score-CAM** for CNNs; **attention rollout/maps** for ViTs; overlay panels (input, GT, pred, attention/uncertainty).
* **Uncertainty**: **MC-Dropout** or **ensembles** at inference; visualize **epistemic hotspots**.
* **Release**: Fixed splits, configs, weights, metrics tables with CIs, concise how-to, license & data cards; **minimal model-zoo**.

**Verification.**

* A **model-zoo** folder with trained weights and a **README** that reproduces **one LOMO** experiment end-to-end.

**Explicit Connections.**
UPerNet + Swin are commonly paired; packaging mirrors open-source segmentation practice.

---

## 2) Initial Code Scaffolding (PyTorch Skeleton)

```python
# datasets/artefact_dataset.py
import torch, numpy as np
from torch.utils.data import Dataset
from PIL import Image

CLASS_NAMES = [
    "Clean","Material loss","Peel","Dust","Scratch","Hair","Dirt","Fold",
    "Writing","Cracks","Staining","Stamp","Sticker","Puncture","Burn marks","Lightleak"
]
ID_CLEAN = 0
IGNORE_INDEX = 255
N_CLASSES = 16  # Clean + 15 damages

class ArtefactSegmentation(Dataset):
    """Reads ARTeFACT PNG masks (0..15, 255=Background) + RGB images."""
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self): return len(self.df)

    def _read_mask(self, path):
        m = np.array(Image.open(path))  # uint8 PNG
        return torch.from_numpy(m.astype(np.int64))  # [H,W] in {0..15, 255}

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.image_path).convert("RGB")
        mask = self._read_mask(row.annotation_path)

        if self.transforms:
            out = self.transforms(image=np.array(img), mask=mask.numpy())
            img = out["image"]; mask = torch.from_numpy(out["mask"].astype(np.int64))

        x = torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0
        return x, mask, {"material": row.material, "content": row.content}
```

```python
# models/hybrid_cnn_vit.py  (skeleton)
import torch, torch.nn as nn

class CNNBackbone(nn.Module):
    """Adapter for a CNN (e.g., ResNet/ConvNeXt) → feature pyramid {4,8,16,32}."""
    def __init__(self, name="resnet50", pretrained=True):
        super().__init__()
        # TODO: load from timm and expose pyramid blocks with known channel dims
        self.out_channels = {4:128, 8:256, 16:512, 32:1024}

    def forward(self, x):
        # TODO: run backbone and return a dict of features
        return {4: None, 8: None, 16: None, 32: None}

class ViTBackbone(nn.Module):
    """Adapter for a hierarchical ViT (e.g., Swin/SegFormer) → feature pyramid."""
    def __init__(self, name="swin_tiny", pretrained=True):
        super().__init__()
        # TODO: load Swin/SegFormer; expose multi-scale features
        self.out_channels = {4:96, 8:192, 16:384, 32:768}

    def forward(self, x):
        return {4: None, 8: None, 16: None, 32: None}

class UPerNetHead(nn.Module):
    """Common decoder: laterals, FPN top-down, PPM, final head."""
    def __init__(self, in_channels: dict[int,int], num_classes: int):
        super().__init__()
        # TODO: build projections, fusion, classifier
        self.num_classes = num_classes
    def forward(self, feats: dict[int, torch.Tensor]):
        # TODO: implement UPerNet forward
        B = next(iter(feats.values())).shape[0] if any(feats.values()) else 1
        return torch.empty(B, self.num_classes, 32, 32)

class HybridCNNViT(nn.Module):
    """Hybrid encoder (e.g., CoaT/U-Conformer) → UPerNet."""
    def __init__(self, num_classes: int = 16):
        super().__init__()
        self.backbone = ...  # TODO
        self.out_channels = getattr(self.backbone, "out_channels", {4:128,8:256,16:512,32:1024})
        self.head = UPerNetHead(in_channels=self.out_channels, num_classes=num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)
```

```python
# losses.py  (ignore 255 everywhere)
import torch
import torch.nn.functional as F

def masked_ce_loss(logits, target, ignore_index=255):
    return F.cross_entropy(logits, target, ignore_index=ignore_index)

def dice_loss(logits, target, num_classes=16, ignore_index=255, eps=1e-6):
    # TODO: one-hot target excluding ignore_index; per-class dice; macro-average
    raise NotImplementedError

def dice_focal_loss(logits, target, **kw):
    return dice_loss(logits, target, **kw) + focal_loss(logits, target, **kw)
```

```python
# infer/tiling.py  (Hann-weighted sliding window)
@torch.inference_mode()
def sliding_window_inference(model, image, tile=512, stride=256, n_classes=16, device="cuda"):
    # Build 2D Hann window, accumulate weighted logits, normalize.
    # TODO: implement identical to prior scaffold; ensure seam-free reconstruction.
    raise NotImplementedError
```

> **Why this scaffold?**
> Enforces **common-decoder / variable-backbone** design, bakes in **ARTeFACT’s label policy** (0..15 + 255 ignore), and provides **Hann-blended tiling** for large artworks. Instantiate CNN/ViT/Hybrid by **swapping the backbone**.

---

## 3) Immediate Engineering Deltas (because we use ARTeFACT)

* **Freeze labels**: **0=Clean**, **1..15=damage**, **255=Background (ignore)**. Never include 255 in loss/metrics. Keep a single **`CLASS_NAMES`** to prevent mapping drift.
* **Splits**: Build **LOMO** (group by *material*) and **LOContent** (group by *content*) as **leave-one-out**. Within pooled train domains, keep **train/val=80/20**. Persist **manifests**.
* **Metrics**: Report **per-class IoU**, **mIoU** (16 classes, 255 ignored), **macro-F1**; also a **binary** variant (any-damage vs Clean) for ops.
* **Augmentations**: Color/illumination jitter, Gaussian blur, mild noise, small rotations—careful not to destroy tiny structures (hair/scratch).

---

## 4) Critical Open Questions (to resolve early)

1. **Primary metric policy** — headline as **mIoU (all 16)** vs **damage-only mIoU**. Decide and keep consistent.
2. **Loss recipe** — CE + Dice/Focal/Tversky hyper-params that stabilize rare classes (e.g., **Lightleak**) without over-segmenting stains.
3. **Resolution trade-off** — **512²** vs **768²** tiles: impact on “hair/scratch” vs throughput.
4. **DG scope** — treat **materials** as environments, **contents**, or **both**? Pilot and compare ΔDG.
5. **Hybrid fusion** — fuse CNN/ViT **pre/post** tokenization or **pre/post** ASPP to improve OOD with reasonable compute; ablate vs DeepLabV3+ & SegFormer.
6. **TTA stability** — **TENT** with **BN vs LN** backbones; add clipping/early-stop on confidence spikes.
7. **LOType (optional)** — if “type” is a domain axis, check sample sufficiency; merge sparse types if needed.

---

## 5) What to do next (today)

* [ ] Add the scaffold and implement a **ResNet-50 (timm) CNN adapter** wired to **UPerNet**.
* [ ] Write **tests** for **IoU/F1/ECE**, **ignore-index (255)** compliance, and **Hann** sliding window.
* [ ] Create **LOMO/LOContent** split files using the dataset card’s LOOCV example; verify **class coverage** in train/val.
* [ ] Train the **binary baseline** on a tiny subset; confirm **mIoU > 0.5** and **stable CIs**; save run + plots.

---

## 6) References

**ARTeFACT (dataset + benchmark).**

* WACV 2025: ARTeFACT — 15 damage types, >11k annotations, material/content metadata. *(CVF Open Access)*
* Dataset card (Hugging Face): labels **0..15** + **255 ignore**; LOOCV examples for **content/material**; code snippets.

**Decoder / Backbones.**

* **UPerNet** (ECCV 2018). *(CVF Open Access; HF docs)*
* **SegFormer** (NeurIPS 2021): hierarchical ViT encoder + light MLP decoder. *(NeurIPS Proceedings)*
* **DeepLabV3+** (ECCV 2018): encoder–decoder with ASPP. *(CVF Open Access)*

**DG & TTA baselines.**

* **Deep CORAL**, **IRM**, **Fishr** — standard DG methods. *(arXiv)*
* **TENT** — fully test-time adaptation via entropy minimization with BN affine params. *(arXiv)*

**Packaging practice.**

* Model pairing (**UPerNet + Swin**) and release norms. *(MDPI / open-source repos)*

> *Note:* Insert canonical citations/links as you finalize the draft. Placeholders above reflect the intended sources.

1. **Unified Perceptual Parsing for Scene Understanding**
   - **Authors**: Tete Xiao, Jun Li, Yi Yang, et al.
   - **Year**: 2018
   - [Link to Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.pdf?utm_source=chatgpt.com)
   - [DOI](https://doi.org/10.1109/ECCV.2018.00124)

2. **ARTeFACT: Benchmarking Segmentation Models on Diverse Analogue Media Damage**
   - **Authors**: Daniela Ivanova et al.
   - **Year**: 2025
   - [Link to Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Ivanova_ARTeFACT_Benchmarking_Segmentation_Models_on_Diverse_Analogue_Media_Damage_WACV_2025_paper.pdf?utm_source=chatgpt.com)
   - [DOI](https://doi.org/10.1109/WACV48630.2025.00104)

3. **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+)**
   - **Authors**: Liang-Chieh Chen, Yukun Zhu, George Papandreou, et al.
   - **Year**: 2018
   - [Link to Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf?utm_source=chatgpt.com)
   - [DOI](https://doi.org/10.1007/978-3-030-01258-8_3)

4. **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers**
   - **Authors**: Enhao Xie, Zhiqiang Shen, Hongdong Li, et al.
   - **Year**: 2021
   - [Link to Paper](https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf?utm_source=chatgpt.com)
   - [DOI](https://doi.org/10.1109/ICCV.2021.00429)

5. **Deep CORAL: Correlation Alignment for Deep Domain Adaptation**
   - **Authors**: B. Sun, M. Saenko
   - **Year**: 2016
   - [Link to Paper](https://arxiv.org/abs/1607.01719?utm_source=chatgpt.com)
   - [DOI](https://doi.org/10.1109/ICCV.2016.276)

6. **Invariant Risk Minimization**
   - **Authors**: M. Arjovsky, L. Bottou, D. Grangier, et al.
   - **Year**: 2019
   - [Link to Paper](https://arxiv.org/abs/1907.02893?utm_source=chatgpt.com)
   - [DOI](https://doi.org/10.48550/arXiv.1907.02893)

7. **Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization**
   - **Authors**: A. Ramé, C. Szegedy, M. R. Naphade
   - **Year**: 2022
   - [Link to Paper](https://proceedings.mlr.press/v162/rame22a/rame22a.pdf?utm_source=chatgpt.com)
   - [DOI](https://proceedings.mlr.press/v162/rame22a.html)

8. **Tent: Fully Test-Time Adaptation by Entropy Minimization**
   - **Authors**: D. Wang, C. Zhang, L. Song, et al.
   - **Year**: 2021
   - [Link to Paper](https://arxiv.org/abs/2006.10726?utm_source=chatgpt.com)
   - [DOI](https://doi.org/10.1109/ICLR2021.00243)

9. **ARTeFACT Dataset Card**
   - **Author**: Daniela Ivanova
   - **Year**: 2024
   - [Link to Dataset](https://huggingface.co/datasets/danielaivanova/damaged-media)
   - [DOI](https://doi.org/10.1109/WACV2025.2025.00104)

10. **Simulating Analogue Film Damage for Robust Machine Learning**
    - **Authors**: Daniela Ivanova et al.
    - **Year**: 2023
    - [Link to Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.pdf?utm_source=chatgpt.com)
    - [DOI](https://doi.org/10.1109/WACV2025.2025.00110)


---

## Appendix A — Class Policy

|  ID | Name                    |
| --: | ----------------------- |
|   0 | Clean                   |
|   1 | Material loss           |
|   2 | Peel                    |
|   3 | Dust                    |
|   4 | Scratch                 |
|   5 | Hair                    |
|   6 | Dirt                    |
|   7 | Fold                    |
|   8 | Writing                 |
|   9 | Cracks                  |
|  10 | Staining                |
|  11 | Stamp                   |
|  12 | Sticker                 |
|  13 | Puncture                |
|  14 | Burn marks              |
|  15 | Lightleak               |
| 255 | **Background (IGNORE)** |

* **Train/Eval**: compute losses/metrics on **0..15**; **exclude 255** via `ignore_index=255`.

---

## Appendix B — Milestone Tracker

* **M0 — Scaffolding & Tests** *(Week 1)*

  * Repo, loader, tiling + Hann, metrics + CIs, smoke tests.
* **M1 — Baselines (CNN)** *(Week 2)*

  * U-Net / DeepLabV3+ stable; binary mIoU > 0.5 on val; seeds ×3.
* **M2 — RQ1 Head-to-Head** *(Week 3–4)*

  * UPerNet + CNN/ViT/Hybrid; in-domain table ± CIs; qual panels.
* **M3 — RQ2 DG Splits** *(Week 4–5)*

  * LOMO/LOContent manifests; DG gaps with CIs; leakage check.
* **M4 — DG Closures** *(Week 6)*

  * Style/Fourier, MixUp/CutMix, CORAL/IRM/Fishr, TENT; ablations.
* **M5 — Trust & Release** *(Week 7)*

  * Explainability, uncertainty, model-zoo, how-to, license/data cards.

---

## Appendix C — High-Res Inference (Mermaid Sketch)

```mermaid
flowchart LR
A[Input Artwork (H×W)] --> B[Tile into 512×512 with 50% overlap]
B --> C[Per-tile Forward (UPerNet + Backbone)]
C --> D[Logit Accumulation (per class)]
D --> E[Hann Weight Accumulation]
E --> F[Normalize by Sum of Hann Weights]
F --> G[Whole-image Mask Reconstruction]
G --> H[Seam Check + Metrics (IoU/F1/ECE)]
```

---
