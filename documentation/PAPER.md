::: IEEEkeywords
Computer Vision; Cultural Heritage Preservation; Digital Heritage;
Artwork Deterioration Detection; Convolutional Neural Networks; Vision
Transformers
:::

# Introduction {#sec:Introduction}

Artwork conservation isa core aspect of cultural heritage stewardship.
Socially, it underpins social identity, community pride, and
education [@mishra_artificial_2024]. It sustains social identity,
community pride, and education [@mishra_artificial_2024], while also
contributing to tourism and the art market, generating jobs and
supporting economies [@ernst__young_global_limited_2014_creating_2014].
Conservation helps preserve the historical, aesthetic, and ethical value
of artworks for future
generations [@ernst__young_global_limited_2014_creating_2014].

The field of conservation can be broadly sliced to three branches:
preventive conservation, remedial conservation, and
restoration [@icom-cc_icom-cc_2009]. We focus on **preventive
conservation**, i.e. managing environments and risks to slow or avoid
deterioration before damage occurs [@icom-cc_icom-cc_2009]. Since damage
must be identified before being addressed, our analysis could also
assist restoration workflows.

Timely recognition of surface deterioration---such as abrasion, pigment
loss, craquelure, varnish degradation, or water staining---is
key [@ivanova_artefact_2024]. Conservators traditionally use visual
inspection and scientific imaging methods, including X-ray, infrared
reflectography, microscopy, and spectroscopy [@borg_application_2020].
Early computer-assisted tools included brushstroke
extraction [@lamberti_computer-assisted_2014], crack and craquelure
analysis [@sidorov_craquelure_2019; @dulecha_pixel-wise_2019], and
material-layer separation [@pu_cross-domain_2022]. However, these were
generally developed as case-specific tools with limited generalization
and no standardized benchmarks.

During the 2010s, **Convolutional Neural Networks** (CNNs) arose as an
architecture of deep learning models composed of multiple layers of
convolution filters and pooling operations that progressively extract
higher-level representations [@mauricio_comparing_2023]. CNN's became
the default machine learning (ML) tool for image classification,
segmentation and pattern recognition, due to their ability to learn
spatial hierarchies of features at multiple
scales [@yunusa_exploring_2024].

In 2020, the **Vision Transformers** (ViTs) [@dosovitskiy_image_2021]
architecture extended the Transformer architecture from natural language
processing to vision tasks, enabling modeling of global context and
long-range dependencies in a single
layer [@mauricio_comparing_2023].ViTs often match or exceed CNN
performance but typically demand larger datasets and higher compute
resources [@ali_vision_2023; @elharrouss_vits_2025; @yunusa_exploring_2024; @dosovitskiy_image_2021],
which are important considerations when adopting them for niche domains
like cultural heritage imaging.

Since 2019, AI applications to cultural heritage have accelerated,
spanning damage
detection [@kwon_automatic_2019; @samhouri_prediction_2022; @hou_using_2024],
materials analysis [@go_comparison_2025],
accessibility [@girbacia_analysis_2024; @mishra_artificial_2024], and
image reconstruction [@yi_zhang_artificial_2025]. These models automate
visual inspection and support scalable, less subjective analysis
compared to manual
methods [@sankar_transforming_2023; @yunusa_exploring_2024].

Yet, most studies use either CNNs or ViTs in isolation, rather than
benchmarking them directly [@ali_vision_2023; @roy_multimodal_2023]. A
notable exception is the 2024 **ARTeFACT
benchmark** [@ivanova_artefact_2024], which introduced a large annotated
dataset and tested several CNN and ViT models for multi-type damage
segmentation. This was accompanied by dataset documentation
efforts [@alkemade_datasheets_2023]. Another recent study compared CNNs
and ViTs for pigment classification [@go_comparison_2025], but not
damage detection.

Progress is also hindered by data scarcity. Licensing restrictions and
the cost of expert annotations [@ivanova_artefact_2024], combined with
institutional differences in materials, imaging, and damage taxonomies,
create cross-domain variability that affects model training and results
in poor generalization when moving across domains. Consequently, there
is strong interest in metadata automation [@diem_automatic_2023] and
transferable models.

We aim to address this by comparing three families of vision
architectures---CNNs, ViTs, and hybrids---for damage detection in
digitized artworks. Our research questions are:

-   Which model architecture (CNN, Vision Transformers, or hybrid
    CNN--ViT) achieves the best multi-category paint damage
    segmentation?

-   Which model architecture (CNN, ViT, or hybrid) generalizes best to
    unseen collections, materials, and institutions, minimizing the
    performance gap (in-domain vs. out-of-domain)?

We focus on damage segmentation in 2D artworks using RGB, X-ray, and
infrared modalities. While deployment in Costa Rican heritage
collections is our long-term goal, we leave it to future work pending
dataset availability.

# Previous Work {#sec:PreviousWork}

As computer vision has advanced, its intersection with cultural heritage
has grown, alongside a rapidly expanding body of literature. To situate
our contribution within this landscape, we surveyed literature from
2013--2025 on computer vision for cultural heritage, focusing on
image-based deterioration detection and on comparisons of Convolutional
Neural Networks (CNNs) versus Vision Transformers (ViTs). We queried
Scopus and arXiv (English/Spanish, keywords like "CNN", "ViT", "cultural
heritage", "damage detection") and screened for peer-reviewed studies
most relevant to our topic. Key findings from this review are summarized
in Table [\[tab:prevwork\]](#tab:prevwork){reference-type="ref"
reference="tab:prevwork"}

Early works applied classical image processing to assist restoration
(e.g. crack detection and virtual inpainting on paintings). With the
rise of deep learning, CNN-based detectors became prevalent for heritage
surfaces. For example, Kwon and Yu (2019) trained a Faster R-CNN to
automatically label cracks, losses, and biological growth on stone
relics, achieving high confidence (94%+ on test
images) [@kwon_automatic_2019]. Hatır et al. (2021) similarly used Mask
R-CNN to segment seven weathering forms on a Hittite rock sanctuary,
reporting mAP up to 89--100% per class in training [@hatir_deep_2021].
These case studies -- often on historic
façades [@wang_automatic_2024; @samhouri_prediction_2022],
monuments [@kwon_automatic_2019; @hatir_deep_2021],
murals [@yi_zhang_artificial_2025] or decorative
tiles [@karimi_deep_2024; @hu_integrating_2025] -- demonstrate that deep
CNNs can learn visible damage patterns (cracks, spills, detachments,
bio-colonization) with high in-sample accuracy. However, they generally
rely on domain-specific datasets and do not test generalization beyond
the original site or material. A few studies targeted painted artworks:
Angheluta and Chiroșca (2020) re-used high-resolution photogrammetry
images to train a CNN that detects cracks, blisters, and paint losses on
a polychrome wood icon, successfully highlighting deterioration regions
via activation maps [@angheluta_physical_2020]. Likewise, García-Moreno
et al. (2024) developed ARTDET, a Mask R-CNN--based tool to identify
lacunae (paint layer loss) in easel paintings, achieving  80% recall for
missing-paint and stucco areas [@garcia-moreno_artdet_2024]. Other case
studies have explored degradation classification in temple
walls [@huang_deep_2025], multispectral separation in panel
paintings [@pu_mixed_2022], and wall painting pixel
classification [@dulecha_crack_2019], among
others [@cornelis_crack_2013; @sankar_transforming_2023]. These works
confirmed the feasibility of automated damage mapping in art, but each
was limited to one damage type or artwork, with no direct comparison
between different model architectures.

Recently, researchers have begun assembling broader benchmarks. Ivanova
et al. (2024) introduced ARTeFACT [@ivanova_artefact_2024], a
first-of-its-kind dataset of 418 images with over 11,000 expert
annotations for 15 types of deterioration (e.g. craquelure, peeling,
staining) across diverse analogue media. Using ARTeFACT to evaluate
state-of-the-art segmentation models, they found that both CNN-backbone
networks (UPerNet with ConvNeXt) and Transformer-based networks
(Swin-Unet, SegFormer) fall short of reliable performance. For instance,
even framing the task as binary segmentation (damaged vs. clean), the
best supervised models reached only moderate pixel-level accuracy
(F1\~0.5--0.6) and struggled to generalize across different materials
and image content [@ivanova_artefact_2024]. No single architecture (CNN
or ViT) clearly outperformed the other in all cases, suggesting each has
complementary strengths and limitations.

Comparisons between CNNs, ViTs, and hybrids are also emerging across
domains, including pigment imaging [@go_comparison_2025], portrait
classification [@diem_automatic_2023], and artwork
identification [@wang_fusion_2025]. In fact, hybrid approaches are
gaining attention---for example, a CNN--ViT fusion recently outperformed
standalone models in anomaly detection for solar
farms [@darban_anomaly_2025] and for surface degradation in
vineyards [@leite_comparative_2024]---hinting that combining both
feature types could benefit cultural heritage tasks as well.

Several surveys and scientometric reviews highlight this trend. Mishra
and Lourenço (2024) reviewed AI applications across CH
subdomains [@mishra_artificial_2024], Rathi et al. (2025) synthesized
deep models for fine art classification [@rathi_survey_2025], and Yunusa
et al. (2024) surveyed hybrid CNN--ViT models for vision
tasks [@yunusa_survey_2024], all noting limited deployment in
deterioration detection.

In summary, prior work shows growing interest in AI-driven visual
inspection for cultural heritage. Deep CNNs have achieved high accuracy
in detecting certain damages under controlled settings, and ViTs are
being explored in related classification tasks. However, there remains
no comprehensive study comparing traditional methods, CNNs, ViTs, and
hybrid models on a common problem in artwork deterioration. Most studies
address a single domain or damage type, often with small datasets and ad
hoc metrics, making their results hard to generalize. The difficulty of
robust damage detection across different artworks and conditions
persists. This gap motivates our work: we seek to benchmark a hybrid
CNN-ViT on a multi-type deterioration segmentation task, thereby
advancing towards generalizable and reproducible solutions.

# Methods {#sec:Methodology}

NOTA, EN LAS PRUEBAS HAY POSIB9ILIDADES DE QUE SALGAN PRUEBAS/TEST SETS
SIN CATEGORIAS SI LA CATEGORIA ES MUY PEQUENA

## Overview

To answer RQ1 and RQ2, we implemented a unified segmentation pipeline
that evaluates the performance of three architecture families---a CNN
(ConvNeXt), a Transformer (Swin), and a hybrid (MaxViT)---on multi-class
deterioration segmentation. We used the ARTeFACT
dataset [@ivanova_artefact_2024] for training and evaluation. Our focus
was twofold: benchmarking per-class segmentation performance in-domain
(RQ1), and evaluating cross-domain generalization (RQ2).

## Dataset

#### ARTeFACT Benchmark {#artefact-benchmark .unnumbered}

We used the ARTeFACT dataset [@ivanova_artefact_2024] as our primary
training and test corpus. It includes 418 high-resolution images of
artworks from diverse media and styles, annotated with 15 pixel-level
damage classes plus a \"Clean\" label. The dataset is stratified by
material and content type, but presents class imbalance and
heterogeneous damage scales: many images contain large clean areas with
only small damaged regions, and certain damage categories have far fewer
examples than others. We accounted for this in model training (via loss
function) and evaluation (using macro-averaged metrics).

#### Data Splits {#data-splits .unnumbered}

Following the original dataset's
recommendation [@ivanova_artefact_2024], we created a stratified split
such that the training set contains roughly 70% of the images, the
validation set 15%, and the test set 15%. This yielded approximately 290
train images, 60 val, 60 test (exact counts vary slightly by ensuring at
least one example of each damage type in each set). The stratification
helps prevent a scenario where, for example, a particular material or
damage type is completely missing from validation or test, which could
unfairly skew results. No images from the same artwork or collection
appear in both training and test. All model selection (hyperparameter
tuning, early stopping) was done on the validation set, and final
performance for RQ1/RQ2 was reported on the held-out test set.

#### Data Preprocessing {#data-preprocessing .unnumbered}

All images were kept in RGB color and no significant color correction or
filtering was applied beyond what the dataset provides. We resized or
downsampled images only if needed to fit memory during patch processing,
otherwise maintaining the original resolution for maximal detail. Prior
to feeding to the models, image pixel values were normalized to the
range expected by the pretrained backbones (for ImageNet-pretrained
models, we subtracted the mean and divided by the standard deviation of
ImageNet training data). We did not perform explicit contrast
enhancement or denoising, as the provided scans are generally of good
quality [@ivanova_artefact_2024]. However, our data augmentation
implicitly covered some variations in color and scale.

## Model Architecture and Training Pipeline

To address our research questions, we implemented a semantic
segmentation pipeline with three different model architectures. All
three models share a similar encoder-decoder design: an encoder backbone
(convolutional, transformer, or hybrid) extracts multi-scale features
from the input image, and a decoder head produces a pixel-wise
classification (damage type) mask. For a fair comparison, we used the
same decoder framework for all models -- U-Net [@ronneberger_u-net_2015]. 
We chose U-Net for its proven effectiveness in dense prediction tasks and 
its ability to combine low-level and high-level features through skip 
connections. This ensures that differences in performance can be attributed 
mainly to the backbone architecture (CNN vs ViT vs hybrid), not the decoding 
mechanism. All models were trained at 384×384 pixel resolution to balance 
memory efficiency with spatial detail preservation. Below we detail each
model variant:

### CNN Model -- ConvNeXt-Tiny + U-Net

For the convolutional baseline, we used ConvNeXt-Tiny as the backbone 
encoder. ConvNeXt is a modernized CNN architecture with ViT-inspired design
improvements [@liu_convnet_2022]. It builds on the ResNet paradigm but 
incorporates transformer-inspired features such as larger kernel convolutions 
(7×7), depthwise convolutions, LayerNorm instead of BatchNorm, and GELU 
activation functions, resulting in a pure-CNN that achieves competitive 
performance with vision transformers [@liu_convnet_2022]. We selected 
ConvNeXt-Tiny (33.1M parameters) to maintain a model size appropriate for 
our dataset scale while avoiding overfitting. The hierarchical feature maps 
extracted at multiple scales (1/4, 1/8, 1/16, 1/32 of input resolution) are 
fed to the U-Net decoder, which progressively upsamples and combines features 
through skip connections to produce the final segmentation mask.

#### Pretraining {#pretraining .unnumbered}

We initialized ConvNeXt-Tiny with weights pretrained on ImageNet-1K. 
The U-Net decoder was randomly initialized and trained from scratch on 
the ARTeFACT dataset. This approach follows standard transfer learning 
practices where pretrained encoders provide strong visual representations 
that are then adapted to the target segmentation task.

### Transformer Model -- SegFormer-B3 + U-Net

For the pure vision transformer approach, we used SegFormer-B3 (Mix 
Transformer B3, MiT-B3) as the backbone encoder [@xie_segformer_2021]. 
SegFormer represents a hierarchical transformer specifically designed for 
segmentation tasks. Unlike standard ViTs that produce single-scale features, 
SegFormer's Mix Transformer encoder generates multi-scale feature maps at 
four different resolutions through progressive patch merging and efficient 
self-attention mechanisms [@xie_segformer_2021]. The B3 variant (45.0M 
parameters) was selected to maintain comparable model capacity with our CNN 
baseline while representing state-of-the-art transformer segmentation 
approaches. SegFormer's key innovation is its hierarchical structure that 
eliminates the need for positional encodings, making it more robust to 
resolution changes. We coupled the SegFormer encoder with the same U-Net 
decoder used for all models, ensuring architectural consistency in our 
comparison.

#### Pretraining

The SegFormer-B3 encoder was initialized with weights pretrained on 
ImageNet-1K. As with the CNN model, the U-Net decoder was randomly 
initialized. The pretrained transformer backbone provides strong semantic 
feature extraction capabilities that are then fine-tuned on the ARTeFACT 
damage segmentation task.

### Hybrid Model -- MaxViT-Tiny + U-Net

As an example of a hybrid architecture that combines convolutional and 
transformer elements, we chose MaxViT-Tiny [@tu_maxvit_2022] as our third 
model. MaxViT (Multi-Axis Vision Transformer) strategically integrates 
convolution blocks with multi-axis self-attention mechanisms to capture both 
local patterns and global context [@tu_maxvit_2022]. Each MaxViT stage 
contains two types of blocks: (1) MBConv blocks that apply depthwise 
convolutions for local feature extraction with strong inductive biases, and 
(2) transformer blocks that perform both local window attention and grid 
attention across the entire feature map for global receptive fields. This 
design aims to combine the strengths of CNNs (translation equivariance, 
local pattern recognition, computational efficiency) with the strengths of 
ViTs (long-range dependencies, flexible receptive fields, superior 
generalization). We selected the Tiny variant (31.0M parameters) to maintain 
comparable model capacity with our other architectures. As with the other 
models, MaxViT-Tiny serves as the encoder backbone paired with the U-Net 
decoder.

#### Pretraining

The MaxViT-Tiny encoder was initialized with weights pretrained on 
ImageNet-1K. The U-Net decoder was randomly initialized and trained from 
scratch. This hybrid backbone provides a unique combination of local and 
global feature extraction that we hypothesize may be particularly effective 
for diverse damage patterns in heritage artworks.

## Training Procedure {#training-procedure .unnumbered}

To ensure fair comparison across architectures, all models followed an 
identical training pipeline with only architecture-specific hyperparameters 
adjusted. We trained on the augmented ARTeFACT dataset using an 80/20 
train-validation split, processing images at 384×384 pixel resolution. This 
resolution balances computational efficiency with sufficient spatial detail 
for damage detection.

### Data Augmentation

Our augmentation strategy combined geometric and photometric transformations 
to improve model robustness. We applied random horizontal flips (p=0.5), 
random rotation (±15°), random scaling (0.9-1.1×), and elastic 
deformations. For photometric augmentation, we used random brightness and 
contrast adjustments (±10%), random gamma correction (0.9-1.1), and Gaussian 
blur (σ ∈ [0, 1.5]). We intentionally avoided aggressive color jittering to 
preserve damage-specific chromatic signatures (e.g., brown stains, yellow 
fading). Training patches were sampled uniformly from the full-resolution 
images, exposing models to damage patterns at various scales and positions.

### Optimization and Regularization

We employed the AdamW optimizer [@loshchilov_decoupled_2019] with a weight 
decay of 0.01 for all models. To account for different GPU memory 
requirements of each architecture, we adjusted batch sizes while maintaining 
effective learning rates through proportional scaling: ConvNeXt-Tiny used 
batch size 96 with learning rate 0.001, SegFormer-B3 used batch size 32 with 
learning rate 0.000333 (scaled by 32/96), and MaxViT-Tiny used batch size 48 
with learning rate 0.0005 (scaled by 48/96). This ensures comparable 
gradient statistics across architectures. We used OneCycleLR scheduling with 
a maximum learning rate at 30% of training and cosine annealing afterward. 
All models trained for 50 epochs on an NVIDIA Tesla V100S PCIe (32GB VRAM).

### Loss Function

We addressed class imbalance through a carefully designed loss function. 
After systematic ablation studies, we selected a Dice-Focal hybrid loss with 
balanced class weights. The Dice component encourages spatial overlap for 
each class independently, while the Focal component [@lin_focal_2017] 
down-weights easy examples to focus learning on hard cases. Class weights 
were computed using inverse square-root log scaling on the class 
frequencies, producing a balanced weight ratio of 36.4:1 (minimum to 
maximum). This was significantly more stable than direct inverse frequency 
weighting (734:1 ratio), which caused training collapse. The loss function 
can be written as:

$$\mathcal{L} = \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}} + \lambda_{\text{Focal}} \mathcal{L}_{\text{Focal}}$$

where both components use the pre-computed balanced class weights. This 
compound loss encourages the model to learn all damage categories while 
maintaining training stability.

## Inference {#inference .unnumbered}

For validation and testing, we performed inference at the same 384×384 pixel 
resolution used during training. Unlike training where we sample random 
patches, at inference time we process each full image using a sliding window 
approach with 50% overlap between adjacent windows. This overlap provides 
redundancy that improves boundary predictions. Each window is fed through the 
model independently, producing a segmentation prediction. The overlapping 
predictions are then merged using Gaussian weighting, where central pixels 
receive higher confidence than edge pixels. This reduces boundary artifacts 
common in patch-based inference [@pielawski_introducing_2020]. The final 
per-pixel class prediction is determined by argmax over the accumulated class 
logits. This inference strategy balances computational efficiency with 
prediction quality, ensuring consistent segmentation across the entire image.

## Evaluation Metrics {#evaluation-metrics .unnumbered}

We evaluate model performance using multiple complementary metrics to assess 
both classification accuracy and spatial localization quality. Our primary 
metric is mean Intersection-over-Union (mIoU), computed as the macro-average 
across all 16 damage classes (including the "Clean" background class). For 
each class $c$, IoU is defined as:

$$\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}$$

where $TP_c$ are true positive pixels, $FP_c$ are false positives, and 
$FN_c$ are false negatives. The mean IoU averages these per-class values, 
giving equal weight to rare and common categories. This macro-averaging 
approach prevents the abundant "Clean" class from dominating the metric.

We additionally report per-class Precision, Recall, and F1-score to provide 
insight into type I and type II error rates for each damage category:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad \text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

These per-class metrics reveal which damage types each architecture handles 
well versus poorly. For computational efficiency analysis, we measure 
inference time (milliseconds per image) on the validation set, averaged over 
all images. This provides a practical measure of deployment feasibility.
`Material Loss` are typically easier to detect than fine-grained classes
like `Dust` or `Cracks`.

These metrics enable a quantitative comparison of architecture families
(RQ1) and assessment of cross-domain performance drop (RQ2), revealing
which models best generalize under distribution shift.

# Results {#sec:Results}

To evaluate the comparative performance of the three architecture 
families---CNN (U-Net + ConvNeXt-Tiny), Transformer (U-Net + SegFormer-B3), 
and Hybrid (U-Net + MaxViT-Tiny)---we report pixel-level segmentation 
accuracy across all 16 ARTeFACT damage categories. Our evaluation addresses 
Research Question 1: *Which model architecture achieves the best 
multi-category paint damage segmentation?*

## Overall Performance

Table~\ref{tab:overall_performance} summarizes the overall performance and 
computational characteristics of each architecture. The hybrid MaxViT-Tiny 
model achieved the highest mean IoU of 37.55%, outperforming both the pure 
transformer SegFormer-B3 (35.64%) and the CNN ConvNeXt-Tiny (30.69%). This 
represents a 22% relative improvement over the CNN baseline. Interestingly, 
the hybrid model also achieved the best balance between accuracy and 
efficiency, with an inference time of 7.52 ms per image---only 0.45 ms slower 
than the fastest CNN model and actually faster than the transformer model 
(8.05 ms).

\begin{table}[h]
\centering
\caption{Overall Performance Comparison}
\label{tab:overall_performance}
\begin{tabular}{lccccc}
\hline
\textbf{Model} & \textbf{Family} & \textbf{Params} & \textbf{mIoU (\%)} & \textbf{Inference (ms)} & \textbf{Throughput} \\
\hline
ConvNeXt-Tiny & CNN & 33.1M & 30.69 & 7.07 & 122.6 imgs/s \\
SegFormer-B3 & ViT & 45.0M & 35.64 & 8.05 & 81.9 imgs/s \\
MaxViT-Tiny & Hybrid & 31.0M & \textbf{37.55} & 7.52 & 65.1 imgs/s \\
\hline
\end{tabular}
\end{table}

Figure~\ref{fig:overall_comparison} visualizes these results, highlighting 
that MaxViT achieves superior segmentation quality with parameter efficiency 
(31M parameters vs. SegFormer's 45M) and competitive inference speed. The 
CNN's advantage in throughput (122.6 imgs/s) comes at the cost of 
significantly lower accuracy.

## Per-Class Performance

Table~\ref{tab:perclass_iou} presents detailed per-class IoU scores, 
revealing nuanced architectural strengths. The hybrid model dominates across 
most categories, achieving the best performance on 11 out of 16 classes. 
Notably, MaxViT excels at challenging categories that require both local and 
global context: Material_loss (81.95%), Peel (65.33%), Fold (50.41%), 
Writing (53.16%), and Staining (52.47%).

\begin{table}[h]
\centering
\caption{Per-Class IoU Performance (\%) - All Damage Categories}
\label{tab:perclass_iou}
\small
\begin{tabular}{lccc}
\hline
\textbf{Damage Type} & \textbf{CNN} & \textbf{ViT} & \textbf{Hybrid} \\
\hline
Clean & 92.80 & 94.47 & \textbf{94.99} \\
Material\_loss & 70.68 & 77.69 & \textbf{81.95} \\
Peel & 52.01 & 61.07 & \textbf{65.33} \\
Dust & 0.00 & 0.00 & 0.00 \\
Scratch & 2.58 & 20.52 & \textbf{23.58} \\
Hair & 0.00 & 0.00 & 0.00 \\
Dirt & 7.86 & 14.94 & \textbf{25.67} \\
Fold & 32.84 & 48.92 & \textbf{50.41} \\
Writing & 24.92 & 50.56 & \textbf{53.16} \\
Cracks & 10.18 & 28.46 & \textbf{29.00} \\
Staining & 13.75 & 41.93 & \textbf{52.47} \\
Stamp & 0.00 & 0.00 & \textbf{60.57} \\
Sticker & \textbf{62.50} & 68.47 & 0.00 \\
Puncture & \textbf{69.03} & 0.00 & 0.00 \\
Burn\_marks & 0.00 & 0.01 & 0.00 \\
Lightleak & 51.83 & 63.23 & \textbf{63.70} \\
\hline
\textbf{Mean IoU} & 30.69 & 35.64 & \textbf{37.55} \\
\hline
\end{tabular}
\end{table}

The transformer model shows particular strength on Writing (50.56%) and Fold 
(48.92%), suggesting its global attention mechanisms effectively capture 
elongated and scattered damage patterns. The CNN baseline, while generally 
weaker, achieves competitive performance on large, well-defined regions 
(Clean: 92.80%, Puncture: 69.03%, Sticker: 62.50%).

Notably, several damage categories prove extremely challenging for all 
architectures: Dust (0.00% IoU for all models), Hair (0.00%), and Burn_marks 
(effectively 0.00%). These categories suffer from extreme class imbalance, 
with very few annotated examples in the dataset, and exhibit subtle visual 
characteristics that are difficult to distinguish from the undamaged 
substrate.

## Quantitative Analysis

### Architecture-Specific Strengths

Figure~\ref{fig:per_class_iou} visualizes the per-class performance across 
all three architectures, revealing distinct patterns:

**CNN (ConvNeXt-Tiny)** performs reliably on large, visually salient 
categories with clear boundaries. It achieves strong IoU on Clean (92.80%), 
Material_loss (70.68%), and Puncture (69.03%). However, its performance 
degrades significantly on fine-grained or scattered damage types. For 
Scratch (2.58%), Dirt (7.86%), and Cracks (10.18%), the purely convolutional 
architecture struggles to capture small-scale or elongated patterns. This 
limitation likely stems from the limited receptive field of convolutional 
operations, even with the modernized ConvNeXt architecture.

**Transformer (SegFormer-B3)** consistently outperforms the CNN baseline 
across most categories, achieving a 16% relative improvement in mean IoU. Its 
global self-attention mechanisms prove particularly effective for categories 
requiring long-range context: Writing (50.56% vs CNN's 24.92%), Staining 
(41.93% vs 13.75%), and Scratch (20.52% vs 2.58%). The transformer 
excels at recognizing dispersed or elongated damage patterns where spatial 
relationships extend beyond local neighborhoods. However, it completely fails 
on Puncture (0.00%) and Sticker (0.00% vs CNN's 62.50%), suggesting that 
pure attention-based models may struggle with certain compact, well-defined 
damage types that benefit from convolutional inductive biases.

**Hybrid (MaxViT-Tiny)** achieves the best overall performance by combining 
convolutional and attention mechanisms. It dominates on 11 out of 16 
categories, including challenging ones like Staining (52.47%), Writing 
(53.16%), and Dirt (25.67%). The hybrid architecture appears to leverage 
convolutional blocks for local pattern recognition while using multi-axis 
attention for global context. This combination proves particularly effective 
for heterogeneous damage that varies in scale and spatial distribution. 
Notably, MaxViT achieves a remarkable 60.57% IoU on Stamp---a category where 
both CNN and ViT completely fail (0.00%)---demonstrating the value of 
architectural diversity.

### Failure Modes

Despite architectural differences, all three models exhibit systematic 
failures on specific categories. Dust, Hair, and Burn_marks receive IoU 
values of zero (or near-zero) across all architectures. Analysis of the 
dataset reveals these categories suffer from extreme scarcity: they appear in 
fewer than 1% of training images and occupy minuscule pixel fractions when 
present. Their visual manifestation is also highly ambiguous---dust particles 
may be indistinguishable from image noise, hair strands are thread-thin, and 
burn marks exhibit subtle tonal variations that blend with natural color 
variation in aged artworks.

The inconsistent performance on Sticker (CNN: 62.50%, ViT: 68.47%, Hybrid: 
0.00%) and Puncture (CNN: 69.03%, ViT and Hybrid: 0.00%) is particularly 
intriguing. This suggests that different models have learned different 
feature hierarchies, and there may be annotation inconsistencies or visual 
ambiguities in these categories that lead to divergent learned 
representations. This variability highlights the challenge of establishing 
consistent damage taxonomies across diverse heritage collections.

# Discussion {#sec:Discussion}

Our empirical comparison of CNN, transformer, and hybrid architectures for 
heritage artwork damage segmentation reveals several important findings that 
address our research questions and provide guidance for future system 
deployment.

## Addressing Research Question 1: Best Architecture

**RQ1: Which model architecture (CNN, Vision Transformers, or hybrid CNN-ViT) 
achieves the best multi-category paint damage segmentation?**

The hybrid MaxViT-Tiny architecture achieves the highest overall performance 
(37.55% mIoU), followed by the pure transformer SegFormer-B3 (35.64%), and 
the CNN ConvNeXt-Tiny (30.69%). This ordering suggests that architectural 
diversity---combining convolutional inductive biases with global 
attention---provides measurable benefits for the heterogeneous damage 
patterns present in heritage artworks.

The superiority of MaxViT over SegFormer, despite having fewer parameters 
(31M vs 45M), is particularly noteworthy. We hypothesize three contributing 
factors:

1. **Multi-scale feature fusion**: MaxViT's alternating convolution and 
attention blocks allow it to capture both fine-grained local textures (via 
MBConv) and global spatial relationships (via multi-axis attention) within 
each stage. This may be better suited to damage that varies dramatically in 
scale---from millimeter-scale cracks to centimeter-scale material loss.

2. **Inductive bias balance**: Pure transformers lack built-in translation 
equivariance, requiring more data to learn basic visual concepts. MaxViT's 
convolutional components provide this inductive bias, potentially improving 
sample efficiency on our moderately-sized dataset (1,458 training images).

3. **Local-global complementarity**: Many damage types exhibit both local 
signatures (texture, color) and global context cues (shape, location 
relative to artwork regions). MaxViT's dual-pathway design may enable better 
integration of these complementary information sources.

The CNN's relatively weaker performance (30.69% mIoU) aligns with prior 
findings that pure convolutional architectures struggle with long-range 
dependencies and scattered patterns [@dosovitskiy_image_2021]. However, its 
superior training throughput (122.6 imgs/s) and minimal inference latency 
(7.07 ms) make it a viable option for resource-constrained deployment 
scenarios where 30% mIoU is acceptable.

## Comparison with ARTeFACT Baseline

Ivanova et al. [@ivanova_artefact_2024] reported a top performance of 40.6% 
mIoU using UPerNet with a Swin-Large backbone (197M parameters) on the 
ARTeFACT dataset. Our best model (MaxViT-Tiny, 31M parameters) achieves 
37.55% mIoU---approximately 92% of their performance with only 16% of the 
parameters. This comparison is not entirely fair, as their model used 
UPerNet's more sophisticated decoder, ADE20K-pretrained decoder weights, and 
a significantly larger backbone. Nevertheless, our results demonstrate that 
efficient hybrid architectures can approach state-of-the-art performance with 
substantially lower computational requirements, making them more practical for 
institutions with limited GPU resources.

## Per-Category Insights

The per-class analysis reveals that architectural choice should ideally be 
informed by the specific damage types of interest:

- **For large, compact damage** (Material_loss, Peel, Puncture): All 
architectures perform reasonably well (>50% IoU), with CNNs being sufficient 
if computational efficiency is prioritized.

- **For scattered or elongated damage** (Writing, Staining, Scratch): 
Transformers and hybrids substantially outperform CNNs (often 2-3× 
improvement), justifying their higher computational cost.

- **For rare or subtle damage** (Dust, Hair, Burn_marks): No architecture 
succeeds with current training strategies, indicating the need for 
specialized approaches (e.g., synthetic data augmentation, focal loss with 
extreme weighting, or few-shot learning).

## Addressing Research Question 2: Generalization

**RQ2: Which model architecture (CNN, ViT, or hybrid) generalizes best to 
unseen collections, materials, and institutions?**

Unfortunately, due to time and resource constraints, we were unable to 
conduct comprehensive cross-domain evaluation on external heritage 
collections. The ARTeFACT dataset itself combines multiple institutions and 
imaging modalities, providing some domain diversity within the training set. 
However, a rigorous assessment of generalization would require testing on 
completely held-out collections with different damage taxonomies, imaging 
protocols, and material substrates. This remains an important direction for 
future work.

Based on architectural principles, we hypothesize that hybrid models may 
offer better generalization due to their balanced inductive biases. Pure 
CNNs may overfit to local texture patterns specific to the training 
distribution, while pure transformers may struggle with limited training data 
from new domains. The hybrid approach potentially combines robust local 
feature extraction with flexible global reasoning, but empirical validation 
on cross-domain benchmarks is needed.

## Practical Deployment Considerations

For institutions planning to deploy automated damage detection systems, we 
offer the following recommendations:

1. **If accuracy is paramount** and GPU resources are available: Use MaxViT-
Tiny or SegFormer-B3. The 2% mIoU difference favors MaxViT, but SegFormer 
may be preferable if computational budget allows for its larger parameter 
count and you prioritize long-range pattern recognition.

2. **If inference speed is critical** (e.g., interactive tools, mobile 
deployment): Use ConvNeXt-Tiny. Its 7.07 ms inference time enables near-
real-time processing, and its 30.69% mIoU may be sufficient for initial 
screening or coarse damage surveys.

3. **For ensemble approaches**: The divergent predictions on Sticker and 
Puncture suggest that combining all three architectures via ensemble voting 
could further improve robustness. This would increase computational cost but 
may achieve superior performance on edge cases.

## Limitations

Several limitations qualify our findings:

1. **Single-dataset evaluation**: Our experiments use only the ARTeFACT 
dataset. While it spans multiple institutions and modalities, results may not 
generalize to collections with substantially different characteristics (e.g., 
Asian lacquerware, modern acrylic paintings, outdoor sculptures).

2. **No cross-validation**: Due to computational constraints, we performed a 
single 80/20 train-validation split. Results might vary with different random 
splits, though we expect the overall architectural ranking to remain stable.

3. **Parameter count variation**: Perfect parameter matching across 
architectures is impossible due to fundamental design differences (SegFormer-
B3: 45M, ConvNeXt-Tiny: 33.1M, MaxViT-Tiny: 31M). However, all models are in 
the same order of magnitude, making comparisons meaningful.

4. **Limited decoder exploration**: We standardized on U-Net for fair 
encoder comparison, but other decoders (PSPNet, DeepLabV3+, UPerNet) might 
interact differently with each backbone architecture.

5. **Class imbalance persistence**: Despite balanced class weights, extremely 
rare categories (Dust, Hair, Burn_marks) remain undetected. More aggressive 
resampling or synthetic data generation may be needed.

## Future Work

Several promising directions emerge from this study:

1. **Cross-domain benchmarking**: Evaluate all three architectures on 
external heritage collections to rigorously assess generalization (RQ2).

2. **Hierarchical damage taxonomies**: Investigate whether grouping rare 
categories into coarser superclasses (e.g., "surface deposits" = Dust + Dirt) 
improves learning while maintaining practical utility.

3. **Few-shot and zero-shot methods**: Explore meta-learning or prompt-based 
approaches to handle new damage types without extensive retraining.

4. **Multi-modal fusion**: Integrate X-ray, infrared, and UV imaging 
alongside RGB to provide complementary information about subsurface and 
material-specific damage.

5. **Active learning deployment**: Develop interactive systems where 
conservators correct model predictions, and these corrections are used to 
continuously improve the model through active learning loops.

6. **Ensemble optimization**: Systematically explore ensemble strategies 
(voting, stacking, knowledge distillation) to combine the complementary 
strengths observed across architectures.

7. **Computational optimization**: Investigate knowledge distillation to 
compress hybrid models for edge deployment, or neural architecture search to 
discover optimal hybrid designs for heritage-specific tasks.
7. **Computational optimization**: Investigate knowledge distillation to 
compress hybrid models for edge deployment, or neural architecture search to 
discover optimal hybrid designs for heritage-specific tasks.

# Conclusion

This study provides a systematic empirical comparison of three architecture 
families---CNNs, Vision Transformers, and hybrid models---for automated 
damage detection in heritage artworks. Using the ARTeFACT benchmark dataset, 
we trained ConvNeXt-Tiny (CNN), SegFormer-B3 (ViT), and MaxViT-Tiny (Hybrid) 
with identical U-Net decoders, training procedures, and loss functions to 
ensure fair architectural comparison.

Our results demonstrate that the hybrid MaxViT-Tiny architecture achieves the 
best performance (37.55% mIoU), outperforming both the pure transformer 
(35.64%) and CNN (30.69%) while maintaining competitive inference speed (7.52 
ms/image) and parameter efficiency (31M parameters). This finding suggests 
that combining convolutional inductive biases with global attention 
mechanisms provides measurable benefits for the heterogeneous damage patterns 
present in heritage artworks. The pure transformer excels at scattered and 
elongated damage requiring long-range context, while the CNN offers superior 
computational efficiency for resource-constrained scenarios.

All architectures struggle with extremely rare damage categories (Dust, Hair, 
Burn_marks), indicating that architectural innovations alone cannot overcome 
severe class imbalance---specialized training strategies or synthetic data 
augmentation are needed. Future work should focus on cross-domain evaluation 
to assess generalization, hierarchical damage taxonomies to handle rare 
classes, and multi-modal fusion to leverage complementary imaging modalities.

For conservation practitioners, we recommend MaxViT-Tiny or SegFormer-B3 when 
accuracy is paramount, ConvNeXt-Tiny for real-time applications, and ensemble 
approaches for maximum robustness. As heritage institutions increasingly 
adopt AI-assisted workflows, these architectural insights can guide deployment 
decisions based on specific collection characteristics and computational 
resources.

# Acknowledgment {#acknowledgment .unnumbered}

This research was supported by the School of Computer Science and
Informatics (ECCI), and the Center for Research in Information and
Communication Technologies (CITIC), both of the University of Costa Rica
(UCR).