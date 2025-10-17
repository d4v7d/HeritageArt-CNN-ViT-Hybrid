"""UPerNet-Swin Base via Hugging Face, with a fallback to SegFormer (no mmcv required).

We first attempt to load a public UPerNet-Swin checkpoint from Hugging Face. If that
requires mmcv/mmdet ops (unavailable in this PoC), we gracefully fall back to
`SegformerForSemanticSegmentation` (B2 ADE20K). In both cases, we project logits to 16
classes via a 1x1 conv to satisfy the PoC output shape. This does not imply a semantic
mapping between ADE20K and our 16 ARTeFACT-like classes—the PoC is focused on runtime
plumbing, not accuracy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    UperNetForSemanticSegmentation,
    SegformerForSemanticSegmentation,
)


class UPerNetSwinBase16(nn.Module):
    def __init__(self, num_classes: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.backend = "upernet"
        self.processor = None
        try:
            # Try UPerNet Swin Base (may require mmcv in some variants)
            self.processor = AutoImageProcessor.from_pretrained(
                "openmmlab/upernet-swin-base", trust_remote_code=True
            )
            self.model = UperNetForSemanticSegmentation.from_pretrained(
                "openmmlab/upernet-swin-base", trust_remote_code=True
            )
            in_ch = int(getattr(self.model.config, "num_labels", 150))
            self.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)
            self.backend = "upernet"
        except Exception:
            # Fallback: SegFormer B2 ADE20K (pure HF, no mmcv)
            self.processor = AutoImageProcessor.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512"
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512"
            )
            in_ch = int(getattr(self.model.config, "num_labels", 150))
            self.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)
            self.backend = "segformer"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inputs are already normalized (ImageNet). We call the model and adapt outputs.
        out = self.model(pixel_values=x)
        logits = out.logits  # [B, num_labels, h, w] (often upsampled to input size)
        proj = self.classifier(logits)
        if proj.shape[-2:] != x.shape[-2:]:
            proj = F.interpolate(proj, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return proj
