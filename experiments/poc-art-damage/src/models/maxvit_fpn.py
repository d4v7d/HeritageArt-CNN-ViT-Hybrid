"""MaxViT Tiny backbone + FPN head for 16-class segmentation (hybrid PoC).

Uses timm's maxvit_tiny_tf_224.in1k with features_only=True to get multi-scale features
and decodes them to 16 logits with a lightweight FPN.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .convnext_fpn import FPN


class MaxViTTinyFPN(nn.Module):
    def __init__(self, num_classes: int = 16):
        super().__init__()
        self.backbone = timm.create_model(
            "maxvit_tiny_tf_224.in1k", pretrained=True, features_only=True
        )
        chs = self.backbone.feature_info.channels()
        self.decoder = FPN(in_channels=chs, out_channels=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decoder(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
