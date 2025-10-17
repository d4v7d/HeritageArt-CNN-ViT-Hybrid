"""CoaT-Lite Small backbone + FPN head for 16-class segmentation.

We reuse the same FPN decoder as convnext_fpn. timm provides coat_lite_small backbone.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .convnext_fpn import FPN


class CoaTLiteSmallFPN(nn.Module):
    def __init__(self, num_classes: int = 16):
        super().__init__()
        self.backbone = timm.create_model(
            "coat_lite_small", pretrained=True, features_only=True
        )
        chs = self.backbone.feature_info.channels()
        self.decoder = FPN(in_channels=chs, out_channels=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decoder(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
