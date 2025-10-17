"""CoAtNet-0 backbone + FPN head for 16-class segmentation (hybrid stand-in for PoC).

This uses timm's coatnet_0 with features_only=True to obtain a feature pyramid and decode
with a lightweight FPN to 16 logits. Serves as a practical hybrid model for the PoC.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .convnext_fpn import FPN


class CoAtNet0FPN(nn.Module):
    def __init__(self, num_classes: int = 16):
        super().__init__()
        self.backbone = timm.create_model(
            "coatnet_0", pretrained=True, features_only=True
        )
        chs = self.backbone.feature_info.channels()
        self.decoder = FPN(in_channels=chs, out_channels=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decoder(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
