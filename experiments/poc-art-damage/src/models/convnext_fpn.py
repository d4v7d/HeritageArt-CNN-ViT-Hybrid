"""ConvNeXt-Tiny backbone + lightweight FPN head for 16-class segmentation.

We use timm to get an ImageNet-1k pretrained ConvNeXt-Tiny backbone. We tap multi-scale
features and pass them through a simple FPN decoder to produce 16-channel logits.

This is inference-only for the PoC, so no training code.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 128, num_classes: int = 16):
        super().__init__()
        # Laterals 1x1
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        # Smooth convs 3x3
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels])
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # feats expected: [C2, C3, C4, C5] low->high resolution
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        # top-down pathway
        x = lat[-1]
        for i in range(len(lat) - 2, -1, -1):
            x = F.interpolate(x, size=lat[i].shape[-2:], mode="bilinear", align_corners=False)
            x = x + lat[i]
            x = self.smooth[i](x)
        out = self.classifier(x)
        return out


class ConvNeXtTinyFPN(nn.Module):
    def __init__(self, num_classes: int = 16):
        super().__init__()
        # Do not hardcode out_indices; let timm choose valid feature maps for this model
        self.backbone = timm.create_model("convnext_tiny", pretrained=True, features_only=True)
        chs = self.backbone.feature_info.channels()  # e.g., may be [96, 192, 384, 768]
        self.decoder = FPN(in_channels=chs, out_channels=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # list of feature maps
        logits = self.decoder(feats)
        # upsample to input size
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
