"""
Timm Encoder Wrapper for Segmentation Models PyTorch
Allows using any timm model as encoder backend
"""
import torch
import torch.nn as nn
import timm
from typing import List


class TimmEncoder(nn.Module):
    """Universal encoder wrapper for timm models"""
    
    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        
        # Create timm model with feature extraction
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
        )
        
        # Get output channels for each stage from timm's feature_info
        # timm returns 4-5 stages depending on model
        self._feature_channels = [info['num_chs'] for info in self.model.feature_info]
        
        # SMP DeepLabV3+ expects: [input_channels, stage0, stage1, stage2, stage3, stage4]
        # i.e., 6 elements total
        # We prepend input channels (3 for RGB)
        self.out_channels = [3] + self._feature_channels
        
        # Store model name for debugging
        self.name = name
        
    def forward(self, x):
        """
        Forward pass returning features at each stage
        Returns: [input, stage0, stage1, stage2, stage3, (stage4 if exists)]
        """
        features = self.model(x)
        
        # Fix Swin Transformer output format
        # Swin outputs (B, H, W, C) instead of (B, C, H, W)
        features_fixed = []
        for feat in features:
            if len(feat.shape) == 4 and feat.shape[-1] in self._feature_channels:
                # Swin format detected: (B, H, W, C) -> (B, C, H, W)
                feat = feat.permute(0, 3, 1, 2)
            features_fixed.append(feat)
        
        # Return input + features
        return [x] + features_fixed


def create_timm_encoder(name: str, pretrained: bool = True) -> TimmEncoder:
    """
    Factory function to create timm encoder
    
    Args:
        name: timm model name (e.g., 'convnext_tiny', 'swin_tiny_patch4_window7_224')
        pretrained: Load ImageNet pretrained weights
        
    Returns:
        TimmEncoder instance compatible with SMP
    """
    return TimmEncoder(name=name, pretrained=pretrained)
