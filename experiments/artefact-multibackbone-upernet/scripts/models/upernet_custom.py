"""
Custom UPerNet Decoder Implementation
Unified Perceptual Parsing for Scene Understanding (arXiv:1807.10221)

Components:
- PPM (Pyramid Pooling Module): Multi-scale context aggregation
- FPN (Feature Pyramid Network): Multi-level feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    """Pyramid Pooling Module
    
    Pools features at multiple scales and fuses them.
    Original scales: [1, 2, 3, 6] from PSPNet
    """
    
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Create pooling + conv layers for each scale
        self.ppm_branches = nn.ModuleList()
        for scale in pool_scales:
            self.ppm_branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Bottleneck after concatenation
        # Input: original + (num_scales * out_channels)
        bottleneck_channels = in_channels + len(pool_scales) * out_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - typically from last encoder stage
        
        Returns:
            (B, out_channels, H, W) - multi-scale fused features
        """
        input_size = x.shape[2:]  # (H, W)
        
        # Original features
        ppm_outs = [x]
        
        # Multi-scale pooling
        for ppm_branch in self.ppm_branches:
            pooled = ppm_branch(x)
            # Upsample back to original size
            upsampled = F.interpolate(
                pooled,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
            ppm_outs.append(upsampled)
        
        # Concatenate all scales
        ppm_out = torch.cat(ppm_outs, dim=1)  # (B, bottleneck_channels, H, W)
        
        # Bottleneck fusion
        output = self.bottleneck(ppm_out)
        
        return output


class FPN(nn.Module):
    """Feature Pyramid Network
    
    Fuses multi-level features from encoder with top-down pathway.
    """
    
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list: List of channel dims from encoder stages
                              e.g., [96, 192, 384, 768] for ConvNeXt-Tiny
            out_channels: Unified output channels (typically 256)
        """
        super(FPN, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # Lateral connections (1x1 conv to unify channels)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1)
            )
        
        # Output convs (3x3 conv after upsampling + lateral addition)
        self.fpn_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, inputs):
        """
        Args:
            inputs: List of features from encoder stages (low to high resolution)
                    e.g., [stage1, stage2, stage3, stage4]
                    Typically stage4 is smallest (highest semantic level)
        
        Returns:
            List of FPN features at each level (same length as inputs)
        """
        assert len(inputs) == len(self.in_channels_list), \
            f"Expected {len(self.in_channels_list)} inputs, got {len(inputs)}"
        
        # Bottom-up: lateral connections
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down pathway with skip connections
        # Start from highest semantic level (smallest spatial)
        fpn_outs = []
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                # Highest level: no upsampling needed
                fpn_out = laterals[i]
            else:
                # Upsample previous level and add to current lateral
                prev_shape = laterals[i].shape[2:]
                upsampled = F.interpolate(
                    fpn_outs[-1],
                    size=prev_shape,
                    mode='bilinear',
                    align_corners=False
                )
                fpn_out = laterals[i] + upsampled
            
            # Apply output conv
            fpn_out = self.fpn_convs[i](fpn_out)
            fpn_outs.append(fpn_out)
        
        # Reverse to match input order (low to high resolution)
        fpn_outs.reverse()
        
        return fpn_outs


class UPerNetDecoder(nn.Module):
    """UPerNet Decoder: PPM + FPN
    
    Combines:
    1. PPM on highest semantic level for global context
    2. FPN for multi-level feature fusion
    """
    
    def __init__(self, in_channels_list, out_channels=256, 
                 ppm_pool_scales=(1, 2, 3, 6), dropout=0.1, num_classes=2):
        """
        Args:
            in_channels_list: Channel dims from encoder stages (e.g., [96, 192, 384, 768])
            out_channels: FPN output channels (default 256)
            ppm_pool_scales: PPM scales (default [1,2,3,6])
            dropout: Dropout rate before final classifier
            num_classes: Number of segmentation classes
        """
        super(UPerNetDecoder, self).__init__()
        
        # PPM on last (highest semantic) encoder stage
        self.ppm = PPM(
            in_channels=in_channels_list[-1],
            out_channels=out_channels,
            pool_scales=ppm_pool_scales
        )
        
        # FPN on all encoder stages
        # Replace last stage channels with PPM output
        fpn_in_channels = in_channels_list[:-1] + [out_channels]
        self.fpn = FPN(
            in_channels_list=fpn_in_channels,
            out_channels=out_channels
        )
        
        # Final fusion: concatenate all FPN levels
        # Then upsample to input resolution
        self.fusion = nn.Sequential(
            nn.Conv2d(
                out_channels * len(in_channels_list),  # Concat all FPN levels
                out_channels,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, num_classes, 1)
        )
    
    def forward(self, encoder_features, input_size=None):
        """
        Args:
            encoder_features: List of [stage1, stage2, stage3, stage4] from encoder
                              Ordered from high-res (stage1) to low-res (stage4)
            input_size: Optional (H, W) tuple for target output size
        
        Returns:
            logits: (B, num_classes, H, W) - segmentation logits at input resolution
        """
        # PPM on last stage (highest semantic, lowest resolution)
        ppm_out = self.ppm(encoder_features[-1])
        
        # Replace last stage with PPM output for FPN
        fpn_inputs = encoder_features[:-1] + [ppm_out]
        
        # FPN fusion
        fpn_outs = self.fpn(fpn_inputs)
        
        # Upsample all FPN levels to highest resolution (stage1 size)
        target_size = fpn_outs[0].shape[2:]
        upsampled_fpn = []
        for fpn_out in fpn_outs:
            if fpn_out.shape[2:] != target_size:
                upsampled = F.interpolate(
                    fpn_out,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                upsampled_fpn.append(upsampled)
            else:
                upsampled_fpn.append(fpn_out)
        
        # Concatenate all levels
        fused = torch.cat(upsampled_fpn, dim=1)
        
        # Final fusion conv
        fused = self.fusion(fused)
        
        # Classifier
        logits = self.classifier(fused)
        
        # Upsample to input resolution
        # If input_size provided, use it; otherwise calculate from stage1
        if input_size is not None:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
        else:
            # Calculate scale_factor from stage1 (typically 4x, but can vary)
            scale_factor = 4  # Default for most encoders
            logits = F.interpolate(
                logits,
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=False
            )
        
        return logits


if __name__ == '__main__':
    # Test with ConvNeXt-Tiny feature dims
    print("Testing UPerNet Decoder...")
    
    # ConvNeXt-Tiny feature channels: [96, 192, 384, 768]
    in_channels = [96, 192, 384, 768]
    batch_size = 2
    
    # Simulate encoder features at different scales
    # Assuming input is 512x512, encoder downsamples: /4, /8, /16, /32
    stage1 = torch.randn(batch_size, 96, 128, 128)    # /4
    stage2 = torch.randn(batch_size, 192, 64, 64)     # /8
    stage3 = torch.randn(batch_size, 384, 32, 32)     # /16
    stage4 = torch.randn(batch_size, 768, 16, 16)     # /32
    
    encoder_features = [stage1, stage2, stage3, stage4]
    
    # Create decoder
    decoder = UPerNetDecoder(
        in_channels_list=in_channels,
        out_channels=256,
        ppm_pool_scales=(1, 2, 3, 6),
        dropout=0.1,
        num_classes=2
    )
    
    # Forward pass
    logits = decoder(encoder_features)
    
    print(f"Output shape: {logits.shape}")  # Should be (2, 2, 512, 512)
    print(f"Expected: (2, 2, 512, 512)")
    
    assert logits.shape == (batch_size, 2, 512, 512), "Output shape mismatch!"
    print("✅ UPerNet decoder test passed!")
