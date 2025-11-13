"""
Custom Timm Encoder Wrapper for SMP
Allows using any timm model as encoder in segmentation_models_pytorch
"""
import torch
import torch.nn as nn
import timm
from typing import List, Optional


class TimmEncoder(nn.Module):
    """Universal encoder wrapper for timm models"""
    
    def __init__(self, name: str, pretrained: bool = True, depth: Optional[int] = None):
        super().__init__()
        
        # Create timm model with features_only to get intermediate features
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
        )
        
        # Get output channels for each stage
        self._out_channels = [
            info['num_chs'] for info in self.model.feature_info
        ]
        
        # Add input channels (3 for RGB) as stage 0
        self._out_channels = [3] + self._out_channels
        
        # Store actual depth (number of feature stages)
        self._depth = len(self._out_channels) - 1
        
    def forward(self, x):
        """Forward pass returning features at each stage"""
        features = self.model(x)
        return [x] + features  # Include input as stage 0
    
    @property
    def out_channels(self) -> List[int]:
        """Return output channels for each stage"""
        return self._out_channels
    
    def load_state_dict(self, state_dict, **kwargs):
        """Load pretrained weights"""
        self.model.load_state_dict(state_dict, **kwargs)


def create_timm_encoder(name: str, pretrained: bool = True):
    """
    Factory function to create timm encoder
    
    Args:
        name: Model name from timm (e.g., 'convnext_tiny', 'swin_tiny_patch4_window7_224')
        pretrained: Load ImageNet pretrained weights
        
    Returns:
        TimmEncoder instance
    """
    return TimmEncoder(name=name, pretrained=pretrained)


# Test if model works
if __name__ == "__main__":
    import torch
    
    # Test ConvNeXt
    print("Testing ConvNeXt Tiny...")
    encoder = create_timm_encoder('convnext_tiny', pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    features = encoder(x)
    print(f"  Input: {x.shape}")
    for i, feat in enumerate(features):
        print(f"  Stage {i}: {feat.shape} - {encoder.out_channels[i]} channels")
    print(f"  Total stages: {encoder._depth}")
    print(f"  ✅ ConvNeXt works!\n")
    
    # Test Swin
    print("Testing Swin Tiny...")
    encoder = create_timm_encoder('swin_tiny_patch4_window7_224', pretrained=False)
    features = encoder(x)
    print(f"  Input: {x.shape}")
    for i, feat in enumerate(features):
        print(f"  Stage {i}: {feat.shape} - {encoder.out_channels[i]} channels")
    print(f"  Total stages: {encoder._depth}")
    print(f"  ✅ Swin works!\n")
    
    # Test MobileViT
    print("Testing MobileViT v2-200...")
    encoder = create_timm_encoder('mobilevitv2_200', pretrained=False)
    x_256 = torch.randn(2, 3, 256, 256)  # MobileViT works better with 256px
    features = encoder(x_256)
    print(f"  Input: {x_256.shape}")
    for i, feat in enumerate(features):
        print(f"  Stage {i}: {feat.shape} - {encoder.out_channels[i]} channels")
    print(f"  Total stages: {encoder._depth}")
    print(f"  ✅ MobileViT works!\n")