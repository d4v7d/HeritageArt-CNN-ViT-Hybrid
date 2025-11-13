"""
Custom Model Factory for Timm Encoders + SMP Decoders
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from timm_encoder import create_timm_encoder


class TimmDeepLabV3Plus(nn.Module):
    """DeepLabV3+ with custom Timm encoder"""
    
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 16,
        activation: str = None
    ):
        super().__init__()
        
        # Create encoder
        pretrained = encoder_weights == 'imagenet'
        self.encoder = create_timm_encoder(encoder_name, pretrained=pretrained)
        
        # Create DeepLabV3+ decoder
        # Use encoder output channels for decoder
        encoder_channels = self.encoder.out_channels
        
        # DeepLabV3+ expects 5 encoder stages
        # If encoder has less, pad with zeros
        if len(encoder_channels) < 6:  # 6 = input + 5 stages
            encoder_channels = encoder_channels + [0] * (6 - len(encoder_channels))
        
        self.decoder = smp.decoders.deeplabv3.DeepLabV3PlusDecoder(
            encoder_channels=encoder_channels[:6],  # Use first 6
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
        )
        
        # Segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
        )
        
    def forward(self, x):
        """Forward pass"""
        # Encoder
        features = self.encoder(x)
        
        # Fix Swin output format (B, H, W, C) -> (B, C, H, W)
        features_fixed = []
        for feat in features:
            if len(feat.shape) == 4 and feat.shape[1] != feat.shape[3]:
                # Swin format: (B, H, W, C)
                if feat.shape[-1] in self.encoder.out_channels:
                    feat = feat.permute(0, 3, 1, 2)  # -> (B, C, H, W)
            features_fixed.append(feat)
        
        # Decoder
        decoder_output = self.decoder(*features_fixed)
        
        # Segmentation head
        masks = self.segmentation_head(decoder_output)
        
        return masks


def create_model(config: dict):
    """
    Factory function to create model based on config
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        Model instance
    """
    encoder_name = config['encoder_name']
    
    # Check if it's a timm encoder (custom models)
    timm_models = ['convnext', 'swin', 'mobilevit']
    is_timm = any(model in encoder_name.lower() for model in timm_models)
    
    if is_timm:
        # Use custom wrapper
        model = TimmDeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            in_channels=config.get('in_channels', 3),
            classes=config.get('classes', 16),
            activation=config.get('activation', None)
        )
    else:
        # Use standard SMP
        architecture = config.get('architecture', 'DeepLabV3Plus')
        model_class = getattr(smp, architecture)
        model = model_class(
            encoder_name=encoder_name,
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            in_channels=config.get('in_channels', 3),
            classes=config.get('classes', 16),
            activation=config.get('activation', None)
        )
    
    return model


if __name__ == "__main__":
    # Test model creation
    configs = [
        {'encoder_name': 'convnext_tiny', 'classes': 16},
        {'encoder_name': 'swin_tiny_patch4_window7_224', 'classes': 16},
        {'encoder_name': 'mobilevitv2_200', 'classes': 16},
        {'encoder_name': 'resnet50', 'classes': 16},  # Standard SMP
    ]
    
    for cfg in configs:
        print(f"\nTesting {cfg['encoder_name']}...")
        model = create_model(cfg)
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Input: {x.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Params: {params:.1f}M")
        print(f"  ✅ Works!")