"""
Model Factory for Multi-Backbone UPerNet

Builds segmentation models by combining:
- timm encoders: ConvNeXt, Swin, CoaT
- Custom UPerNet decoder
"""

import torch
import torch.nn as nn
import timm

# Handle both package import and direct execution
try:
    from .upernet_custom import UPerNetDecoder
except ImportError:
    from upernet_custom import UPerNetDecoder


class UPerNetModel(nn.Module):
    """Complete segmentation model: timm encoder + UPerNet decoder"""
    
    def __init__(self, encoder_name, encoder_weights, in_channels_list, 
                 out_channels=256, ppm_pool_scales=(1, 2, 3, 6), 
                 dropout=0.1, num_classes=2, img_size=512):
        """
        Args:
            encoder_name: timm model name (e.g., 'convnext_tiny')
            encoder_weights: Pretrained weights (e.g., 'imagenet_in1k')
            in_channels_list: Channel dims from encoder stages
            out_channels: FPN output channels
            ppm_pool_scales: PPM scales
            dropout: Dropout rate
            num_classes: Number of classes
            img_size: Input image size (for transformers with fixed pos embeddings)
        """
        super(UPerNetModel, self).__init__()
        
        self.encoder_name = encoder_name
        self.img_size = img_size
        
        # Load timm encoder with feature extraction
        # For Swin/ViT models, we need to specify img_size to handle position embeddings
        encoder_kwargs = {
            'pretrained': (encoder_weights is not None),
            'features_only': True,  # Return intermediate features
            'out_indices': (0, 1, 2, 3)  # 4 stages (default)
        }
        
        # MaxViT has 5 stages, use indices 1-4 for better channel progression
        if 'maxvit' in encoder_name.lower():
            encoder_kwargs['out_indices'] = (1, 2, 3, 4)
        
        # Add img_size for transformers (Swin, ViT, etc.)
        if 'swin' in encoder_name.lower() or 'vit' in encoder_name.lower() or 'deit' in encoder_name.lower():
            encoder_kwargs['img_size'] = img_size
        
        self.encoder = timm.create_model(encoder_name, **encoder_kwargs)
        
        # Verify feature channels match config
        # Use batch_size=2 to avoid BatchNorm issues
        dummy_input = torch.randn(2, 3, img_size, img_size)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        
        # Handle different feature formats
        # Note: MaxViT contains "vit" in name but uses standard CNN format (B,C,H,W)
        is_swin_vit = (('swin' in encoder_name.lower() or 'vit' in encoder_name.lower()) 
                       and 'maxvit' not in encoder_name.lower())
        
        if is_swin_vit:
            # Swin/ViT returns (B, H, W, C), need to transpose
            actual_channels = [f.shape[3] for f in features]
        else:
            # ConvNeXt/MaxViT/CNN returns (B, C, H, W)
            actual_channels = [f.shape[1] for f in features]
        if actual_channels != in_channels_list:
            print(f"⚠️  Warning: Expected channels {in_channels_list}, got {actual_channels}")
            print(f"   Using actual channels from encoder.")
            in_channels_list = actual_channels
        
        # UPerNet decoder
        self.decoder = UPerNetDecoder(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            ppm_pool_scales=ppm_pool_scales,
            dropout=dropout,
            num_classes=num_classes
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - input images
        
        Returns:
            logits: (B, num_classes, H, W) - segmentation logits
        """
        # Get input size for final upsampling
        input_size = (x.shape[2], x.shape[3])
        
        # Encoder forward
        encoder_features = self.encoder(x)
        
        # Swin/ViT models return features in (B, H, W, C) format
        # Need to transpose to (B, C, H, W) for UPerNet
        # Note: MaxViT contains "vit" in name but uses standard CNN format
        is_swin_vit = (('swin' in self.encoder_name.lower() or 'vit' in self.encoder_name.lower()) 
                       and 'maxvit' not in self.encoder_name.lower())
        
        if is_swin_vit:
            encoder_features = [f.permute(0, 3, 1, 2).contiguous() for f in encoder_features]
        
        # Decoder forward with input_size for correct upsampling
        logits = self.decoder(encoder_features, input_size=input_size)
        
        return logits
    
    def get_encoder_features(self, x):
        """Extract encoder features (useful for visualization)"""
        return self.encoder(x)


def build_model(config):
    """Build model from config dict
    
    Args:
        config: Dict with model configuration
    
    Returns:
        model: UPerNetModel instance
    """
    model_cfg = config['model']
    
    # Get encoder config
    encoder_name = model_cfg['encoder']
    encoder_weights = model_cfg.get('encoder_weights', 'imagenet_in1k')
    
    # Get decoder config
    upernet_cfg = model_cfg.get('upernet', {})
    ppm_pool_scales = tuple(upernet_cfg.get('ppm_pool_scales', [1, 2, 3, 6]))
    fpn_out_channels = upernet_cfg.get('fpn_out_channels', 256)
    dropout = upernet_cfg.get('dropout', 0.1)
    
    num_classes = model_cfg.get('classes', 2)
    
    # Get image size from data config
    img_size = config.get('data', {}).get('image_size', 512)
    
    # Encoder-specific channel configurations
    ENCODER_CHANNELS = {
        'convnext_tiny': [96, 192, 384, 768],
        'swin_tiny_patch4_window7_224': [96, 192, 384, 768],
        'coat_lite_small': [64, 128, 320, 512],  # From CoaT paper (not used - features_only unsupported)
        'maxvit_tiny_tf_512': [64, 128, 256, 512],  # MaxViT-Tiny stages 1-4
    }
    
    # Get expected channels for encoder
    in_channels_list = ENCODER_CHANNELS.get(encoder_name)
    
    if in_channels_list is None:
        # Auto-detect channels (slower but works for any encoder)
        print(f"⚠️  Channel config not found for {encoder_name}, auto-detecting...")
        
        # For auto-detection, create encoder with img_size if needed
        encoder_kwargs = {
            'pretrained': False,
            'features_only': True,
            'out_indices': (0, 1, 2, 3)
        }
        
        # MaxViT special case
        if 'maxvit' in encoder_name.lower():
            encoder_kwargs['out_indices'] = (1, 2, 3, 4)
        
        if 'swin' in encoder_name.lower() or 'vit' in encoder_name.lower():
            encoder_kwargs['img_size'] = img_size
        
        dummy_encoder = timm.create_model(encoder_name, **encoder_kwargs)
        dummy_input = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            features = dummy_encoder(dummy_input)
        in_channels_list = [f.shape[1] for f in features]
        print(f"   Detected channels: {in_channels_list}")
        del dummy_encoder
    
    # Build model
    model = UPerNetModel(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels_list=in_channels_list,
        out_channels=fpn_out_channels,
        ppm_pool_scales=ppm_pool_scales,
        dropout=dropout,
        num_classes=num_classes,
        img_size=img_size
    )
    
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import yaml
    
    print("Testing Model Factory...")
    
    # Test ConvNeXt
    config = {
        'model': {
            'encoder': 'convnext_tiny',
            'encoder_weights': None,  # Don't download for test
            'classes': 2,
            'upernet': {
                'ppm_pool_scales': [1, 2, 3, 6],
                'fpn_out_channels': 256,
                'dropout': 0.1
            }
        }
    }
    
    model = build_model(config)
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters: {count_parameters(model) / 1e6:.2f}M")
    
    assert logits.shape == (2, 2, 512, 512), "Output shape mismatch!"
    print("✅ Model factory test passed!")
