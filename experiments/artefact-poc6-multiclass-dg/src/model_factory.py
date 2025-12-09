"""
Model Factory for POC-6
Supports:
- Standard U-Net (POC-5.8 legacy)
- Hierarchical UPerNet (POC-6 Innovation #1)
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from src.timm_encoder import create_timm_encoder
from src.models.hierarchical_upernet import build_hierarchical_model


def create_model(config: dict):
    """
    Create segmentation model (Standard U-Net or Hierarchical UPerNet)
    
    Args:
        config: Model configuration dict
    
    Returns:
        Model instance
    """
    model_config = config.get('model', config)
    
    # Check for hierarchical model (POC-6)
    if model_config.get('hierarchical', False):
        # Ensure 'encoder' key exists (map from 'encoder_name' if needed)
        if 'encoder' not in model_config and 'encoder_name' in model_config:
            model_config['encoder'] = model_config['encoder_name'].replace('tu-', '')
            
        print(f"Creating Hierarchical UPerNet with encoder: {model_config['encoder']}")
        return build_hierarchical_model(config)
    
    encoder_name = model_config['encoder_name']
    encoder_weights = model_config.get('encoder_weights', 'imagenet')
    classes = model_config.get('classes', 16)
    activation = model_config.get('activation', None)
    
    # Check if it's a timm encoder (prefix 'tu-' for timm-universal)
    if encoder_name.startswith('tu-'):
        print(f"Creating custom Timm Unet with encoder: {encoder_name}")
        
        # Remove 'tu-' prefix to get actual timm model name
        timm_name = encoder_name[3:]
        
        # Create custom model with timm encoder
        pretrained = encoder_weights == 'imagenet'
        encoder = create_timm_encoder(timm_name, pretrained=pretrained)
        
        # Build U-Net with custom encoder
        model = TimmUnet(
            encoder=encoder,
            classes=classes,
            activation=activation
        )
        
        print(f"  Encoder channels: {encoder.out_channels}")
        print(f"  Model created successfully")
        
    else:
        # Use standard SMP U-Net
        print(f"Creating SMP Unet with encoder: {encoder_name}")
        
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation
        )
    
    return model


class TimmUnet(nn.Module):
    """U-Net with custom Timm encoder for fair architecture comparison"""
    
    def __init__(
        self,
        encoder,
        classes: int = 16,
        activation: str = None
    ):
        super().__init__()
        
        self.encoder = encoder
        encoder_channels = list(encoder.out_channels)
        
        # UNet expects 5 decoder blocks (encoder_depth=5)
        # If encoder has only 4 stages, pad with last channel
        if len(encoder_channels) == 5:  # 4 stages: [input, s0, s1, s2, s3]
            # Pad: [input, s0, s1, s2, s3] -> [input, s0, s1, s2, s3, s3]
            encoder_channels = encoder_channels + [encoder_channels[-1]]
        elif len(encoder_channels) == 6:  # 5 stages: already correct
            pass
        else:
            raise ValueError(f"Unexpected encoder channels: {len(encoder_channels)}")
        
        # Import UNet decoder
        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
        
        # Create decoder - UNet is more flexible with channels
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            attention_type=None,
        )
        
        # Segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=1,
        )
        
        self.name = f'timm-unet-{encoder.name}'
        
    def forward(self, x):
        """Forward pass"""
        # Get encoder features
        features = self.encoder(x)
        
        # Pad to 6 if needed (4-stage encoders)
        if len(features) == 5:
            features = features + [features[-1]]
        
        # Pass to decoder (UNet expects list)
        decoder_output = self.decoder(features)
        
        # Segmentation head
        masks = self.segmentation_head(decoder_output)
        
        return masks


# Test script - UNet only for fair comparison
if __name__ == "__main__":
    import torch
    
    print("Testing U-Net with Custom Timm Encoders\n" + "="*50)
    
    # POC-5.8 target models: CNN, ViT, Hybrid
    models_to_test = [
        ('convnext_tiny', 384, 'CNN'),
        ('swin_tiny_patch4_window7_224', 224, 'ViT'),
        ('coatnet_0_rw_224.sw_in1k', 224, 'Hybrid'),
    ]
    
    for model_name, img_size, arch_type in models_to_test:
        print(f"\nTesting {model_name} @ {img_size}px [{arch_type}]...")
        
        try:
            config = {
                'model': {
                    'architecture': 'Unet',
                    'encoder_name': f'tu-{model_name}',
                    'encoder_weights': None,
                    'classes': 16,
                    'activation': None
                }
            }
            
            model = create_model(config)
            model.eval()
            
            x = torch.randn(2, 3, img_size, img_size)
            with torch.no_grad():
                output = model(x)
            
            params = sum(p.numel() for p in model.parameters()) / 1e6
            
            # Verify output matches input size
            match = output.shape[-2:] == x.shape[-2:]
            status = "✅ MATCH!" if match else "❌ MISMATCH!"
            
            print(f"  {status}")
            print(f"     Input: {tuple(x.shape)}")
            print(f"     Output: {tuple(output.shape)}")
            print(f"     Parameters: {params:.1f}M")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}")
