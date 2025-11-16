"""
Model Factory for POC-5.8
Supports both SMP native encoders and custom Timm encoders
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from timm_encoder import create_timm_encoder


def create_model(config: dict, decoder_type='deeplabv3plus'):
    """
    Create segmentation model based on config
    
    Args:
        config: Model configuration dict with keys:
            - architecture: 'DeepLabV3Plus' or 'Unet'
            - encoder_name: encoder name (e.g., 'resnet50' or 'tu-convnext_tiny')
            - encoder_weights: 'imagenet' or None
            - in_channels: input channels (default 3)
            - classes: number of output classes
            - activation: output activation (None for logits)
        decoder_type: 'deeplabv3plus' or 'unet' (overrides config)
    
    Returns:
        Model instance (SMP or custom Timm wrapper)
    """
    model_config = config.get('model', config)
    
    encoder_name = model_config['encoder_name']
    encoder_weights = model_config.get('encoder_weights', 'imagenet')
    classes = model_config.get('classes', 16)
    activation = model_config.get('activation', None)
    architecture = model_config.get('architecture', 'DeepLabV3Plus')
    
    # Override with decoder_type param if provided
    if decoder_type:
        architecture = 'Unet' if decoder_type == 'unet' else 'DeepLabV3Plus'
    
    # Check if it's a timm encoder (prefix 'tu-' for timm-universal)
    if encoder_name.startswith('tu-'):
        print(f"Creating custom Timm {architecture} with encoder: {encoder_name}")
        
        # Remove 'tu-' prefix to get actual timm model name
        timm_name = encoder_name[3:]
        
        # Create custom model with timm encoder
        pretrained = encoder_weights == 'imagenet'
        encoder = create_timm_encoder(timm_name, pretrained=pretrained)
        
        # Build model with custom encoder
        if architecture == 'Unet' or decoder_type == 'unet':
            model = TimmUnet(
                encoder=encoder,
                classes=classes,
                activation=activation
            )
        else:
            model = TimmDeepLabV3Plus(
                encoder=encoder,
                classes=classes,
                activation=activation
            )
        
        print(f"  Encoder channels: {encoder.out_channels}")
        print(f"  Model created successfully")
        
    else:
        # Use standard SMP
        print(f"Creating SMP {architecture} with encoder: {encoder_name}")
        
        if architecture == 'Unet' or decoder_type == 'unet':
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=classes,
                activation=activation
            )
        else:
            model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=classes,
                activation=activation
            )
    
    return model


class TimmUnet(nn.Module):
    """U-Net with custom Timm encoder"""
    
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


class TimmDeepLabV3Plus(nn.Module):
    """DeepLabV3+ with custom Timm encoder"""
    
    def __init__(
        self,
        encoder,
        classes: int = 16,
        activation: str = None
    ):
        super().__init__()
        
        self.encoder = encoder
        encoder_channels = encoder.out_channels
        
        # DeepLabV3+ expects exactly 6 encoder_channels: [input, s0, s1, s2, s3, s4]
        # Some encoders have only 5 (4 stages): [input, s0, s1, s2, s3]
        # We need to pad to 6 by duplicating the last stage
        
        if len(encoder_channels) == 5:  # ConvNeXt, Swin (4 stages)
            # Duplicate last stage: [3, 96, 192, 384, 768] -> [3, 96, 192, 384, 768, 768]
            encoder_channels_padded = encoder_channels + [encoder_channels[-1]]
            encoder_depth = 4  # Original 4 stages
        elif len(encoder_channels) == 6:  # ResNet, MobileViT (5 stages)
            encoder_channels_padded = encoder_channels
            encoder_depth = 5
        else:
            raise ValueError(f"Unexpected encoder channels length: {len(encoder_channels)}")
        
        # Import decoder
        from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
        
        # Create decoder with padded channels
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=encoder_channels_padded,
            encoder_depth=encoder_depth,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
            aspp_separable=True,
            aspp_dropout=0.5,
        )
        
        # Segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=256,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
        )
        
        self.name = f'timm-deeplabv3plus-{encoder.name}'
        
    def forward(self, x):
        """Forward pass"""
        # Get encoder features: [input, stage0, stage1, stage2, stage3, (stage4)]
        features = self.encoder(x)
        
        # Pad to 6 features if needed (for 4-stage encoders like ConvNeXt/Swin)
        if len(features) == 5:
            # Duplicate last feature: [x, s0, s1, s2, s3] -> [x, s0, s1, s2, s3, s3]
            features = features + [features[-1]]
        
        # Pass to decoder (expects single list argument)
        decoder_output = self.decoder(features)
        
        # Segmentation head
        masks = self.segmentation_head(decoder_output)
        
        return masks


# Test script
if __name__ == "__main__":
    import torch
    
    print("Testing Timm Encoder Wrapper\n" + "="*50)
    
    print("\n### Testing with DeepLabV3+ ###")
    models_to_test_deeplabv3 = [
        ('convnext_tiny', 384),
        ('swin_tiny_patch4_window7_224', 224),
    ]
    
    for model_name, img_size in models_to_test_deeplabv3:
        print(f"\nTesting {model_name} @ {img_size}px (DeepLabV3+)...")
        
        try:
            config = {
                'model': {
                    'architecture': 'DeepLabV3Plus',
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
            
            print(f"  ✅ Success!")
            print(f"     Input: {tuple(x.shape)}")
            print(f"     Output: {tuple(output.shape)}")
            print(f"     Parameters: {params:.1f}M")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}")
    
    print("\n" + "="*50)
    print("\n### Testing with UNet ###")
    models_to_test_unet = [
        ('mobilevitv2_200.cvnets_in1k', 256),
        ('coatnet_0_rw_224.sw_in1k', 224),
        ('maxvit_tiny_tf_224.in1k', 224),
    ]
    
    for model_name, img_size in models_to_test_unet:
        print(f"\nTesting {model_name} @ {img_size}px (UNet)...")
        
        try:
            # Create config
            config = {
                'model': {
                    'architecture': 'Unet',
                    'encoder_name': f'tu-{model_name}',
                    'encoder_weights': None,  # None = no pretrained weights
                    'classes': 16,
                    'activation': None
                }
            }
            
            # Create model with UNet
            model = create_model(config, decoder_type='unet')
            model.eval()
            
            # Test forward pass
            x = torch.randn(2, 3, img_size, img_size)
            with torch.no_grad():
                output = model(x)
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters()) / 1e6
            
            print(f"  ✅ Success!")
            print(f"     Input: {tuple(x.shape)}")
            print(f"     Output: {tuple(output.shape)}")
            print(f"     Parameters: {params:.1f}M")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}")
    
    # Also test standard SMP
    print("\n" + "="*50)
    print("\n### Testing SMP Native ###")
    print(f"\nTesting resnet50 (SMP DeepLabV3+)...")
    try:
        config = {
            'model': {
                'encoder_name': 'resnet50',
                'encoder_weights': None,
                'classes': 16,
            }
        }
        model = create_model(config)
        model.eval()
        x = torch.randn(2, 3, 384, 384)
        with torch.no_grad():
            output = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✅ Success!")
        print(f"     Input: {tuple(x.shape)}")
        print(f"     Output: {tuple(output.shape)}")
        print(f"     Parameters: {params:.1f}M")
    except Exception as e:
        print(f"  ❌ Error: {e}")
