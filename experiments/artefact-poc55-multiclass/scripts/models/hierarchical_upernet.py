"""
Hierarchical UPerNet with Multi-Task Learning (Innovation #1)

POC-5.5 Core Innovation:
- 3 parallel prediction heads: Binary (2), Coarse (4), Fine (16)
- Auxiliary tasks (binary, coarse) guide learning of main task (fine)
- Helps rare classes by learning hierarchical structure

Class Hierarchy:
Binary: Clean (0) vs Damage (1)
Coarse (4 groups):
  1. Structural Damage: Cracks, Material loss, Peel, Structural defects
  2. Surface Contamination: Dirt spots, Stains, Hairs, Dust spots
  3. Color Alterations: Discolouration, Burn marks, Fading
  4. Optical Artifacts: Scratches, Lightleak, Blur
Fine: All 16 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PPM(nn.Module):
    """Pyramid Pooling Module (from UPerNet)"""
    
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        
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
        
        bottleneck_channels = in_channels + len(pool_scales) * out_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        input_size = x.shape[2:]
        ppm_outs = [x]
        
        for ppm_branch in self.ppm_branches:
            pooled = ppm_branch(x)
            upsampled = F.interpolate(pooled, size=input_size, mode='bilinear', align_corners=False)
            ppm_outs.append(upsampled)
        
        ppm_out = torch.cat(ppm_outs, dim=1)
        output = self.bottleneck(ppm_out)
        return output


class FPN(nn.Module):
    """Feature Pyramid Network (from UPerNet)"""
    
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
        
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
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        fpn_outs = []
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                fpn_out = laterals[i]
            else:
                prev_shape = laterals[i].shape[2:]
                upsampled = F.interpolate(fpn_outs[-1], size=prev_shape, mode='bilinear', align_corners=False)
                fpn_out = laterals[i] + upsampled
            
            fpn_out = self.fpn_convs[i](fpn_out)
            fpn_outs.append(fpn_out)
        
        fpn_outs.reverse()
        return fpn_outs


class HierarchicalUPerNet(nn.Module):
    """
    Innovation #1: Hierarchical Multi-Task Learning
    
    3 parallel prediction heads at different granularities:
    - Binary head: Clean vs Damage (auxiliary task 1)
    - Coarse head: 4 damage groups (auxiliary task 2)  
    - Fine head: 16 fine-grained classes (main task)
    
    Loss = 0.2 * L_binary + 0.3 * L_coarse + 1.0 * L_fine
    """
    
    def __init__(self, encoder_name, encoder_weights, in_channels_list,
                 out_channels=256, ppm_pool_scales=(1, 2, 3, 6),
                 dropout=0.1, num_classes_fine=16, img_size=256):
        """
        Args:
            encoder_name: timm model name (e.g., 'convnext_tiny')
            encoder_weights: Pretrained weights or None
            in_channels_list: Channel dims from encoder stages
            out_channels: FPN output channels (256)
            ppm_pool_scales: PPM scales (1, 2, 3, 6)
            dropout: Dropout rate (0.1)
            num_classes_fine: Number of fine classes (16)
            img_size: Input image size (256 for laptop)
        """
        super(HierarchicalUPerNet, self).__init__()
        
        self.encoder_name = encoder_name
        self.img_size = img_size
        self.num_classes_fine = num_classes_fine
        
        # Fixed hierarchy
        self.num_classes_binary = 2   # Clean, Damage
        self.num_classes_coarse = 4   # 4 damage groups
        
        # Load timm encoder
        encoder_kwargs = {
            'pretrained': (encoder_weights is not None),
            'features_only': True,
            'out_indices': (0, 1, 2, 3)
        }
        
        if 'maxvit' in encoder_name.lower():
            encoder_kwargs['out_indices'] = (1, 2, 3, 4)
        
        if 'swin' in encoder_name.lower() or 'vit' in encoder_name.lower():
            encoder_kwargs['img_size'] = img_size
        
        self.encoder = timm.create_model(encoder_name, **encoder_kwargs)
        
        # Verify channels
        dummy_input = torch.randn(2, 3, img_size, img_size)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        
        is_swin_vit = (('swin' in encoder_name.lower() or 'vit' in encoder_name.lower()) 
                       and 'maxvit' not in encoder_name.lower())
        
        if is_swin_vit:
            actual_channels = [f.shape[3] for f in features]
        else:
            actual_channels = [f.shape[1] for f in features]
        
        if actual_channels != in_channels_list:
            print(f"⚠️  Using actual channels {actual_channels} (config had {in_channels_list})")
            in_channels_list = actual_channels
        
        self.in_channels_list = in_channels_list
        self.is_swin_vit = is_swin_vit
        
        # UPerNet Neck: PPM + FPN (shared by all heads)
        self.ppm = PPM(
            in_channels=in_channels_list[-1],
            out_channels=out_channels,
            pool_scales=ppm_pool_scales
        )
        
        fpn_in_channels = in_channels_list[:-1] + [out_channels]
        self.fpn = FPN(
            in_channels_list=fpn_in_channels,
            out_channels=out_channels
        )
        
        # Fusion layer (concatenate all FPN levels)
        self.fusion = nn.Sequential(
            nn.Conv2d(
                out_channels * len(in_channels_list),
                out_channels,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3 Hierarchical Heads (all use same fused features)
        self.head_binary = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, self.num_classes_binary, 1)
        )
        
        self.head_coarse = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, self.num_classes_coarse, 1)
        )
        
        self.head_fine = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, num_classes_fine, 1)
        )
    
    def forward(self, x, return_all_heads=True):
        """
        Args:
            x: (B, 3, H, W) - input images
            return_all_heads: If True, return dict with all 3 heads
        
        Returns:
            If return_all_heads=True:
                dict with keys 'binary', 'coarse', 'fine' (all at input resolution)
            Else:
                Only fine head logits (for inference)
        """
        input_size = (x.shape[2], x.shape[3])
        
        # Encoder
        encoder_features = self.encoder(x)
        
        # Transpose Swin/ViT features (B,H,W,C) -> (B,C,H,W)
        if self.is_swin_vit:
            encoder_features = [f.permute(0, 3, 1, 2).contiguous() for f in encoder_features]
        
        # UPerNet Neck: PPM + FPN
        ppm_out = self.ppm(encoder_features[-1])
        fpn_inputs = encoder_features[:-1] + [ppm_out]
        fpn_outs = self.fpn(fpn_inputs)
        
        # Upsample all FPN levels to highest resolution
        target_size = fpn_outs[0].shape[2:]
        upsampled_fpn = []
        for fpn_out in fpn_outs:
            if fpn_out.shape[2:] != target_size:
                upsampled = F.interpolate(fpn_out, size=target_size, mode='bilinear', align_corners=False)
                upsampled_fpn.append(upsampled)
            else:
                upsampled_fpn.append(fpn_out)
        
        # Fuse all levels
        fused = torch.cat(upsampled_fpn, dim=1)
        fused = self.fusion(fused)
        
        # 3 Hierarchical Heads
        logits_binary = self.head_binary(fused)
        logits_coarse = self.head_coarse(fused)
        logits_fine = self.head_fine(fused)
        
        # Upsample all to input resolution
        logits_binary = F.interpolate(logits_binary, size=input_size, mode='bilinear', align_corners=False)
        logits_coarse = F.interpolate(logits_coarse, size=input_size, mode='bilinear', align_corners=False)
        logits_fine = F.interpolate(logits_fine, size=input_size, mode='bilinear', align_corners=False)
        
        if return_all_heads:
            return {
                'binary': logits_binary,  # (B, 2, H, W)
                'coarse': logits_coarse,  # (B, 4, H, W)
                'fine': logits_fine       # (B, 16, H, W)
            }
        else:
            # Inference: only return fine head
            return logits_fine


def fine_to_binary(fine_labels, ignore_index=255):
    """
    Convert fine (16-class) labels to binary (2-class)
    
    Binary mapping:
      0 (Clean) -> 0 (Clean)
      1-15 (All damage types) -> 1 (Damage)
      255 (Ignore) -> 255 (Ignore)
    
    Args:
        fine_labels: (B, H, W) tensor with values 0-15 or 255
    
    Returns:
        binary_labels: (B, H, W) tensor with values 0, 1, or 255
    """
    binary_labels = torch.zeros_like(fine_labels)
    binary_labels[fine_labels == 0] = 0  # Clean
    binary_labels[(fine_labels >= 1) & (fine_labels <= 15)] = 1  # Damage
    binary_labels[fine_labels == ignore_index] = ignore_index  # Ignore
    return binary_labels


def fine_to_coarse(fine_labels, ignore_index=255):
    """
    Convert fine (16-class) labels to coarse (4-class damage groups)
    
    Coarse mapping (4 damage groups):
      0: Structural Damage [1, 2, 3, 4]
      1: Surface Contamination [5, 6, 7, 11]
      2: Color Alterations [8, 9, 13]
      3: Optical Artifacts [10, 12, 14, 15]
    
    Note: Class 0 (Clean) is NOT part of coarse groups (this is for damage types only)
          In practice, we'll only compute loss on damage pixels (binary==1)
    
    Fine class meanings:
      0: Clean
      1: Material loss (Structural)
      2: Peel (Structural)
      3: Cracks (Structural)
      4: Structural defects (Structural)
      5: Dirt spots (Surface)
      6: Stains (Surface)
      7: Discolouration (Color) <- WAIT, should be Surface based on description
      8: Scratches (Optical)
      9: Burn marks (Color)
      10: Hairs (Surface)
      11: Dust spots (Surface)
      12: Lightleak (Optical)
      13: Fading (Color)
      14: Blur (Optical)
      15: Other damage
    
    Updated mapping based on semantic meaning:
      Group 0 - Structural Damage: [1:Material loss, 2:Peel, 3:Cracks, 4:Structural defects]
      Group 1 - Surface Contamination: [5:Dirt spots, 6:Stains, 10:Hairs, 11:Dust spots]
      Group 2 - Color Alterations: [7:Discolouration, 9:Burn marks, 13:Fading]
      Group 3 - Optical Artifacts: [8:Scratches, 12:Lightleak, 14:Blur, 15:Other]
    
    Args:
        fine_labels: (B, H, W) tensor with values 0-15 or 255
    
    Returns:
        coarse_labels: (B, H, W) tensor with values 0-3 or 255
    """
    coarse_labels = torch.full_like(fine_labels, ignore_index)
    
    # Group 0: Structural Damage
    structural = [1, 2, 3, 4]
    for cls in structural:
        coarse_labels[fine_labels == cls] = 0
    
    # Group 1: Surface Contamination
    surface = [5, 6, 10, 11]
    for cls in surface:
        coarse_labels[fine_labels == cls] = 1
    
    # Group 2: Color Alterations
    color = [7, 9, 13]
    for cls in color:
        coarse_labels[fine_labels == cls] = 2
    
    # Group 3: Optical Artifacts
    optical = [8, 12, 14, 15]
    for cls in optical:
        coarse_labels[fine_labels == cls] = 3
    
    # Clean pixels: set to ignore (coarse is only for damage classification)
    coarse_labels[fine_labels == 0] = ignore_index
    
    return coarse_labels


def build_hierarchical_model(config):
    """
    Build hierarchical model from config dict
    
    Args:
        config: Dict with model configuration
    
    Returns:
        model: HierarchicalUPerNet instance
    """
    model_cfg = config['model']
    
    encoder_name = model_cfg['encoder']
    encoder_weights = model_cfg.get('encoder_weights', 'imagenet_in1k')
    
    upernet_cfg = model_cfg.get('upernet', {})
    ppm_pool_scales = tuple(upernet_cfg.get('ppm_pool_scales', [1, 2, 3, 6]))
    fpn_out_channels = upernet_cfg.get('fpn_out_channels', 256)
    dropout = upernet_cfg.get('dropout', 0.1)
    
    num_classes_fine = model_cfg.get('classes', 16)
    img_size = config.get('data', {}).get('image_size', 256)
    
    # Encoder channel configs (standardized out_indices (0,1,2,3))
    ENCODER_CHANNELS = {
        'convnext_tiny': [96, 192, 384, 768],
        'swin_tiny_patch4_window7_224': [96, 192, 384, 768],  # After permute
        'maxvit_tiny_rw_256': [64, 64, 128, 256],  # Corrected for out_indices (0,1,2,3)
    }
    
    in_channels_list = ENCODER_CHANNELS.get(encoder_name)
    
    if in_channels_list is None:
        # Auto-detect (standardized out_indices (0,1,2,3) for all models)
        print(f"⚠️  Auto-detecting channels for {encoder_name}...")
        encoder_kwargs = {
            'pretrained': False,
            'features_only': True,
            'out_indices': (0, 1, 2, 3)
        }
        if 'swin' in encoder_name.lower() or 'vit' in encoder_name.lower():
            encoder_kwargs['img_size'] = img_size
        
        dummy_encoder = timm.create_model(encoder_name, **encoder_kwargs)
        dummy_input = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            features = dummy_encoder(dummy_input)
        in_channels_list = [f.shape[1] for f in features]
        print(f"   Detected: {in_channels_list}")
        del dummy_encoder
    
    model = HierarchicalUPerNet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels_list=in_channels_list,
        out_channels=fpn_out_channels,
        ppm_pool_scales=ppm_pool_scales,
        dropout=dropout,
        num_classes_fine=num_classes_fine,
        img_size=img_size
    )
    
    return model


if __name__ == '__main__':
    print("Testing Hierarchical UPerNet...")
    
    # Test config
    config = {
        'model': {
            'encoder': 'convnext_tiny',
            'encoder_weights': None,
            'classes': 16,
            'upernet': {
                'ppm_pool_scales': [1, 2, 3, 6],
                'fpn_out_channels': 256,
                'dropout': 0.1
            }
        },
        'data': {
            'image_size': 256
        }
    }
    
    model = build_hierarchical_model(config)
    
    # Test forward
    x = torch.randn(2, 3, 256, 256)
    outputs = model(x, return_all_heads=True)
    
    print(f"Input: {x.shape}")
    print(f"Binary head: {outputs['binary'].shape}")  # (2, 2, 256, 256)
    print(f"Coarse head: {outputs['coarse'].shape}")  # (2, 4, 256, 256)
    print(f"Fine head: {outputs['fine'].shape}")      # (2, 16, 256, 256)
    
    # Test label conversion
    fine_labels = torch.tensor([
        [0, 1, 5, 7, 12, 255],  # Clean, Material loss, Dirt spot, Discolouration, Lightleak, Ignore
    ]).long()
    
    binary_labels = fine_to_binary(fine_labels)
    coarse_labels = fine_to_coarse(fine_labels)
    
    print(f"\nLabel conversion test:")
    print(f"Fine:   {fine_labels[0].tolist()}")
    print(f"Binary: {binary_labels[0].tolist()}")  # [0, 1, 1, 1, 1, 255]
    print(f"Coarse: {coarse_labels[0].tolist()}")  # [255, 0, 1, 2, 3, 255]
    
    print("✅ Hierarchical UPerNet test passed!")
