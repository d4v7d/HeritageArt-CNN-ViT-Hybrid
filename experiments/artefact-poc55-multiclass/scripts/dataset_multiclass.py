"""
Multiclass hierarchical dataset loader for POC-5.5
Simplified version without sklearn dependency
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ArtefactMulticlassDataset(Dataset):
    """ARTeFACT Dataset for hierarchical multiclass segmentation."""
    
    # Excluded images (too large, cause OOM/performance issues)
    EXCLUDED_IMAGES = {
        'cljmrkz5o342f07clh6hz82sk.png',  # 187 MB, 12288x10860 px (127x larger than avg)
    }
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[A.Compose] = None,
        ignore_index: int = 255
    ):
        # Filter out excluded images
        filtered_pairs = [
            (img, mask) for img, mask in zip(image_paths, mask_paths)
            if Path(img).name not in self.EXCLUDED_IMAGES
        ]
        
        if len(filtered_pairs) < len(image_paths):
            excluded_count = len(image_paths) - len(filtered_pairs)
            print(f"⚠️  Excluded {excluded_count} oversized image(s) from dataset")
        
        self.image_paths = [p[0] for p in filtered_pairs]
        self.mask_paths = [p[1] for p in filtered_pairs]
        self.transform = transform
        self.ignore_index = ignore_index
        
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch between images and masks"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def fine_to_binary(self, mask):
        """Convert fine labels to binary: Clean (0) vs Damage (1)"""
        binary = np.zeros_like(mask, dtype=np.uint8)
        binary[mask == 0] = 0  # Clean
        binary[(mask >= 1) & (mask <= 15)] = 1  # All damage
        binary[mask == self.ignore_index] = self.ignore_index
        return binary
    
    def fine_to_coarse(self, mask):
        """Convert fine labels to coarse (4 damage groups)."""
        coarse = np.full_like(mask, self.ignore_index, dtype=np.uint8)
        
        # Group 0: Structural (1-4)
        coarse[(mask >= 1) & (mask <= 4)] = 0
        # Group 1: Surface (5,6,10,11)
        coarse[np.isin(mask, [5, 6, 10, 11])] = 1
        # Group 2: Color (7,9,13)
        coarse[np.isin(mask, [7, 9, 13])] = 2
        # Group 3: Optical (8,12,14,15)
        coarse[np.isin(mask, [8, 12, 14, 15])] = 3
        
        return coarse
    
    def __getitem__(self, idx: int):
        # Load image and mask
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask_fine = np.array(Image.open(self.mask_paths[idx]))
        
        # Generate hierarchical labels
        mask_binary = self.fine_to_binary(mask_fine)
        mask_coarse = self.fine_to_coarse(mask_fine)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(
                image=image,
                mask=mask_fine,
                mask1=mask_binary,
                mask2=mask_coarse
            )
            image = augmented['image']
            mask_fine = (augmented['mask'] * 255).long().clamp(0, 15)
            mask_binary = augmented['mask1'].long().clamp(0, 1)
            mask_coarse = augmented['mask2'].long().clamp(0, 3)
        
        return image, {
            'fine': mask_fine,
            'binary': mask_binary,
            'coarse': mask_coarse
        }


def get_multiclass_transforms(config: Dict, mode: str = 'train') -> A.Compose:
    """
    Get augmentation transforms.
    
    Implements Medium Augmentation (5-7x effective dataset) for Tiny models.
    Based on DATA-AUGMENTATION-STRATEGY.md recommendations.
    """
    
    img_size = config['data'].get('image_size', 256)
    
    if mode == 'train':
        # Medium Augmentation: Balanced diversity + realism
        transforms_list = [
            # Geometric (heritage-safe)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
            A.Rotate(limit=30, p=0.6, border_mode=0),
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(0.7, 1.0),      # 70-100% crop
                ratio=(0.85, 1.15),    # Slight aspect distortion
                p=0.8
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,       # 10% translation
                scale_limit=0.15,      # ±15% zoom
                rotate_limit=20,
                border_mode=0,
                p=0.5
            ),
            
            # Photometric (lighting/color variations)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # ±20% brightness
                contrast_limit=0.2,    # ±20% contrast
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05,
                p=0.4
            ),
            
            # Advanced (texture preservation)
            A.OneOf([
                A.ElasticTransform(alpha=50, sigma=5, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.1, border_mode=0, p=1.0),
                A.OpticalDistortion(distort_limit=0.1, border_mode=0, p=1.0),
            ], p=0.3),
            
            # Noise & Blur (simulate scanning/aging)
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=1.0),
            ], p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Heritage-Specific Augmentations (domain adaptation)
            A.OneOf([
                # Aging simulation: yellowing, fading
                A.ToSepia(p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=0,
                    sat_shift_limit=(-30, 0),  # Desaturation only
                    val_shift_limit=(-15, 0),  # Darkening only
                    p=0.5
                ),
                # Vignetting (darker edges from archival photos)
                A.GaussNoise(var_limit=(10, 50), p=0.3),
            ], p=0.3),
            
            # Scanning artifacts
            A.OneOf([
                A.ImageCompression(
                    quality_lower=70,
                    quality_upper=95,
                    p=0.5
                ),
                A.GridDistortion(num_steps=10, distort_limit=0.05, p=0.2),
            ], p=0.25),
            
            # Lighting variation (museum/archive conditions)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.15,
                    p=0.6
                ),
                A.RGBShift(
                    r_shift_limit=15,  # Warm/cool lighting shifts
                    g_shift_limit=10,
                    b_shift_limit=15,
                    p=0.4
                ),
            ], p=0.4),
            
            # Ensure consistent size (CRITICAL - some transforms may change dimensions)
            A.Resize(height=img_size, width=img_size, interpolation=0),
            
            # Normalization and tensor conversion
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
    else:
        # Validation: resize and normalize only (NO augmentation)
        transforms_list = [
            A.Resize(height=img_size, width=img_size, interpolation=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
    
    return A.Compose(
        transforms_list, 
        additional_targets={'mask1': 'mask', 'mask2': 'mask'}
    )
