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
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[A.Compose] = None,
        ignore_index: int = 255
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.ignore_index = ignore_index
        
        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"
    
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
            mask_fine = augmented['mask'].long()
            mask_binary = augmented['mask1'].long()
            mask_coarse = augmented['mask2'].long()
        
        return image, {
            'fine': mask_fine,
            'binary': mask_binary,
            'coarse': mask_coarse
        }


def get_multiclass_transforms(config: Dict, mode: str = 'train') -> A.Compose:
    """Get augmentation transforms."""
    
    img_size = config['data'].get('image_size', 256)
    
    if mode == 'train':
        aug_config = config['augmentation']['train']
        transforms_list = [
            A.Resize(height=img_size, width=img_size, interpolation=1),
            A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
            A.VerticalFlip(p=aug_config.get('vertical_flip', 0.3)),
            A.RandomRotate90(p=aug_config.get('rotate_90', 0.3)),
        ]
        
        if aug_config.get('brightness_contrast', 0.0) > 0:
            transforms_list.append(
                A.RandomBrightnessContrast(p=aug_config['brightness_contrast'])
            )
        
        if aug_config.get('gaussian_noise', 0.0) > 0:
            transforms_list.append(
                A.GaussNoise(p=aug_config['gaussian_noise'])
            )
        
        # Normalization and tensor conversion
        transforms_list.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    else:
        # Validation: resize and normalize only
        transforms_list = [
            A.Resize(height=img_size, width=img_size, interpolation=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
    
    return A.Compose(transforms_list, additional_targets={'mask1': 'mask', 'mask2': 'mask'})
