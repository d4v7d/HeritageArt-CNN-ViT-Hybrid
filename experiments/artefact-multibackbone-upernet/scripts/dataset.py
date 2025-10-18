"""
ARTeFACT Dataset Loader for Binary Segmentation
Binary task: Damage (1) vs Clean (0)
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class ArtefactDataset(Dataset):
    """
    ARTeFACT Dataset for semantic segmentation.
    
    Binary mode: Collapse damage classes (1-15) → 1 (Damage), keep 0 (Clean)
    Ignore index: 255 (Background)
    """
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[A.Compose] = None,
        binary_mode: bool = True,
        ignore_index: int = 255
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.binary_mode = binary_mode
        self.ignore_index = ignore_index
        
        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]))
        
        # Convert mask to binary BEFORE transforms (work with numpy)
        if self.binary_mode:
            # 0 = Clean, 1-15 = Damage → 1, 255 = Ignore (keep as 255)
            binary_mask = np.zeros_like(mask, dtype=np.uint8)
            binary_mask[mask > 0] = 1  # All damage classes → 1
            binary_mask[mask == self.ignore_index] = self.ignore_index  # Keep ignore
            mask = binary_mask
        
        # Apply transforms (this will convert to tensor)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.long()


def get_transforms(config: Dict, mode: str = 'train') -> A.Compose:
    """Get augmentation transforms."""
    
    # Get image size from config (default 512x512)
    img_size = config['data'].get('image_size', 512)
    
    if mode == 'train':
        aug_config = config['augmentation']['train']
        transforms = [
            A.Resize(height=img_size, width=img_size, interpolation=1),  # Add resize first
            A.HorizontalFlip(p=aug_config['horizontal_flip']),
            A.VerticalFlip(p=aug_config['vertical_flip']),
            A.Rotate(limit=aug_config['rotate_limit'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness_limit'],
                contrast_limit=aug_config['contrast_limit'],
                p=0.5
            ),
            A.Blur(blur_limit=aug_config['blur_limit'], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
    else:
        # Validation/Test: resize and normalize
        transforms = [
            A.Resize(height=img_size, width=img_size, interpolation=1),  # Add resize first
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
    
    return A.Compose(transforms)


def prepare_dataloaders(
    config: Dict,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        batch_size: Override config batch size
        num_workers: Override config num workers
    
    Returns:
        train_loader, val_loader
    """
    
    # Get data paths
    data_root = Path(config['data']['root'])
    images_dir = data_root / 'images'
    masks_dir = data_root / 'annotations'
    
    # Get all image and mask paths
    image_files = sorted(images_dir.glob('*.png'))
    mask_files = sorted(masks_dir.glob('*.png'))
    
    assert len(image_files) > 0, f"No images found in {images_dir}"
    assert len(image_files) == len(mask_files), "Mismatch between images and masks"
    
    image_paths = [str(f) for f in image_files]
    mask_paths = [str(f) for f in mask_files]
    
    # Train/val split
    train_split = config['data']['train_val_split']
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths,
        train_size=train_split,
        random_state=config['seed'],
        shuffle=True
    )
    
    print(f"Dataset split: {len(train_images)} train, {len(val_images)} validation")
    
    # Create datasets
    train_dataset = ArtefactDataset(
        train_images, train_masks,
        transform=get_transforms(config, 'train'),
        binary_mode=config['data']['binary_mode'],
        ignore_index=config['data']['ignore_index']
    )
    
    val_dataset = ArtefactDataset(
        val_images, val_masks,
        transform=get_transforms(config, 'val'),
        binary_mode=config['data']['binary_mode'],
        ignore_index=config['data']['ignore_index']
    )
    
    # Create dataloaders
    bs = batch_size or config['training']['batch_size']
    nw = num_workers or config['num_workers']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True
    )
    
    return train_loader, val_loader


def verify_dataset(data_root: str, binary_mode: bool = True):
    """Verify dataset integrity and print statistics."""
    
    data_path = Path(data_root)
    images_dir = data_path / 'images'
    masks_dir = data_path / 'annotations'
    
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Data directories not found in {data_root}")
    
    image_files = list(images_dir.glob('*.png'))
    mask_files = list(masks_dir.glob('*.png'))
    
    print(f"\nDataset Verification:")
    print(f"  Location: {data_root}")
    print(f"  Images: {len(image_files)}")
    print(f"  Masks: {len(mask_files)}")
    
    # Check first mask for class distribution
    if mask_files:
        mask = np.array(Image.open(mask_files[0]))
        unique_classes = np.unique(mask)
        print(f"  Unique classes in first mask: {unique_classes}")
        
        if binary_mode:
            print(f"  Binary mode: Clean (0) vs Damage (1), Ignore (255)")
        else:
            print(f"  Multiclass mode: 16 classes (0-15) + Ignore (255)")
    
    return len(image_files) > 0
