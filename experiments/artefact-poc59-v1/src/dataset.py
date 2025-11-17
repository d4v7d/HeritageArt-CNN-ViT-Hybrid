"""
Single-task dataset loader for POC-5.9
Simplified from POC-5.5 (no hierarchical multi-task)
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold


class ArtefactDataset(Dataset):
    """
    ARTeFACT Dataset for single-task multiclass segmentation (POC-5.9).
    
    Returns 16-class masks directly (no binary/coarse heads).
    """
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[A.Compose] = None,
        ignore_index: int = 255
    ):
        """
        Args:
            image_paths: List of paths to RGB images
            mask_paths: List of paths to PNG masks (0-15 classes + 255 ignore)
            transform: Albumentations transform pipeline
            ignore_index: Class index to ignore (default 255)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.ignore_index = ignore_index
        
        assert len(self.image_paths) == len(self.mask_paths)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: (C, H, W) tensor, float32 in [0, 1]
            mask: (H, W) tensor, int64 with class indices 0-15 or ignore_index
        """
        # Load image and mask
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]))
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Convert to tensor manually if no transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return image, mask


def get_transforms(config: Dict, mode: str = 'train') -> A.Compose:
    """
    Get augmentation transforms for POC-5.9.
    
    Medium augmentation strategy (balanced diversity + realism).
    
    Args:
        config: Config dict with data.image_size
        mode: 'train' or 'val'
    
    Returns:
        Albumentations Compose transform
    """
    img_size = config['data'].get('image_size', 384)
    
    if mode == 'train':
        # Medium Augmentation for heritage damage detection
        transforms_list = [
            # ALWAYS resize first to ensure consistent size
            A.Resize(height=img_size, width=img_size),
            
            # Geometric (heritage-safe)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
            A.Rotate(limit=20, p=0.5, border_mode=0),
            # RandomResizedCrop disabled to ensure exact 384x384 output
            # A.RandomResizedCrop(
            #     height=img_size,
            #     width=img_size,
            #     scale=(0.7, 1.0),
            #     ratio=(0.9, 1.1),
            #     p=0.7
            # ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=0,
                p=0.4
            ),
            
            # Photometric (lighting/color variations)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.4
            ),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=0.3
            ),
            
            # Quality degradation (simulate aging)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.ISONoise(p=0.2),
            
            # Normalize to [0, 1] and convert to tensor
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    else:
        # Validation: only resize + normalize
        transforms_list = [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    
    return A.Compose(
        transforms_list,
        additional_targets={'mask': 'mask'}
    )


def create_dataloaders(
    config: Dict,
    fold: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Config dictionary
        fold: If None, use random split. If int, use K-fold split.
    
    Returns:
        train_loader, val_loader
    """
    data_dir = Path(config['data']['data_dir'])
    
    # Get all image and mask paths
    image_dir = data_dir / 'images'
    mask_dir = data_dir / 'annotations'
    
    image_paths = sorted(list(image_dir.glob('*.png')))
    mask_paths = [
        mask_dir / img_path.name for img_path in image_paths
    ]
    
    # Verify all masks exist
    assert all(p.exists() for p in mask_paths), "Some masks missing!"
    
    print(f"Found {len(image_paths)} images")
    
    # Convert to strings
    image_paths = [str(p) for p in image_paths]
    mask_paths = [str(p) for p in mask_paths]
    
    # Split data
    if fold is not None:
        # K-fold cross-validation
        n_folds = config['cross_validation']['n_folds']
        seed = config['cross_validation']['seed']
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kfold.split(image_paths))
        train_idx, val_idx = splits[fold]
        
        train_images = [image_paths[i] for i in train_idx]
        train_masks = [mask_paths[i] for i in train_idx]
        val_images = [image_paths[i] for i in val_idx]
        val_masks = [mask_paths[i] for i in val_idx]
        
        print(f"Fold {fold}/{n_folds}: {len(train_images)} train, {len(val_images)} val")
    else:
        # Random split
        from sklearn.model_selection import train_test_split
        
        train_images, val_images, train_masks, val_masks = train_test_split(
            image_paths,
            mask_paths,
            test_size=1.0 - config['data']['train_split'],
            random_state=config['data']['seed']
        )
        
        print(f"Random split: {len(train_images)} train, {len(val_images)} val")
    
    # Create datasets
    train_transform = get_transforms(config, mode='train')
    val_transform = get_transforms(config, mode='val')
    
    train_dataset = ArtefactDataset(
        train_images,
        train_masks,
        transform=train_transform,
        ignore_index=config['data']['ignore_index']
    )
    
    val_dataset = ArtefactDataset(
        val_images,
        val_masks,
        transform=val_transform,
        ignore_index=config['data']['ignore_index']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['hardware'].get('pin_memory', True),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['hardware'].get('pin_memory', True)
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


# Class names for reference
CLASS_NAMES = [
    "Clean",           # 0
    "Material_loss",   # 1
    "Peel",            # 2
    "Dust",            # 3
    "Scratch",         # 4
    "Hair",            # 5
    "Dirt",            # 6
    "Fold",            # 7
    "Writing",         # 8
    "Cracks",          # 9
    "Staining",        # 10
    "Stamp",           # 11
    "Sticker",         # 12
    "Puncture",        # 13
    "Burn_marks",      # 14
    "Lightleak"        # 15
]


if __name__ == '__main__':
    """Test dataset loading."""
    
    # Mock config
    config = {
        'data': {
            'data_dir': '../../artefact-poc55-multiclass/data/artefact_augmented',
            'image_size': 384,
            'num_workers': 4,
            'train_split': 0.8,
            'seed': 42,
            'ignore_index': 255
        },
        'training': {
            'batch_size': 4
        },
        'hardware': {
            'pin_memory': True
        },
        'cross_validation': {
            'n_folds': 3,
            'seed': 42
        }
    }
    
    # Test single split
    print("Testing single split:")
    train_loader, val_loader, train_ds, val_ds = create_dataloaders(config, fold=None)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Test fold 0
    print("\nTesting fold 0:")
    train_loader, val_loader, train_ds, val_ds = create_dataloaders(config, fold=0)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Load one batch
    images, masks = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Masks: {masks.shape}")
    print(f"  Mask classes: {masks.unique()}")
    
    print("\n✅ Dataset test passed!")
