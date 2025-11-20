"""
POC-60: ARTeFACT Dataset Loader with Hierarchical Labels

Supports both standard (16-class) and hierarchical (binary/coarse/fine) labels.
For hierarchical models, returns dict with all 3 label granularities.
"""

from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def fine_to_binary(fine_labels: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    Convert fine (16-class) labels to binary (2-class)
    
    Binary mapping:
      0 (Clean) -> 0 (Clean)
      1-15 (All damage types) -> 1 (Damage)
      255 (Ignore) -> 255 (Ignore)
    
    Args:
        fine_labels: (H, W) tensor with values 0-15 or 255
    
    Returns:
        binary_labels: (H, W) tensor with values 0, 1, or 255
    """
    binary_labels = torch.zeros_like(fine_labels)
    binary_labels[fine_labels == 0] = 0  # Clean
    binary_labels[(fine_labels >= 1) & (fine_labels <= 15)] = 1  # Damage
    binary_labels[fine_labels == ignore_index] = ignore_index  # Ignore
    return binary_labels


def fine_to_coarse(fine_labels: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    Convert fine (16-class) labels to coarse (4-class damage groups)
    
    Coarse mapping (4 damage groups):
      Group 0 - Structural Damage: [1:Material loss, 2:Peel, 3:Cracks, 4:Structural defects]
      Group 1 - Surface Contamination: [5:Dirt spots, 6:Stains, 10:Hairs, 11:Dust spots]
      Group 2 - Color Alterations: [7:Discolouration, 9:Burn marks, 13:Fading]
      Group 3 - Optical Artifacts: [8:Scratches, 12:Lightleak, 14:Blur, 15:Other]
    
    Note: Clean pixels (0) are set to ignore_index (coarse is only for damage classification)
    
    Args:
        fine_labels: (H, W) tensor with values 0-15 or 255
    
    Returns:
        coarse_labels: (H, W) tensor with values 0-3 or 255
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


class ArtefactDataset(Dataset):
    """ARTeFACT dataset for segmentation"""
    
    EXCLUDED_IMAGES = {'cljmrkz5o342f07clh6hz82sk.png'}  # Oversized image
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[A.Compose] = None
    ):
        """
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            transform: Albumentations transform
        """
        # Filter excluded images
        filtered = [
            (img, mask) for img, mask in zip(image_paths, mask_paths)
            if Path(img).name not in self.EXCLUDED_IMAGES
        ]
        
        if len(filtered) < len(image_paths):
            excluded_count = len(image_paths) - len(filtered)
            print(f"⚠️  Excluded {excluded_count} images")
        
        self.image_paths = [p[0] for p in filtered]
        self.mask_paths = [p[1] for p in filtered]
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: (3, H, W) float tensor
            mask: (H, W) long tensor with class indices
        """
        # Load image and mask
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]))
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask is long tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        
        # Clip mask values to valid range [0, 15]
        mask = mask.clamp(0, 15)
        
        return image, mask


def get_transforms(image_size: int, is_train: bool = True) -> A.Compose:
    """
    Get augmentation pipeline
    
    Args:
        image_size: Target image size
        is_train: Training or validation
    
    Returns:
        Albumentations compose
    """
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])


def create_dataloaders(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    drop_last: bool = True,
    seed: int = 42,
    use_augmented: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        data_root: Path to data directory
        image_size: Image size for resize
        batch_size: Batch size
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch
        drop_last: Drop last incomplete batch
        seed: Random seed for split
        use_augmented: Use offline augmented dataset (artefact_augmented)
    
    Returns:
        train_loader, val_loader
    """
    data_path = Path(data_root)
    
    # Check if augmented dataset exists and use_augmented is True
    augmented_path = data_path.parent / 'artefact_augmented'
    if use_augmented and augmented_path.exists():
        print(f"📦 Using augmented dataset: {augmented_path}")
        data_path = augmented_path
    else:
        print(f"📦 Using original dataset: {data_path}")
    
    image_dir = data_path / 'images'
    mask_dir = data_path / 'annotations'
    
    # Get all image paths
    image_paths = sorted(
        list(image_dir.glob('*.png')) +
        list(image_dir.glob('*.jpg'))
    )
    mask_paths = [mask_dir / img.name for img in image_paths]
    
    # Filter existing masks
    mask_paths = [m for m in mask_paths if m.exists()]
    image_paths = image_paths[:len(mask_paths)]
    
    print(f"Found {len(image_paths)} images")
    
    # Train/val split (80/20)
    np.random.seed(seed)
    indices = np.random.permutation(len(image_paths))
    split_idx = int(len(indices) * 0.8)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_images = [str(image_paths[i]) for i in train_indices]
    train_masks = [str(mask_paths[i]) for i in train_indices]
    val_images = [str(image_paths[i]) for i in val_indices]
    val_masks = [str(mask_paths[i]) for i in val_indices]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Create datasets
    train_dataset = ArtefactDataset(
        train_images,
        train_masks,
        transform=get_transforms(image_size, is_train=True)
    )
    
    val_dataset = ArtefactDataset(
        val_images,
        val_masks,
        transform=get_transforms(image_size, is_train=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()
    
    return train_loader, val_loader


def compute_class_weights(
    data_loader: DataLoader,
    num_classes: int,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Compute class weights from dataset for handling imbalance
    
    Args:
        data_loader: DataLoader to compute weights from
        num_classes: Number of classes
        device: Device to put weights on
    
    Returns:
        Class weights tensor (num_classes,)
    """
    print("Computing class weights...")
    
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    
    for _, masks in data_loader:
        for cls in range(num_classes):
            class_counts[cls] += (masks == cls).sum().item()
    
    # Compute weights (inverse frequency)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts)
    
    # Handle classes with no pixels
    class_weights[class_counts == 0] = 0.0
    
    print("\nClass distribution:")
    for cls in range(num_classes):
        print(f"  Class {cls:2d}: {class_weights[cls]:.4f} "
              f"({int(class_counts[cls]):,} pixels)")
    print()
    
    return class_weights.to(device)
