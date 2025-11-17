"""
RAM Pre-loaded Dataset for ARTeFACT Segmentation

Eliminates I/O bottleneck by loading all images into RAM at initialization.
Expected impact: 80% I/O time → ~0%, throughput 26 → 80+ imgs/s
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PreloadedArtefactDataset(Dataset):
    """
    ARTeFACT dataset with full RAM pre-loading
    
    Loads ALL images and masks into RAM at __init__.
    Transforms applied on-the-fly (fast since data already in RAM).
    """
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[A.Compose] = None,
        preload_to_gpu: bool = False
    ):
        """
        Args:
            image_paths: List of absolute paths to images
            mask_paths: List of absolute paths to masks
            transform: Albumentations transform pipeline
            preload_to_gpu: If True, keeps data on GPU (experimental, high VRAM)
        """
        assert len(image_paths) == len(mask_paths), "Mismatch in image/mask counts"
        
        self.transform = transform
        self.preload_to_gpu = preload_to_gpu
        self.device = torch.device('cuda' if preload_to_gpu and torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"🔥 Pre-loading {len(image_paths)} images into RAM...")
        print(f"{'='*60}")
        import sys
        sys.stdout.flush()
        
        self.images = []
        self.masks = []
        
        # Pre-load all data into RAM
        skipped = 0
        for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            if idx % 50 == 0:
                print(f"Loading: {idx}/{len(image_paths)}...")
                sys.stdout.flush()
            
            try:
                # Load as numpy arrays (more efficient than PIL for storage)
                img = np.array(Image.open(img_path).convert('RGB'))
                mask = np.array(Image.open(mask_path))
                
                # Clamp masks to [0, 15] immediately (ignore_index=255 → 15)
                mask = np.clip(mask, 0, 15).astype(np.uint8)
                
                if self.preload_to_gpu:
                    # Convert to tensors and move to GPU
                    img = torch.from_numpy(img).to(self.device)
                    mask = torch.from_numpy(mask).to(self.device)
                
                self.images.append(img)
                self.masks.append(mask)
            except (OSError, IOError) as e:
                print(f"\n⚠️  Skipping corrupted file: {Path(img_path).name}")
                skipped += 1
                continue
        
        # Calculate RAM usage
        import sys
        if not self.preload_to_gpu:
            total_size = sum(img.nbytes + mask.nbytes for img, mask in zip(self.images, self.masks))
            print(f"\n✅ Pre-loaded {len(self.images)} images ({skipped} skipped)")
            print(f"💾 RAM usage: {total_size / 1024**3:.2f} GB")
        else:
            print(f"\n✅ Pre-loaded {len(self.images)} images to GPU ({skipped} skipped)")
        print(f"{'='*60}\n")
        sys.stdout.flush()
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns preprocessed image and mask.
        No I/O - data already in RAM.
        """
        image = self.images[idx]
        mask = self.masks[idx]
        
        # Convert GPU tensors back to numpy for albumentations
        if self.preload_to_gpu:
            image = image.cpu().numpy()
            mask = mask.cpu().numpy()
        
        # Apply transforms (resize, augmentations, normalize)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()  # Ensure long type for loss
        else:
            # Fallback: at minimum convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return image, mask


def create_preloaded_dataloaders(
    data_root: str,
    image_size: int = 384,
    batch_size: int = 64,
    num_workers: int = 8,
    use_augmented: bool = False,
    preload_to_gpu: bool = False,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train/val dataloaders with RAM pre-loading.
    
    Args:
        data_root: Path to data directory  
        image_size: Target image size (square)
        batch_size: Batch size
        num_workers: Number of DataLoader workers (can be lower with pre-loading)
        use_augmented: Use augmented dataset (1,463 images vs 334)
        preload_to_gpu: Keep data on GPU (experimental)
        seed: Random seed for train/val split
    
    Returns:
        train_loader, val_loader
    """
    
    # Paths
    data_path = Path(data_root).resolve()
    
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
    
    print(f"📊 Found {len(image_paths)} images")
    
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
    
    print(f"📊 Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Transforms
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create pre-loaded datasets
    train_dataset = PreloadedArtefactDataset(
        train_images, train_masks, train_transform, preload_to_gpu=preload_to_gpu
    )
    val_dataset = PreloadedArtefactDataset(
        val_images, val_masks, val_transform, preload_to_gpu=preload_to_gpu
    )
    
    # DataLoaders (can use fewer workers since no I/O)
    # With pre-loading, workers mainly do augmentation transforms
    effective_workers = max(2, num_workers // 2) if not preload_to_gpu else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=not preload_to_gpu,  # No need if already on GPU
        persistent_workers=effective_workers > 0,
        prefetch_factor=2 if effective_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=not preload_to_gpu,
        persistent_workers=effective_workers > 0,
        prefetch_factor=2 if effective_workers > 0 else None
    )
    
    print(f"⚙️  DataLoader workers: {effective_workers} (reduced due to pre-loading)")
    
    return train_loader, val_loader

