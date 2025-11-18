#!/usr/bin/env python3
"""
Generate Augmented ARTeFACT Dataset

This script generates the augmented dataset (artefact_augmented/) from the 
original ARTeFACT dataset (artefact/).

⚠️  STATUS: PLACEHOLDER / TO BE IMPLEMENTED

The current artefact_augmented/ (1458 samples) was generated during POC-5.8
development but the exact script was not committed to git.

This file documents the approximate augmentation strategy used.

Usage:
    python generate_augmentations.py --input artefact/ --output artefact_augmented/

Expected:
    - Input: 417 original images + masks (from artefact-data-obtention)
    - Output: ~1458 augmented images + masks (~3.5x multiplier)
    - Augmentations: Horizontal flip, vertical flip, rotation, color jitter, blur

Author: Brandon Trigueros
Date: November 17, 2025 (Placeholder created)
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import json

import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentation_pipeline() -> A.Compose:
    """
    Heritage-specific augmentation pipeline.
    
    Based on POC-5.8 training configurations and commit fae549f
    "add heritage-specific augmentations".
    
    Strategy:
    - Geometric: Flips and small rotations (heritage art is usually upright)
    - Color: Mild jitter to simulate aging/lighting conditions
    - Noise: Mild blur to simulate scan quality variations
    """
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.8),  # Small rotations only
        
        # Color augmentations (simulate aging/lighting)
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
            p=0.7
        ),
        
        # Quality augmentations (simulate scan variations)
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        
        # Rare augmentations (extreme cases)
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
    ], additional_targets={'mask': 'mask'})


def compute_class_weights(annotations_dir: Path, num_classes: int = 16) -> dict:
    """
    Compute class weights for balanced training.
    
    Uses inverse sqrt log scaling for extreme class imbalance.
    """
    print("\n" + "="*80)
    print("COMPUTING CLASS WEIGHTS")
    print("="*80)
    
    # Count pixels per class
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    
    mask_files = sorted(annotations_dir.glob("*.png"))
    print(f"Analyzing {len(mask_files)} masks...")
    
    for mask_path in tqdm(mask_files, desc="Counting pixels"):
        mask = np.array(Image.open(mask_path))
        for class_id in range(num_classes):
            class_pixel_counts[class_id] += np.sum(mask == class_id)
    
    # Compute weights (inverse sqrt log scaled)
    total_pixels = class_pixel_counts.sum()
    class_frequencies = class_pixel_counts / total_pixels
    
    # Inverse sqrt log scaling (handles extreme imbalance)
    epsilon = 1e-7
    raw_weights = np.sqrt(1.0 / (class_frequencies + epsilon))
    raw_weights = np.log1p(raw_weights)  # Log scaling for extreme cases
    
    # Scale to reasonable range (0.1 - 3.0)
    scale_factor = 10.0
    weights = raw_weights / raw_weights.mean() * scale_factor
    weights = np.clip(weights, 0.08, 3.0)
    
    # Report
    print("\nClass Distribution:")
    for class_id in range(num_classes):
        freq = class_frequencies[class_id]
        weight = weights[class_id]
        print(f"  Class {class_id:2d}: {freq*100:6.2f}% pixels → weight {weight:.3f}")
    
    return {
        'method': 'inverse_sqrt_log_scaled',
        'num_images': len(mask_files),
        'weights': weights.tolist(),
        'original_weights': raw_weights.tolist(),
        'scale_factor': scale_factor,
    }


def generate_augmentations(
    input_dir: Path,
    output_dir: Path,
    num_augmentations: int = 3,
    seed: int = 42
):
    """
    Generate augmented dataset.
    
    Args:
        input_dir: Path to original artefact/ directory
        output_dir: Path to output artefact_augmented/ directory
        num_augmentations: Number of augmented versions per image
        seed: Random seed for reproducibility
    """
    print("\n" + "="*80)
    print("GENERATING AUGMENTED DATASET")
    print("="*80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentations per image: {num_augmentations}")
    print(f"Random seed: {seed}")
    print()
    
    # Setup
    np.random.seed(seed)
    transform = get_augmentation_pipeline()
    
    # Create output directories
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    
    # Get input files
    input_images = sorted((input_dir / 'images').glob('*.png'))
    input_masks = sorted((input_dir / 'annotations').glob('*.png'))
    
    assert len(input_images) == len(input_masks), "Mismatch between images and masks"
    print(f"Found {len(input_images)} image-mask pairs")
    
    # Generate augmentations
    total_generated = 0
    
    for img_path, mask_path in tqdm(zip(input_images, input_masks), 
                                     total=len(input_images),
                                     desc="Augmenting"):
        # Load original
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))
        
        base_name = img_path.stem
        
        # Save original (no augmentation)
        Image.fromarray(image).save(output_dir / 'images' / f'{base_name}.png')
        Image.fromarray(mask).save(output_dir / 'annotations' / f'{base_name}.png')
        total_generated += 1
        
        # Generate N augmented versions
        for aug_idx in range(num_augmentations):
            # Apply augmentation
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Save with suffix
            aug_name = f'{base_name}_aug{aug_idx+1}'
            Image.fromarray(aug_image).save(output_dir / 'images' / f'{aug_name}.png')
            Image.fromarray(aug_mask).save(output_dir / 'annotations' / f'{aug_name}.png')
            total_generated += 1
    
    print(f"\n✓ Generated {total_generated} image-mask pairs")
    print(f"  Original: {len(input_images)}")
    print(f"  Augmented: {total_generated - len(input_images)}")
    
    # Compute and save class weights
    print("\nComputing class weights...")
    class_weights = compute_class_weights(output_dir / 'annotations')
    
    weights_path = output_dir / 'class_weights_balanced.json'
    with open(weights_path, 'w') as f:
        json.dump(class_weights, f, indent=2)
    print(f"✓ Saved class weights to {weights_path}")
    
    print("\n" + "="*80)
    print(" AUGMENTATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total samples: {total_generated}")
    print(f"Ready for training!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented ARTeFACT dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # From scripts/ directory - Generate 3x augmentations (default)
    python generate_augmentations.py
    
    # Generate 5x augmentations
    python generate_augmentations.py --num-aug 5
    
    # Custom paths
    python generate_augmentations.py --input ../artefact/ --output ../artefact_augmented/
    
Expected:
    Input:  ../artefact/ (417 images + masks from download_artefact.py)
    Output: ../artefact_augmented/ (~1458 images + masks = 417 + 417*3 augmentations)
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='../artefact',
        help='Input directory with original dataset (default: ../artefact)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../artefact_augmented',
        help='Output directory for augmented dataset (default: ../artefact_augmented)'
    )
    parser.add_argument(
        '--num-aug', '-n',
        type=int,
        default=3,
        help='Number of augmented versions per image (default: 3)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        print("\nDownload original dataset first:")
        print("  cd .. && make download-full")
        print("  (This downloads to ../artefact/)")
        return 1
    
    if not (input_dir / 'images').exists():
        print(f"ERROR: {input_dir}/images/ not found")
        print("Expected structure:")
        print("  ../artefact/")
        print("    ├── images/")
        print("    └── annotations/")
        return 1
    
    output_dir = Path(args.output)
    if output_dir.exists():
        print(f"WARNING: Output directory already exists: {output_dir}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
    
    # Generate
    try:
        generate_augmentations(
            input_dir=input_dir,
            output_dir=output_dir,
            num_augmentations=args.num_aug,
            seed=args.seed
        )
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
