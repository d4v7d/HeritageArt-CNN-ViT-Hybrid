#!/usr/bin/env python3
"""
ARTeFACT Data Obtention - HuggingFace Streaming Approach
=========================================================

Code example demonstrating how to download ARTeFACT dataset using HuggingFace 
datasets library with streaming mode.

WARNING: LIMITATION: This approach crashes on extremely large images (>50M pixels).
   Successfully processes ~5-9 samples before hitting memory limits.
   For production data obtention, use ../artefact-repo-analysis/

Features:
- Streaming download (incremental processing)
- Automatic image resizing (max 512px)
- Progress tracking with tqdm
- Metadata and statistics export
- Sample visualizations (4-panel)

Use Case:
- Learning HuggingFace datasets API
- Training pipeline integration examples
- Small subset downloads for testing

Author: PoC for HeritageArt-CNN-ViT-Hybrid
"""

import os
import sys
import gc
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb protection
import matplotlib.pyplot as plt
from tqdm import tqdm


# Class mappings from ARTeFACT
CLASS_NAMES = [
    "Clean",              # 0
    "Material loss",      # 1
    "Peel",              # 2
    "Dust",              # 3
    "Scratch",           # 4
    "Hair",              # 5
    "Dirt",              # 6
    "Fold",              # 7
    "Writing",           # 8
    "Cracks",            # 9
    "Staining",          # 10
    "Stamp",             # 11
    "Sticker",           # 12
    "Puncture",          # 13
    "Burn marks",        # 14
    "Lightleak",         # 15
    "Background",        # 255 (ignore)
]

# Colors for visualization (hex)
CLASS_COLORS = {
    "Material loss": "#1CE6FF",
    "Peel": "#FF34FF",
    "Dust": "#FF4A46",
    "Scratch": "#008941",
    "Hair": "#006FA6",
    "Dirt": "#A30059",
    "Fold": "#FFA500",
    "Writing": "#7A4900",
    "Cracks": "#0000A6",
    "Staining": "#63FFAC",
    "Stamp": "#004D43",
    "Sticker": "#8FB0FF",
    "Puncture": "#997D87",
    "Background": "#5A0007",
    "Burn marks": "#809693",
    "Lightleak": "#f6ff1b",
    "Clean": "#000000",
}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def load_artefact_dataset(max_samples: int = None):
    """Load ARTeFACT dataset from Hugging Face using streaming to avoid OOM."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed")
        print("Install with: pip install datasets")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("DOWNLOADING ARTeFACT DATASET FROM HUGGING FACE")
    print("="*80)
    print("Dataset: danielaivanova/damaged-media")
    print("Using streaming mode to avoid memory issues...")
    print()
    
    try:
        # Use streaming=True to avoid loading entire dataset in memory
        ds = load_dataset("danielaivanova/damaged-media", split="train", streaming=True)
        print(f"✓ Successfully connected to dataset stream")
        
        # Convert to list with limit
        if max_samples:
            print(f"  Downloading first {max_samples} samples...")
        
        return ds
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        sys.exit(1)


def save_artefact_to_disk(dataset, output_dir: Path, max_samples: int = None, create_visualizations: bool = True):
    """Save ARTeFACT dataset to disk with validation, using streaming."""
    
    print("\n" + "="*80)
    print("SAVING DATASET TO DISK")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print()
    
    # Create directories
    image_dir = output_dir / "images"
    annotation_dir = output_dir / "annotations"
    annotation_rgb_dir = output_dir / "annotations_rgb"
    viz_dir = output_dir / "visualizations"
    
    for d in [image_dir, annotation_dir, annotation_rgb_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Save samples
    metadata_rows = []
    statistics = {
        'total_samples': 0,
        'materials': {},
        'contents': {},
        'class_pixel_counts': {name: 0 for name in CLASS_NAMES},
        'image_sizes': [],
        'samples_with_annotations': 0,
    }
    
    # Iterate through streaming dataset
    print(f"Processing samples from stream...")
    print(f"Target: {max_samples if max_samples else 'all'} successful samples")
    sample_count = 0
    skipped_count = 0
    iterations = 0
    max_iterations = (max_samples * 3) if max_samples else 1000  # Allow for skipped large images
    
    with tqdm(desc="Saving", total=max_samples) as pbar:
        for sample in dataset:
            iterations += 1
            
            # Stop if we have enough successful samples OR too many iterations
            if max_samples and sample_count >= max_samples:
                break
            if iterations >= max_iterations:
                print(f"\n⚠ Reached max iterations ({max_iterations}), stopping...")
                break
            
            try:
                sample_id = sample.get('id', f'unknown_{iterations}')
                material = sample.get('material', 'unknown')
                content = sample.get('content', 'unknown')
                
                # Get data and resize to reduce memory usage
                # Original images can be EXTREMELY large (>133M pixels)
                # Use thumbnail to avoid loading full image in memory
                max_size = 512  # Reduced from 1024 for safety
                
                try:
                    img_pil = sample['image']
                    # Check size before loading
                    orig_width, orig_height = img_pil.size
                    
                    # Skip if absurdly large (>50M pixels = likely causes crash)
                    if orig_width * orig_height > 50_000_000:
                        print(f"\nWARNING:  Skipping sample {sample_id}: Image too large ({orig_width}x{orig_height} = {orig_width*orig_height:,} pixels)")
                        skipped_count += 1
                        continue
                    
                    # Use thumbnail for efficient resizing without loading full image
                    img_pil.thumbnail((max_size, max_size), Image.LANCZOS)
                    
                except Exception as e:
                    print(f"\nWARNING:  Skipping sample {sample_id}: Failed to load image ({e})")
                    skipped_count += 1
                    continue
                
                try:
                    ann_pil = sample['annotation']
                    ann_pil.thumbnail((max_size, max_size), Image.NEAREST)
                except Exception as e:
                    print(f"\nWARNING:  Skipping sample {sample_id}: Failed to load annotation ({e})")
                    skipped_count += 1
                    continue
                
                try:
                    ann_rgb_pil = sample['annotation_rgb']
                    ann_rgb_pil.thumbnail((max_size, max_size), Image.NEAREST)
                except Exception as e:
                    print(f"\nWARNING:  Skipping sample {sample_id}: Failed to load RGB annotation ({e})")
                    skipped_count += 1
                    continue
                
                image = np.array(img_pil, dtype=np.uint8)
                annotation = np.array(ann_pil, dtype=np.uint8)
                annotation_rgb = np.array(ann_rgb_pil, dtype=np.uint8)
                
                # Save images
                img_path = image_dir / f"{sample_id}.png"
                ann_path = annotation_dir / f"{sample_id}.png"
                ann_rgb_path = annotation_rgb_dir / f"{sample_id}.png"
                
                Image.fromarray(image).save(img_path)
                Image.fromarray(annotation).save(ann_path)
                Image.fromarray(annotation_rgb).save(ann_rgb_path)
                
                # Update statistics
                statistics['materials'][material] = statistics['materials'].get(material, 0) + 1
                statistics['contents'][content] = statistics['contents'].get(content, 0) + 1
                statistics['image_sizes'].append(f"{image.shape[1]}x{image.shape[0]}")
                
                # Count class pixels
                unique, counts = np.unique(annotation, return_counts=True)
                if len(unique) > 1 or unique[0] != 255:  # Has annotations
                    statistics['samples_with_annotations'] += 1
                
                for val, count in zip(unique, counts):
                    if val == 0:
                        statistics['class_pixel_counts']['Clean'] += int(count)
                    elif val == 255:
                        statistics['class_pixel_counts']['Background'] += int(count)
                    elif 1 <= val <= 15:
                        statistics['class_pixel_counts'][CLASS_NAMES[val]] += int(count)
                
                # Metadata
                metadata_rows.append({
                    'id': sample_id,
                    'material': material,
                    'content': content,
                    'image_path': str(img_path),
                    'annotation_path': str(ann_path),
                    'annotation_rgb_path': str(ann_rgb_path),
                    'width': image.shape[1],
                    'height': image.shape[0],
                })
                
                # Create visualization for first 10 samples
                if create_visualizations and sample_count < 10:
                    create_sample_visualization(
                        image, annotation, annotation_rgb,
                        sample_id, viz_dir
                    )
                
                # Explicitly delete large objects to free memory
                del image, annotation, annotation_rgb
                del img_pil, ann_pil, ann_rgb_pil
                
                sample_count += 1
                pbar.update(1)
            
            except Exception as e:
                print(f"\nWarning: Failed to process sample #{iterations}: {e}")
                skipped_count += 1
                continue
    
    statistics['total_samples'] = sample_count
    statistics['skipped_samples'] = skipped_count
    
    print(f"\n✓ Successfully processed {sample_count} samples")
    if skipped_count > 0:
        print(f"⚠ Skipped {skipped_count} samples due to memory/size issues")
    
    # Save metadata
    df = pd.DataFrame(metadata_rows)
    metadata_path = output_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"✓ Saved metadata to {metadata_path}")
    
    # Save statistics (convert numpy types to Python types for JSON)
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    statistics_json = convert_numpy_types(statistics)
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(statistics_json, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")
    
    return df, statistics


def create_sample_visualization(image, annotation, annotation_rgb, sample_id, viz_dir):
    """Create a 4-panel visualization of a sample."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Annotation (grayscale)
    axes[0, 1].imshow(annotation, cmap='tab20')
    axes[0, 1].set_title('Annotation (Grayscale)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Annotation RGB
    axes[1, 0].imshow(annotation_rgb)
    axes[1, 0].set_title('Annotation (RGB Colored)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Overlay
    overlay = image.copy()
    mask = annotation != 255  # Non-background pixels
    overlay[mask] = (overlay[mask] * 0.5 + annotation_rgb[mask] * 0.5).astype(np.uint8)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (Image + Annotation)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle(f'Sample: {sample_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = viz_dir / f"{sample_id}_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_statistics(statistics: Dict, df: pd.DataFrame):
    """Print detailed statistics about the dataset."""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    print(f"\n General Information:")
    print(f"  Total samples: {statistics['total_samples']}")
    print(f"  Samples with damage annotations: {statistics['samples_with_annotations']}")
    print(f"  Samples without damage: {statistics['total_samples'] - statistics['samples_with_annotations']}")
    
    print(f"\n🎨 Materials Distribution:")
    for material, count in sorted(statistics['materials'].items(), key=lambda x: -x[1]):
        percentage = (count / statistics['total_samples']) * 100
        print(f"  {material:15s}: {count:3d} samples ({percentage:5.1f}%)")
    
    print(f"\n📚 Content Distribution:")
    for content, count in sorted(statistics['contents'].items(), key=lambda x: -x[1]):
        percentage = (count / statistics['total_samples']) * 100
        print(f"  {content:15s}: {count:3d} samples ({percentage:5.1f}%)")
    
    print(f"\n📏 Image Sizes:")
    from collections import Counter
    size_counts = Counter(statistics['image_sizes'])
    for size, count in size_counts.most_common(10):
        print(f"  {size:15s}: {count:3d} images")
    
    print(f"\n🏷️  Damage Class Pixel Distribution:")
    total_pixels = sum(statistics['class_pixel_counts'].values())
    sorted_classes = sorted(
        [(name, count) for name, count in statistics['class_pixel_counts'].items() if count > 0],
        key=lambda x: -x[1]
    )
    for class_name, pixel_count in sorted_classes:
        percentage = (pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"  {class_name:20s}: {pixel_count:12,d} pixels ({percentage:5.2f}%)")


def validate_dataset(output_dir: Path, df: pd.DataFrame):
    """Validate that all files exist and are readable."""
    print("\n" + "="*80)
    print("VALIDATING DATASET")
    print("="*80)
    
    errors = []
    
    print("\nChecking file integrity...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        # Check image
        if not Path(row['image_path']).exists():
            errors.append(f"Missing image: {row['image_path']}")
        else:
            try:
                img = Image.open(row['image_path'])
                img.verify()
            except Exception as e:
                errors.append(f"Corrupted image {row['image_path']}: {e}")
        
        # Check annotation
        if not Path(row['annotation_path']).exists():
            errors.append(f"Missing annotation: {row['annotation_path']}")
        
        # Check RGB annotation
        if not Path(row['annotation_rgb_path']).exists():
            errors.append(f"Missing RGB annotation: {row['annotation_rgb_path']}")
    
    if errors:
        print(f"\nWARNING:  Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("\n All files validated successfully!")
    
    return len(errors) == 0


def create_summary_visualization(output_dir: Path, statistics: Dict):
    """Create summary visualizations."""
    print("\n" + "="*80)
    print("CREATING SUMMARY VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Materials distribution
    materials = list(statistics['materials'].keys())
    material_counts = list(statistics['materials'].values())
    axes[0, 0].bar(materials, material_counts, color='steelblue')
    axes[0, 0].set_title('Materials Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Material Type')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Contents distribution
    contents = list(statistics['contents'].keys())
    content_counts = list(statistics['contents'].values())
    axes[0, 1].bar(contents, content_counts, color='coral')
    axes[0, 1].set_title('Contents Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Content Type')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Damage class distribution (top 10)
    sorted_classes = sorted(
        [(name, count) for name, count in statistics['class_pixel_counts'].items() if count > 0],
        key=lambda x: -x[1]
    )[:10]
    class_names = [x[0] for x in sorted_classes]
    class_counts = [x[1] for x in sorted_classes]
    axes[1, 0].barh(class_names, class_counts, color='lightgreen')
    axes[1, 0].set_title('Top 10 Damage Classes (by pixel count)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Pixel Count')
    axes[1, 0].invert_yaxis()
    
    # 4. Annotation coverage
    total = statistics['total_samples']
    with_ann = statistics['samples_with_annotations']
    without_ann = total - with_ann
    axes[1, 1].pie(
        [with_ann, without_ann],
        labels=['With Damage', 'Without Damage'],
        autopct='%1.1f%%',
        colors=['#ff9999', '#66b3ff'],
        startangle=90
    )
    axes[1, 1].set_title('Damage Annotation Coverage', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    summary_path = output_dir / 'summary_statistics.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved summary visualization to {summary_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ARTeFACT Dataset Obtention POC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download first 50 samples (quick test)
  python download_artefact.py --max-samples 50 --output ./data/artefact_poc

  # Download full dataset
  python download_artefact.py --output ./data/artefact_full --all

  # Download with visualizations disabled (faster)
  python download_artefact.py --max-samples 100 --no-viz
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./artefact_dataset',
        help='Output directory for downloaded data (default: ./artefact_dataset)'
    )
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=50,
        help='Maximum number of samples to download (default: 50)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Download entire dataset (overrides --max-samples)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip creating visualizations (faster)'
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    max_samples = None if args.all else args.max_samples
    
    print("\n" + "="*80)
    print("ARTeFACT DATA OBTENTION POC")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {'ALL' if max_samples is None else max_samples}")
    print(f"Visualizations: {'No' if args.no_viz else 'Yes'}")
    print()
    
    # Download
    dataset = load_artefact_dataset(max_samples=max_samples)
    
    # Save to disk
    df, statistics = save_artefact_to_disk(
        dataset, 
        output_dir,
        max_samples=max_samples,
        create_visualizations=not args.no_viz
    )
    
    # Print statistics
    print_statistics(statistics, df)
    
    # Validate
    validate_dataset(output_dir, df)
    
    # Create summary visualization
    if not args.no_viz:
        create_summary_visualization(output_dir, statistics)
    
    # Final summary
    print("\n" + "="*80)
    print(" POC COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\n All data saved to: {output_dir}")
    print(f"\n Files generated:")
    print(f"  - metadata.csv: {len(df)} rows")
    print(f"  - statistics.json: Detailed dataset statistics")
    print(f"  - images/: {len(df)} original images")
    print(f"  - annotations/: {len(df)} grayscale annotation masks")
    print(f"  - annotations_rgb/: {len(df)} colored annotation maps")
    if not args.no_viz:
        print(f"  - visualizations/: Sample visualizations + summary")
    print(f"\n Dataset ready for use!")
    print()


if __name__ == '__main__':
    main()
