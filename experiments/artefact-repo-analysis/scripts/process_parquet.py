#!/usr/bin/env python3
"""
Process ARTeFACT Parquet Files

Extracts images and annotations from parquet files to a structured directory.
Handles memory efficiently by processing one sample at a time and resizing large images.
"""

import sys
import io
from pathlib import Path
from typing import Optional
import json

try:
    import pyarrow.parquet as pq
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nInstall with:")
    print("  pip install pyarrow pillow tqdm")
    sys.exit(1)


def bytes_to_image(image_dict: dict) -> Optional[Image.Image]:
    """Convert parquet image dict to PIL Image."""
    try:
        if 'bytes' in image_dict and image_dict['bytes']:
            return Image.open(io.BytesIO(image_dict['bytes']))
        return None
    except Exception as e:
        print(f"  ⚠️  Failed to decode image: {e}")
        return None


def resize_if_large(img: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image if larger than max_size on any dimension."""
    width, height = img.size
    
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Use appropriate resampling
        if img.mode in ('L', 'P', 'I'):
            # For masks/annotations, use nearest neighbor
            return img.resize((new_width, new_height), Image.NEAREST)
        else:
            # For photos, use high-quality resampling
            return img.resize((new_width, new_height), Image.LANCZOS)
    
    return img


def process_parquet_file(
    parquet_path: Path,
    output_dir: Path,
    max_samples: Optional[int] = None,
    max_image_size: int = 1024,
    skip_existing: bool = True
):
    """Process a single parquet file."""
    
    print(f"\n{'='*80}")
    print(f"Processing: {parquet_path.name}")
    print(f"{'='*80}")
    
    # Read parquet
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    total_rows = len(df)
    process_count = min(max_samples, total_rows) if max_samples else total_rows
    
    print(f"Total samples: {total_rows}")
    print(f"Processing: {process_count}")
    print()
    
    # Create output directories
    image_dir = output_dir / "images"
    annotation_dir = output_dir / "annotations"
    annotation_rgb_dir = output_dir / "annotations_rgb"
    
    for d in [image_dir, annotation_dir, annotation_rgb_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Process samples
    metadata = []
    stats = {
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'materials': {},
        'contents': {},
        'types': {},
    }
    
    for idx, row in tqdm(df.iterrows(), total=process_count, desc="Processing"):
        if max_samples and idx >= max_samples:
            break
        
        sample_id = row['id']
        
        # Skip if already processed
        img_path = image_dir / f"{sample_id}.png"
        if skip_existing and img_path.exists():
            stats['skipped'] += 1
            continue
        
        try:
            # Extract images from bytes
            img = bytes_to_image(row['image'])
            ann = bytes_to_image(row['annotation'])
            ann_rgb = bytes_to_image(row['annotation_rgb'])
            
            if img is None or ann is None or ann_rgb is None:
                print(f"  ⚠️  Sample {sample_id}: Missing image data")
                stats['errors'] += 1
                continue
            
            # Resize if too large
            orig_size = img.size
            img = resize_if_large(img, max_image_size)
            ann = resize_if_large(ann, max_image_size)
            ann_rgb = resize_if_large(ann_rgb, max_image_size)
            
            # Save images
            img.save(img_path)
            ann.save(annotation_dir / f"{sample_id}.png")
            ann_rgb.save(annotation_rgb_dir / f"{sample_id}.png")
            
            # Collect metadata
            metadata.append({
                'id': sample_id,
                'material': row.get('material', ''),
                'content': row.get('content', ''),
                'type': row.get('type', ''),
                'damage_description': row.get('damage_description', ''),
                'original_size': f"{orig_size[0]}x{orig_size[1]}",
                'processed_size': f"{img.size[0]}x{img.size[1]}",
                'image_path': str(img_path.relative_to(output_dir)),
                'annotation_path': str((annotation_dir / f"{sample_id}.png").relative_to(output_dir)),
                'annotation_rgb_path': str((annotation_rgb_dir / f"{sample_id}.png").relative_to(output_dir)),
            })
            
            # Update stats
            stats['materials'][row.get('material', 'unknown')] = \
                stats['materials'].get(row.get('material', 'unknown'), 0) + 1
            stats['contents'][row.get('content', 'unknown')] = \
                stats['contents'].get(row.get('content', 'unknown'), 0) + 1
            stats['types'][row.get('type', 'unknown')] = \
                stats['types'].get(row.get('type', 'unknown'), 0) + 1
            
            stats['processed'] += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {sample_id}: {e}")
            stats['errors'] += 1
            continue
    
    return metadata, stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process ARTeFACT parquet files to extract images",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to parquet file or pattern (e.g., data/*.parquet)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data/processed',
        help='Output directory (default: ./data/processed)'
    )
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=None,
        help='Max samples per file (default: all)'
    )
    parser.add_argument(
        '--max-image-size', '-s',
        type=int,
        default=1024,
        help='Max image dimension in pixels (default: 1024)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find parquet files
    input_path = Path(args.input)
    if '*' in str(input_path):
        # Pattern matching
        parquet_files = list(input_path.parent.glob(input_path.name))
    elif input_path.is_file():
        parquet_files = [input_path]
    elif input_path.is_dir():
        parquet_files = list(input_path.glob("*.parquet"))
    else:
        print(f"❌ Invalid input: {input_path}")
        sys.exit(1)
    
    if not parquet_files:
        print(f"❌ No parquet files found matching: {args.input}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("ARTeFACT PARQUET PROCESSOR")
    print("="*80)
    print(f"Files to process: {len(parquet_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Max samples per file: {args.max_samples or 'ALL'}")
    print(f"Max image size: {args.max_image_size}px")
    print()
    
    # Process each file
    all_metadata = []
    all_stats = {
        'total_processed': 0,
        'total_skipped': 0,
        'total_errors': 0,
        'materials': {},
        'contents': {},
        'types': {},
    }
    
    for pf in sorted(parquet_files):
        metadata, stats = process_parquet_file(
            pf, output_dir,
            max_samples=args.max_samples,
            max_image_size=args.max_image_size,
            skip_existing=not args.overwrite
        )
        
        all_metadata.extend(metadata)
        all_stats['total_processed'] += stats['processed']
        all_stats['total_skipped'] += stats['skipped']
        all_stats['total_errors'] += stats['errors']
        
        # Merge dicts
        for key in ['materials', 'contents', 'types']:
            for k, v in stats[key].items():
                all_stats[key][k] = all_stats[key].get(k, 0) + v
        
        print(f"  ✓ Processed: {stats['processed']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
    
    # Save metadata
    import pandas as pd
    df = pd.DataFrame(all_metadata)
    metadata_path = output_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    # Save statistics
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"  Processed: {all_stats['total_processed']}")
    print(f"  Skipped: {all_stats['total_skipped']}")
    print(f"  Errors: {all_stats['total_errors']}")
    
    print(f"\n📁 Output:")
    print(f"  Images: {output_dir}/images/")
    print(f"  Annotations: {output_dir}/annotations/")
    print(f"  RGB Annotations: {output_dir}/annotations_rgb/")
    print(f"  Metadata: {metadata_path}")
    print(f"  Statistics: {stats_path}")
    
    print(f"\n🎨 Materials ({len(all_stats['materials'])}):")
    for mat, count in sorted(all_stats['materials'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {mat}: {count}")
    
    print(f"\n📚 Content Types ({len(all_stats['contents'])}):")
    for cont, count in sorted(all_stats['contents'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {cont}: {count}")
    
    print()


if __name__ == '__main__':
    main()
