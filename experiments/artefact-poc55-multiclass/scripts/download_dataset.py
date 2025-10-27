"""
Download ARTeFACT Full Dataset from HuggingFace

Dataset: danielaivanova/damaged-media  
Expected: ~418 samples with 16-class annotations
"""

import os
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Suppress Pillow warnings for large images
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

# Allow large images (ARTeFACT has high-res scans)
Image.MAX_IMAGE_PIXELS = None


def download_artefact_full(output_dir='./data/artefact'):
    """Download ARTeFACT dataset from HuggingFace."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_path / 'images'
    annotations_dir = output_path / 'annotations'
    annotations_rgb_dir = output_path / 'annotations_rgb'
    
    images_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)
    annotations_rgb_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Downloading ARTeFACT Dataset from HuggingFace")
    print("=" * 70)
    print(f"Source: danielaivanova/damaged-media")
    print(f"Output: {output_path.absolute()}")
    print()
    
    # Load dataset from HuggingFace
    print("Loading dataset (this may take a few minutes)...")
    try:
        dataset = load_dataset("danielaivanova/damaged-media", split="train")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    total_samples = len(dataset)
    print(f"✅ Dataset loaded: {total_samples} samples\n")
    
    metadata = []
    
    # Process each sample with progress bar
    print(f"Processing {total_samples} samples...")
    for idx in tqdm(range(total_samples), desc="Saving images"):
        sample = dataset[idx]
        sample_id = sample.get('id', f'sample_{idx:04d}')
        
        image = sample.get('image')
        annotation = sample.get('annotation')
        annotation_rgb = sample.get('annotation_rgb')
        material = sample.get('material', 'unknown')
        content = sample.get('content', 'unknown')
        damage_type = sample.get('type', 'unknown')
        
        try:
            # Save image
            img_path = images_dir / f"{sample_id}.png"
            if image is not None:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(img_path, 'PNG')
            
            # Save annotation  
            ann_path = annotations_dir / f"{sample_id}.png"
            if annotation is not None:
                ann_array = np.array(annotation)
                ann_uint8 = ann_array.astype(np.uint8)
                Image.fromarray(ann_uint8).save(ann_path, 'PNG')
            
            # Save annotation RGB
            ann_rgb_path = annotations_rgb_dir / f"{sample_id}.png"
            if annotation_rgb is not None:
                if annotation_rgb.mode != 'RGB':
                    annotation_rgb = annotation_rgb.convert('RGB')
                annotation_rgb.save(ann_rgb_path, 'PNG')
            
            metadata.append({
                'id': sample_id,
                'material': material,
                'content': content,
                'type': damage_type
            })
            
        except Exception as e:
            print(f"\n⚠️ Warning: Failed to process sample {sample_id}: {e}")
            continue
    
    # Save metadata CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_path = output_path / 'metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\n✅ Metadata saved: {metadata_path}")
    
    print("\n" + "=" * 70)
    print("✅ Download complete!")
    print("=" * 70)
    print(f"Dataset location: {output_path.absolute()}")
    print(f"  - Images: {len(list(images_dir.glob('*.png')))} files")
    print(f"  - Annotations: {len(list(annotations_dir.glob('*.png')))} files")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download ARTeFACT dataset')
    parser.add_argument('--output-dir', type=str, default='./data/artefact')
    
    args = parser.parse_args()
    
    success = download_artefact_full(args.output_dir)
    sys.exit(0 if success else 1)
