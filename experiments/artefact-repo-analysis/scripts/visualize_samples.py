#!/usr/bin/env python3
"""
Quick visualization script to verify extracted images.
"""

import sys
from pathlib import Path

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# Read metadata
metadata_path = Path("data/processed/metadata.csv")
if not metadata_path.exists():
    print("ERROR: Metadata not found. Run process_parquet.py first.")
    sys.exit(1)

df = pd.read_csv(metadata_path)
print(f"Found {len(df)} processed samples")
print()

# Visualize first 3 samples
num_samples = min(3, len(df))

for idx in range(num_samples):
    row = df.iloc[idx]
    
    print(f"Sample {idx+1}: {row['id']}")
    print(f"  Material: {row['material']}")
    print(f"  Type: {row['type']}")
    print(f"  Damage: {row['damage_description']}")
    print(f"  Size: {row['original_size']} → {row['processed_size']}")
    print()
    
    # Load images
    img_path = Path("data/processed") / row['image_path']
    ann_path = Path("data/processed") / row['annotation_path']
    ann_rgb_path = Path("data/processed") / row['annotation_rgb_path']
    
    img = Image.open(img_path)
    ann = Image.open(ann_path)
    ann_rgb = Image.open(ann_rgb_path)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(ann, cmap='gray')
    axes[1].set_title('Annotation (Grayscale)')
    axes[1].axis('off')
    
    axes[2].imshow(ann_rgb)
    axes[2].set_title('Annotation (RGB)')
    axes[2].axis('off')
    
    # Overlay
    import numpy as np
    overlay = np.array(img).copy()
    ann_np = np.array(ann)
    ann_rgb_np = np.array(ann_rgb)
    mask = ann_np != 255
    if overlay.shape[:2] == ann_rgb_np.shape[:2]:
        overlay[mask] = (overlay[mask] * 0.6 + ann_rgb_np[mask] * 0.4).astype(np.uint8)
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.suptitle(f"{row['id']} - {row['type']}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path("data/processed/visualizations")
    output_path.mkdir(exist_ok=True)
    viz_filename = f"{row['id']}_viz.png"
    plt.savefig(output_path / viz_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization to {output_path / viz_filename}")

print("\n Visualizations complete!")
print(f" See: data/processed/visualizations/")
