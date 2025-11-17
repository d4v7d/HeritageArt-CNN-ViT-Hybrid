"""
Pre-compute class weights for POC-5.9 dataset.

Run once to generate class_weights.json metadata file.
This avoids recalculating weights on every training run.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def compute_class_weights_static(
    mask_dir: Path,
    num_classes: int = 16,
    method: str = 'inverse_sqrt',
    ignore_index: int = 255
):
    """
    Compute class weights from all masks in dataset.
    
    Args:
        mask_dir: Path to annotations directory
        num_classes: Number of classes
        method: 'inverse', 'inverse_sqrt', or 'effective_samples'
        ignore_index: Class index to ignore
    
    Returns:
        dict with weights, counts, and metadata
    """
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    mask_paths = sorted(list(mask_dir.glob('*.png')))
    print(f"Computing class weights from {len(mask_paths)} masks...")
    
    for mask_path in tqdm(mask_paths):
        mask = np.array(Image.open(mask_path))
        
        # Count valid pixels per class
        valid_mask = mask != ignore_index
        for c in range(num_classes):
            class_counts[c] += (mask[valid_mask] == c).sum()
    
    # Compute weights
    if method == 'inverse':
        weights = 1.0 / (class_counts + 1e-8)
    elif method == 'inverse_sqrt':
        weights = 1.0 / (np.sqrt(class_counts) + 1e-8)
    elif method == 'effective_samples':
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
    else:
        weights = np.ones(num_classes, dtype=np.float32)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    # Print stats
    print(f"\nClass distribution (method={method}):")
    print("Class | Pixel Count | Weight")
    print("------|-------------|-------")
    for c in range(num_classes):
        print(f"  {c:2d}  | {int(class_counts[c]):11d} | {weights[c]:6.4f}")
    
    print(f"\nTotal pixels: {int(class_counts.sum())}")
    
    return {
        'weights': weights.tolist(),
        'counts': class_counts.tolist(),
        'method': method,
        'num_images': len(mask_paths),
        'total_pixels': int(class_counts.sum())
    }


if __name__ == '__main__':
    # Path to centralized dataset
    data_dir = Path(__file__).parent.parent.parent / 'common-data' / 'artefact_augmented'
    mask_dir = data_dir / 'annotations'
    
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    
    # Compute weights
    result = compute_class_weights_static(mask_dir, method='inverse_sqrt')
    
    # Save to metadata file
    output_file = data_dir / 'class_weights.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Class weights saved to: {output_file}")
    print(f"   To use: torch.tensor(json.load(open('{output_file}'))['weights'])")
