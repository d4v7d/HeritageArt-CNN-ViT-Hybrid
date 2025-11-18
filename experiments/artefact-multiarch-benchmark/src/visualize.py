#!/usr/bin/env python3
"""
POC-5.9-v2 Visualization Script
Generate prediction visualizations comparing ground truth vs predictions.
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model_factory import create_model
from dataset import ArtefactDataset, get_transforms  # Use normal dataset, not preloaded


# Class names (16 damage classes)
CLASS_NAMES = [
    "Clean", "Material_loss", "Peel", "Cracks", "Structural_defects",
    "Dirt_spots", "Stains", "Discolouration", "Scratches", "Burn_marks",
    "Hairs", "Dust_spots", "Lightleak", "Fading", "Blur", "Other_damage"
]

# Color palette for visualizations
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))[:16]


def load_model_and_checkpoint(config_path: str, checkpoint_path: str = None) -> Tuple[nn.Module, Dict]:
    """Load model and checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Find checkpoint if not specified
    model_config = config['model']
    if checkpoint_path is None:
        model_name = model_config['encoder_name'].replace('/', '_')
        if model_name.startswith('tu-'):
            model_dir = f"Unet_{model_name}"
        else:
            model_dir = f"Unet_{model_name}"
        
        checkpoint_path = f"../logs/{model_dir}/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded checkpoint: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}, Best mIoU: {checkpoint['best_miou']:.4f}")
    
    return model, config, device


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_samples: int = 10
) -> List[Dict]:
    """Collect predictions for visualization."""
    model.eval()
    predictions = []
    
    for images, masks in tqdm(val_loader, desc="Collecting samples"):
        if len(predictions) >= num_samples:
            break
        
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        
        # Store predictions
        for img, mask, pred in zip(images, masks, preds):
            if len(predictions) >= num_samples:
                break
            
            predictions.append({
                'image': img.cpu().numpy(),
                'mask': mask.cpu().numpy(),
                'pred': pred.cpu().numpy()
            })
    
    return predictions


def plot_prediction_grid(predictions: List[Dict], output_path: str, title: str = "Predictions"):
    """Plot grid of predictions (Image | GT | Pred | Overlay)."""
    num_samples = len(predictions)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, pred in enumerate(predictions):
        # Denormalize image
        image = pred['image'].transpose(1, 2, 0)  # CHW -> HWC
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        mask = pred['mask']
        pred_mask = pred['pred']
        
        # Column 1: Input Image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title('Input Image' if idx == 0 else '', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Column 2: Ground Truth
        axes[idx, 1].imshow(mask, cmap='tab20', vmin=0, vmax=15)
        axes[idx, 1].set_title('Ground Truth' if idx == 0 else '', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Column 3: Prediction
        axes[idx, 2].imshow(pred_mask, cmap='tab20', vmin=0, vmax=15)
        axes[idx, 2].set_title('Prediction' if idx == 0 else '', fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')
        
        # Column 4: Overlay (Image + Pred with transparency)
        axes[idx, 3].imshow(image)
        # Create colored mask overlay
        pred_colored = COLORS[pred_mask]
        pred_alpha = (pred_mask > 0).astype(float) * 0.5  # Only show non-background
        axes[idx, 3].imshow(pred_colored[..., :3], alpha=pred_alpha)
        axes[idx, 3].set_title('Overlay' if idx == 0 else '', fontsize=12, fontweight='bold')
        axes[idx, 3].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_class_distribution(predictions: List[Dict], output_path: str):
    """Plot class distribution comparison (GT vs Pred)."""
    gt_counts = np.zeros(16)
    pred_counts = np.zeros(16)
    
    for pred in predictions:
        for cls in range(16):
            gt_counts[cls] += (pred['mask'] == cls).sum()
            pred_counts[cls] += (pred['pred'] == cls).sum()
    
    # Normalize to percentages
    gt_counts = gt_counts / gt_counts.sum() * 100
    pred_counts = pred_counts / pred_counts.sum() * 100
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(16)
    width = 0.35
    
    ax.bar(x - width/2, gt_counts, width, label='Ground Truth', alpha=0.8)
    ax.bar(x + width/2, pred_counts, width, label='Prediction', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution: Ground Truth vs Prediction', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_error_map(predictions: List[Dict], output_path: str):
    """Plot error maps showing correct/incorrect predictions."""
    num_samples = min(len(predictions), 6)
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(num_samples):
        pred = predictions[idx]
        mask = pred['mask']
        pred_mask = pred['pred']
        
        # Denormalize image
        image = pred['image'].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        # Create error map: 0=correct, 1=wrong, 2=GT damage but pred clean, 3=GT clean but pred damage
        error_map = np.zeros_like(mask)
        correct = (mask == pred_mask)
        error_map[correct] = 0
        error_map[~correct] = 1
        
        # Top row: Image
        axes[0, idx].imshow(image)
        axes[0, idx].set_title(f'Sample {idx+1}', fontsize=10, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Bottom row: Error map
        error_vis = axes[1, idx].imshow(error_map, cmap='RdYlGn_r', vmin=0, vmax=1)
        accuracy = (correct.sum() / correct.size) * 100
        axes[1, idx].set_title(f'Accuracy: {accuracy:.1f}%', fontsize=10)
        axes[1, idx].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(error_vis, ax=axes, orientation='horizontal', pad=0.02, fraction=0.03)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Correct', 'Wrong'])
    
    plt.suptitle('Prediction Error Maps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_per_class_predictions(predictions: List[Dict], output_path: str, class_idx: int):
    """Plot predictions for a specific class showing areas where class appears."""
    class_name = CLASS_NAMES[class_idx]
    
    # Find samples where class appears in GT
    samples_with_class = []
    for pred in predictions:
        if (pred['mask'] == class_idx).sum() > 0:
            samples_with_class.append(pred)
    
    if len(samples_with_class) == 0:
        print(f"⚠️  No samples found with class '{class_name}'")
        return
    
    num_samples = min(len(samples_with_class), 4)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        pred = samples_with_class[idx]
        image = pred['image'].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        mask = pred['mask']
        pred_mask = pred['pred']
        
        # Binary masks for target class
        gt_class_mask = (mask == class_idx).astype(float)
        pred_class_mask = (pred_mask == class_idx).astype(float)
        
        # Compute IoU for this class in this sample
        intersection = (gt_class_mask * pred_class_mask).sum()
        union = ((gt_class_mask + pred_class_mask) > 0).sum()
        iou = intersection / union if union > 0 else 0
        
        # Column 1: Image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title('Image' if idx == 0 else '', fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Column 2: GT for this class
        axes[idx, 1].imshow(gt_class_mask, cmap='Reds', vmin=0, vmax=1)
        axes[idx, 1].set_title(f'GT: {class_name}' if idx == 0 else '', fontsize=10, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Column 3: Pred for this class
        axes[idx, 2].imshow(pred_class_mask, cmap='Greens', vmin=0, vmax=1)
        axes[idx, 2].set_title(f'Pred: {class_name}' if idx == 0 else '', fontsize=10, fontweight='bold')
        axes[idx, 2].axis('off')
        
        # Column 4: Overlay showing TP/FP/FN
        # TP=green, FP=blue, FN=red
        overlay = np.zeros((*gt_class_mask.shape, 3))
        tp = (gt_class_mask * pred_class_mask).astype(bool)
        fp = ((1 - gt_class_mask) * pred_class_mask).astype(bool)
        fn = (gt_class_mask * (1 - pred_class_mask)).astype(bool)
        
        overlay[tp] = [0, 1, 0]  # Green: True Positive
        overlay[fp] = [0, 0, 1]  # Blue: False Positive
        overlay[fn] = [1, 0, 0]  # Red: False Negative
        
        axes[idx, 3].imshow(image)
        axes[idx, 3].imshow(overlay, alpha=0.5)
        axes[idx, 3].set_title(f'IoU: {iou:.3f}' if idx == 0 else f'{iou:.3f}', fontsize=10)
        axes[idx, 3].axis('off')
    
    plt.suptitle(f'Class-specific Predictions: {class_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def visualize_model(config_path: str, checkpoint_path: str = None, num_samples: int = 10, 
                   output_dir: str = None):
    """Generate all visualizations for a model."""
    
    # Load model
    model, config, device = load_model_and_checkpoint(config_path, checkpoint_path)
    
    # Setup output directory
    if output_dir is None:
        # Use checkpoint parent directory (models/{name}/)
        output_dir = checkpoint_path.parent / 'visualizations'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}\n")
    
    # Create lightweight dataloader (on-demand loading, no RAM preload)
    print(f"📊 Creating validation dataloader (lazy loading, only {num_samples} samples needed)...")
    
    # Get validation split using normal dataset (no preload)
    data_root = config['data']['data_dir']
    image_size = config['data']['image_size']
    use_augmented = config['data'].get('use_augmented', False)
    
    # Setup data path
    from pathlib import Path
    data_path = Path(data_root)
    
    # Check if augmented dataset exists
    augmented_path = data_path.parent / 'artefact_augmented'
    if use_augmented and augmented_path.exists():
        data_path = augmented_path
    
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
    
    # Train/val split (same as training, seed=42)
    np.random.seed(42)
    indices = np.random.permutation(len(image_paths))
    split_idx = int(len(indices) * 0.8)
    val_indices = indices[split_idx:]
    
    val_images = [str(image_paths[i]) for i in val_indices]
    val_masks = [str(mask_paths[i]) for i in val_indices]
    
    print(f"  Found {len(val_images)} validation images")
    
    # Take only subset we need for visualization
    subset_count = min(num_samples * 4, len(val_images))  # Take 4x extra for variety
    val_images_subset = val_images[:subset_count]
    val_masks_subset = val_masks[:subset_count]
    
    # Create dataset with on-demand loading
    transforms = get_transforms(image_size=image_size, is_train=False)  # Validation transforms
    val_dataset = ArtefactDataset(
        val_images_subset,
        val_masks_subset,
        transform=transforms
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,  # Larger batch for GPU
        shuffle=False,
        num_workers=4,
        pin_memory=True  # Speed up GPU transfer
    )
    
    print(f"✅ Dataloader ready: {len(val_dataset)} samples (loading on-demand, no preload)\n")
    
    # Collect predictions
    print(f"\n🔍 Collecting {num_samples} prediction samples...")
    predictions = collect_predictions(model, val_loader, device, num_samples)
    print(f"✅ Collected {len(predictions)} samples\n")
    
    # Generate visualizations
    print("🎨 Generating visualizations...\n")
    
    # 1. Prediction grid
    plot_prediction_grid(
        predictions,
        os.path.join(output_dir, 'prediction_grid.png'),
        f"POC-5.9-v2 Predictions: {config['model']['encoder_name']}"
    )
    
    # 2. Class distribution
    plot_class_distribution(
        predictions,
        os.path.join(output_dir, 'class_distribution.png')
    )
    
    # 3. Error maps
    plot_error_map(
        predictions,
        os.path.join(output_dir, 'error_maps.png')
    )
    
    # 4. Per-class visualizations for top classes
    top_classes = [0, 1, 2, 7, 8, 15]  # Clean, Material_loss, Peel, Discolouration, Scratches, Other_damage
    for cls_idx in top_classes:
        plot_per_class_predictions(
            predictions,
            os.path.join(output_dir, f'class_{cls_idx:02d}_{CLASS_NAMES[cls_idx]}.png'),
            cls_idx
        )
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='POC-5.9-v2 Visualization')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (auto-detect if not provided)')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of samples to visualize (default: 20)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (auto-detect if not provided)')
    parser.add_argument('--all', action='store_true', help='Visualize all 3 models')
    
    args = parser.parse_args()
    
    if args.all or args.config is None:
        configs = [
            'configs/convnext_tiny.yaml',
            'configs/segformer_b3.yaml',
            'configs/maxvit_tiny.yaml'
        ]
        
        for config_path in configs:
            print("\n" + "="*100)
            print(f"Visualizing: {config_path}")
            print("="*100)
            try:
                visualize_model(config_path, None, args.num_samples, args.output_dir)
            except Exception as e:
                print(f"❌ Error visualizing {config_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
    else:
        visualize_model(args.config, args.checkpoint, args.num_samples, args.output_dir)


if __name__ == '__main__':
    main()
