"""
POC-5.8 Evaluation Script

Evaluates trained DeepLabV3+ models on validation set.
Generates:
- Per-class IoU metrics
- Confusion matrix
- Prediction visualizations
- JSON metrics export

Usage:
    python evaluate.py --config ../configs/resnet50.yaml
    python evaluate.py --config ../configs/convnext_tiny.yaml
    python evaluate.py --config ../configs/swin_tiny.yaml
    python evaluate.py --config ../configs/maxvit_tiny.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import segmentation_models_pytorch as smp
from preload_dataset import create_preloaded_dataloaders

# ARTeFACT 16 classes
CLASS_NAMES = [
    'Clean', 'Material_loss', 'Peel', 'Cracks', 'Structural_defects',
    'Dirt_spots', 'Stains', 'Discolouration', 'Scratches', 'Burn_marks',
    'Hairs', 'Dust_spots', 'Lightleak', 'Fading', 'Blur', 'Other_damage'
]


def load_config(config_path: str) -> Dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: Dict) -> nn.Module:
    """Create SMP model from config."""
    arch = config['model']['architecture']
    encoder = config['model']['encoder_name']
    
    model_class = getattr(smp, arch)
    model = model_class(
        encoder_name=encoder,
        encoder_weights=None,  # Load from checkpoint
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes'],
        activation=config['model']['activation']
    )
    return model


def compute_iou_per_class(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:
    """
    Compute IoU for each class.
    
    Args:
        preds: (N, H, W) predicted class indices
        targets: (N, H, W) ground truth class indices
        num_classes: Number of classes
    
    Returns:
        iou_per_class: (num_classes,) IoU for each class
    """
    ious = []
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union == 0:
            iou = float('nan')  # Class not present
        else:
            iou = (intersection / union).item()
        
        ious.append(iou)
    
    return np.array(ious)


def compute_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    preds_np = preds.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()
    
    for pred, target in zip(preds_np, targets_np):
        cm[target, pred] += 1
    
    return cm


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on validation set.
    
    Returns:
        mean_iou: Mean IoU across all classes
        iou_per_class: IoU for each class
        confusion_matrix: Confusion matrix
    """
    model.eval()
    
    all_ious = []
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for images, masks in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        
        # Compute IoU
        batch_ious = compute_iou_per_class(preds, masks, num_classes)
        all_ious.append(batch_ious)
        
        # Update confusion matrix
        batch_cm = compute_confusion_matrix(preds, masks, num_classes)
        confusion_matrix += batch_cm
    
    # Aggregate IoUs (ignore NaN)
    all_ious = np.array(all_ious)  # (num_batches, num_classes)
    iou_per_class = np.nanmean(all_ious, axis=0)
    mean_iou = np.nanmean(iou_per_class)
    
    return mean_iou, iou_per_class, confusion_matrix


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(14, 12))
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Normalized by Row)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved: {save_path}")


def plot_iou_per_class(iou_per_class: np.ndarray, class_names: list, save_path: Path):
    """Plot per-class IoU bar chart."""
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    bars = plt.bar(x, iou_per_class * 100, color='steelblue', alpha=0.8)
    
    # Color code by performance
    for i, bar in enumerate(bars):
        iou = iou_per_class[i]
        if np.isnan(iou):
            bar.set_color('gray')
        elif iou > 0.5:
            bar.set_color('green')
        elif iou > 0.3:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylabel('IoU (%)')
    plt.title('Per-Class IoU Performance')
    plt.axhline(y=np.nanmean(iou_per_class) * 100, color='black', linestyle='--', label=f'Mean: {np.nanmean(iou_per_class)*100:.2f}%')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Per-class IoU plot saved: {save_path}")


def save_metrics(
    mean_iou: float,
    iou_per_class: np.ndarray,
    class_names: list,
    save_path: Path
):
    """Save metrics to JSON."""
    metrics = {
        'mean_iou': float(mean_iou),
        'per_class_iou': {
            name: float(iou) if not np.isnan(iou) else None
            for name, iou in zip(class_names, iou_per_class)
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='POC-5.8 Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (default: logs/{model}/best_model.pth)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine checkpoint path
    if args.checkpoint is None:
        model_name = Path(args.config).stem
        checkpoint_path = Path(f'../logs/{model_name}/best_model.pth')
    else:
        checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = config['model']['classes']
    
    print("=" * 60)
    print(f"POC-5.8 Evaluation: {Path(args.config).stem}")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Classes: {num_classes}")
    print()
    
    # Create dataloader (validation only)
    _, val_loader = create_preloaded_dataloaders(
        data_root=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        use_augmented=config['data'].get('use_augmented', False),
        preload_to_gpu=config['data'].get('preload_to_gpu', False)
    )
    
    print(f"Validation samples: {len(val_loader.dataset)}")
    print()
    
    # Load model
    print("Loading model...")
    model = create_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print()
    
    # Evaluate
    print("Evaluating...")
    mean_iou, iou_per_class, confusion_matrix = evaluate(
        model, val_loader, device, num_classes
    )
    
    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Mean IoU: {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print()
    print("Per-class IoU:")
    for name, iou in zip(CLASS_NAMES, iou_per_class):
        if np.isnan(iou):
            print(f"  {name:20s}: N/A (not present)")
        else:
            print(f"  {name:20s}: {iou:.4f} ({iou*100:.2f}%)")
    print("=" * 60)
    
    # Save outputs
    model_name = Path(args.config).stem
    output_dir = Path(f'../logs/{model_name}/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    save_metrics(mean_iou, iou_per_class, CLASS_NAMES, output_dir / 'metrics.json')
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion_matrix, CLASS_NAMES, output_dir / 'confusion_matrix.png')
    
    # Plot per-class IoU
    plot_iou_per_class(iou_per_class, CLASS_NAMES, output_dir / 'per_class_iou.png')
    
    print()
    print(f"📁 All outputs saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
