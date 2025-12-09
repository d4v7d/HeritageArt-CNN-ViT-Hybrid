"""
POC-5.9-v2 Evaluation Script

Evaluates trained models on validation set.
Generates:
- Per-class IoU metrics with precision/recall/F1
- Confusion matrix (normalized)
- Per-class IoU bar chart
- JSON metrics export
- Model comparison table

Usage:
    python evaluate.py --config ../configs/convnext_tiny.yaml
    python evaluate.py --config ../configs/segformer_b3.yaml
    python evaluate.py --config ../configs/maxvit_tiny.yaml
    python evaluate.py --all  # Evaluate all 3 models
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
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from model_factory import create_model
from preload_dataset import create_val_only_dataloader

# ARTeFACT 16 classes (official from danielaivanova/damaged-media)
CLASS_NAMES = [
    'Clean', 'Material_loss', 'Peel', 'Dust', 'Scratch',
    'Hair', 'Dirt', 'Fold', 'Writing', 'Cracks',
    'Staining', 'Stamp', 'Sticker', 'Puncture', 'Burn_marks', 'Lightleak'
]


def load_config(config_path: str) -> Dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
        
        if union.item() == 0:
            iou = float('nan')  # Class not present
        else:
            iou = (intersection / union).item()
        
        ious.append(iou)
    
    return np.array(ious)


def compute_precision_recall_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision, recall, and F1 score per class."""
    precision = []
    recall = []
    f1 = []
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        
        tp = (pred_mask & target_mask).sum().float()
        fp = (pred_mask & ~target_mask).sum().float()
        fn = (~pred_mask & target_mask).sum().float()
        
        prec = (tp / (tp + fp)).item() if (tp + fp).item() > 0 else float('nan')
        rec = (tp / (tp + fn)).item() if (tp + fn).item() > 0 else float('nan')
        
        if not (np.isnan(prec) or np.isnan(rec)):
            f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            f1_score = float('nan')
        
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)
    
    return np.array(precision), np.array(recall), np.array(f1)


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
) -> Dict:
    """
    Evaluate model on validation set.
    
    Returns:
        Dict with:
            - mean_iou: Mean IoU across all classes
            - iou_per_class: IoU for each class
            - precision_per_class: Precision for each class
            - recall_per_class: Recall for each class
            - f1_per_class: F1 score for each class
            - confusion_matrix: Confusion matrix
            - inference_time: Average inference time per image (ms)
    """
    model.eval()
    
    all_ious = []
    all_precision = []
    all_recall = []
    all_f1 = []
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    inference_times = []
    
    for images, masks in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward with timing
        start_time = time.time()
        logits = model(images)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = (time.time() - start_time) * 1000 / len(images)  # ms per image
        inference_times.append(inference_time)
        
        preds = torch.argmax(logits, dim=1)
        
        # Compute metrics
        batch_ious = compute_iou_per_class(preds, masks, num_classes)
        batch_prec, batch_rec, batch_f1 = compute_precision_recall_f1(preds, masks, num_classes)
        
        all_ious.append(batch_ious)
        all_precision.append(batch_prec)
        all_recall.append(batch_rec)
        all_f1.append(batch_f1)
        
        # Update confusion matrix
        batch_cm = compute_confusion_matrix(preds, masks, num_classes)
        confusion_matrix += batch_cm
    
    # Aggregate metrics (ignore NaN)
    all_ious = np.array(all_ious)
    all_precision = np.array(all_precision)
    all_recall = np.array(all_recall)
    all_f1 = np.array(all_f1)
    
    iou_per_class = np.nanmean(all_ious, axis=0)
    precision_per_class = np.nanmean(all_precision, axis=0)
    recall_per_class = np.nanmean(all_recall, axis=0)
    f1_per_class = np.nanmean(all_f1, axis=0)
    
    mean_iou = np.nanmean(iou_per_class)
    avg_inference_time = np.mean(inference_times)
    
    return {
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': confusion_matrix,
        'inference_time': avg_inference_time
    }


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
    results: Dict,
    class_names: list,
    save_path: Path
):
    """Save detailed metrics to JSON."""
    metrics = {
        'mean_iou': float(results['mean_iou']),
        'inference_time_ms': float(results['inference_time']),
        'per_class': {}
    }
    
    for i, name in enumerate(class_names):
        metrics['per_class'][name] = {
            'iou': float(results['iou_per_class'][i]) if not np.isnan(results['iou_per_class'][i]) else None,
            'precision': float(results['precision_per_class'][i]) if not np.isnan(results['precision_per_class'][i]) else None,
            'recall': float(results['recall_per_class'][i]) if not np.isnan(results['recall_per_class'][i]) else None,
            'f1': float(results['f1_per_class'][i]) if not np.isnan(results['f1_per_class'][i]) else None
        }
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='POC-5.9-v2 Evaluation')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (auto-detects if not provided)')
    parser.add_argument('--all', action='store_true', help='Evaluate all 3 models (ConvNeXt, SegFormer, MaxViT)')
    args = parser.parse_args()
    
    if args.all:
        # Evaluate all 3 models
        configs = [
            'configs/convnext_tiny.yaml',
            'configs/segformer_b3.yaml',
            'configs/maxvit_tiny.yaml'
        ]
        all_results = []
        
        for config_path in configs:
            config_path = Path(config_path)
            if not config_path.exists():
                # Try from script parent directory
                config_path = Path(__file__).parent.parent / config_path
            
            print("\n" + "=" * 80)
            print(f"Evaluating: {config_path.stem}")
            print("=" * 80)
            
            try:
                result = evaluate_single_model(str(config_path), None)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error evaluating {config_path.stem}: {e}")
                continue
        
        # Print comparison table
        if all_results:
            print("\n" + "=" * 80)
            print("COMPARISON TABLE")
            print("=" * 80)
            print(f"{'Model':<20} {'mIoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10} {'Inf Time (ms)':>15}")
            print("-" * 80)
            
            for res in all_results:
                print(f"{res['model_name']:<20} {res['mean_iou']*100:>9.2f}% "
                      f"{np.nanmean(res['precision_per_class'])*100:>11.2f}% "
                      f"{np.nanmean(res['recall_per_class'])*100:>9.2f}% "
                      f"{np.nanmean(res['f1_per_class'])*100:>9.2f}% "
                      f"{res['inference_time']:>14.2f}")
            print("=" * 80)
            
            # Save comparison to logs/results
            # Use Path(__file__) to get the script location and build correct relative path
            script_dir = Path(__file__).parent
            comparison_file = script_dir.parent / 'logs' / 'results' / 'model_comparison.json'
            comparison_file.parent.mkdir(parents=True, exist_ok=True)
            
            comparison_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'models': [
                    {
                        'name': res['model_name'],
                        'mean_iou': float(res['mean_iou']),
                        'mean_precision': float(np.nanmean(res['precision_per_class'])),
                        'mean_recall': float(np.nanmean(res['recall_per_class'])),
                        'mean_f1': float(np.nanmean(res['f1_per_class'])),
                        'inference_time_ms': float(res['inference_time'])
                    }
                    for res in all_results
                ]
            }
            
            try:
                with open(comparison_file, 'w') as f:
                    json.dump(comparison_data, f, indent=2)
                print(f"\n✅ Comparison saved: {comparison_file.resolve()}")
            except Exception as e:
                print(f"\n❌ Error saving comparison: {e}")
                print(f"   Attempted path: {comparison_file.resolve()}")
        
    else:
        # Single model evaluation
        if args.config is None:
            print("❌ Error: --config required (or use --all to evaluate all models)")
            sys.exit(1)
        
        evaluate_single_model(args.config, args.checkpoint)


def evaluate_single_model(config_path: str, checkpoint_path: str = None) -> Dict:
    """Evaluate a single model and return results."""
    # Load config
    config = load_config(config_path)
    
    # Determine checkpoint path
    if checkpoint_path is None:
        encoder_name = config['model']['encoder_name']
        config_name = Path(config_path).stem
        # Map config names to model directories
        model_dirs = {
            'convnext_tiny': 'convnext_tiny',
            'segformer_b3': 'segformer_b3',
            'maxvit_tiny': 'maxvit_tiny'
        }
        model_dir = model_dirs.get(config_name, config_name)
        # Try different possible paths
        possible_paths = [
            Path(f'logs/models/{model_dir}/best_model.pth'),
            Path(f'logs/Unet_{encoder_name}/best_model.pth'),
            Path(f'logs/{encoder_name}/best_model.pth'),
        ]
        
        checkpoint_path = None
        for p in possible_paths:
            full_path = Path(__file__).parent.parent / p
            if full_path.exists():
                checkpoint_path = full_path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found. Tried: {[str(p) for p in possible_paths]}")
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = config['model']['classes']
    model_name = config['model']['encoder_name']
    
    print(f"Config: {Path(config_path).name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Classes: {num_classes}")
    print()
    
    # Create dataloader (validation only - optimized)
    val_loader = create_val_only_dataloader(
        data_root=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['dataloader'].get('num_workers', 4),
        use_augmented=config['data'].get('use_augmented', False)
    )
    
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Validation batches: {len(val_loader)}")
    print()
    
    # Load model
    print("Loading model...")
    model = create_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, best mIoU: {checkpoint.get('best_miou', '?'):.4f}")
    print()
    
    # Evaluate
    print("Evaluating...")
    results = evaluate(model, val_loader, device, num_classes)
    
    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Mean IoU: {results['mean_iou']:.4f} ({results['mean_iou']*100:.2f}%)")
    print(f"Inference Time: {results['inference_time']:.2f} ms/image")
    print()
    print(f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 60)
    for i, name in enumerate(CLASS_NAMES):
        iou = results['iou_per_class'][i]
        prec = results['precision_per_class'][i]
        rec = results['recall_per_class'][i]
        f1 = results['f1_per_class'][i]
        
        if np.isnan(iou):
            print(f"{name:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        else:
            print(f"{name:<20} {iou*100:>7.2f}% {prec*100:>7.2f}% {rec*100:>7.2f}% {f1*100:>7.2f}%")
    print("=" * 60)
    
    # Save outputs
    output_dir = checkpoint_path.parent / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    save_metrics(results, CLASS_NAMES, output_dir / 'metrics.json')
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], CLASS_NAMES, output_dir / 'confusion_matrix.png')
    
    # Plot per-class IoU
    plot_iou_per_class(results['iou_per_class'], CLASS_NAMES, output_dir / 'per_class_iou.png')
    
    print()
    print(f"📁 All outputs saved to: {output_dir}")
    print()
    
    # Return results for comparison
    return {
        'model_name': model_name,
        **results
    }


if __name__ == '__main__':
    main()
