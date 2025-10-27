"""
Evaluation script for POC-5.5: Hierarchical Multiclass Segmentation
Evaluates trained models on validation set with hierarchical metrics and visualizations.

Usage:
    python evaluate.py --config configs/convnext_tiny.yaml --checkpoint logs/convnext_tiny/checkpoints/best_model.pth
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from models.hierarchical_upernet import build_hierarchical_model
from dataset_multiclass import ArtefactMulticlassDataset, get_multiclass_transforms


# ARTeFACT class names (16 classes)
CLASS_NAMES = [
    'Clean',              # 0
    'Material loss',      # 1
    'Peel',              # 2
    'Cracks',            # 3
    'Structural defects', # 4
    'Dirt spots',        # 5
    'Stains',            # 6
    'Discolouration',    # 7
    'Scratches',         # 8
    'Burn marks',        # 9
    'Hairs',             # 10
    'Dust spots',        # 11
    'Lightleak',         # 12
    'Fading',            # 13
    'Blur',              # 14
    'Other damage'       # 15
]

COARSE_NAMES = [
    'Structural',        # Group 0: Material loss, Peel, Cracks, Structural defects
    'Surface',           # Group 1: Dirt spots, Stains, Hairs, Dust spots
    'Color',             # Group 2: Discolouration, Burn marks, Fading
    'Optical'            # Group 3: Scratches, Lightleak, Blur, Other damage
]

BINARY_NAMES = ['Clean', 'Damage']


class HierarchicalMetrics:
    """Calculate hierarchical segmentation metrics for all 3 heads"""
    
    def __init__(self, num_classes_fine=16, num_classes_coarse=4, num_classes_binary=2, ignore_index=255):
        self.num_classes_fine = num_classes_fine
        self.num_classes_coarse = num_classes_coarse
        self.num_classes_binary = num_classes_binary
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all confusion matrices"""
        self.cm_fine = np.zeros((self.num_classes_fine, self.num_classes_fine), dtype=np.int64)
        self.cm_coarse = np.zeros((self.num_classes_coarse, self.num_classes_coarse), dtype=np.int64)
        self.cm_binary = np.zeros((self.num_classes_binary, self.num_classes_binary), dtype=np.int64)
    
    def update(self, preds_dict, targets_dict):
        """Update all confusion matrices
        
        Args:
            preds_dict: {'binary': Tensor, 'coarse': Tensor, 'fine': Tensor}
            targets_dict: {'binary': Tensor, 'coarse': Tensor, 'fine': Tensor}
        """
        # Update binary
        self._update_single(preds_dict['binary'], targets_dict['binary'], self.cm_binary, self.num_classes_binary)
        
        # Update coarse
        self._update_single(preds_dict['coarse'], targets_dict['coarse'], self.cm_coarse, self.num_classes_coarse)
        
        # Update fine
        self._update_single(preds_dict['fine'], targets_dict['fine'], self.cm_fine, self.num_classes_fine)
    
    def _update_single(self, preds, targets, cm, num_classes):
        """Update single confusion matrix"""
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # Remove ignore index
        mask = targets != self.ignore_index
        preds = preds[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        for pred, target in zip(preds, targets):
            if 0 <= target < num_classes and 0 <= pred < num_classes:
                cm[target, pred] += 1
    
    def compute_metrics(self, cm, num_classes):
        """Compute IoU, F1, Precision, Recall from confusion matrix"""
        iou_per_class = []
        f1_per_class = []
        precision_per_class = []
        recall_per_class = []
        
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            # IoU
            denominator = tp + fp + fn
            iou = tp / denominator if denominator > 0 else 0.0
            iou_per_class.append(iou)
            
            # Precision, Recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)
        
        return {
            'iou_per_class': np.array(iou_per_class),
            'miou': np.mean(iou_per_class),
            'f1_per_class': np.array(f1_per_class),
            'mf1': np.mean(f1_per_class),
            'precision_per_class': np.array(precision_per_class),
            'recall_per_class': np.array(recall_per_class),
        }
    
    def get_all_metrics(self):
        """Get metrics for all 3 heads"""
        metrics = {
            'binary': self.compute_metrics(self.cm_binary, self.num_classes_binary),
            'coarse': self.compute_metrics(self.cm_coarse, self.num_classes_coarse),
            'fine': self.compute_metrics(self.cm_fine, self.num_classes_fine),
        }
        
        # Add confusion matrices
        metrics['binary']['confusion_matrix'] = self.cm_binary.tolist()
        metrics['coarse']['confusion_matrix'] = self.cm_coarse.tolist()
        metrics['fine']['confusion_matrix'] = self.cm_fine.tolist()
        
        return metrics


def load_model(config_path, checkpoint_path, device):
    """Load trained hierarchical model from checkpoint"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build model
    print(f"Building hierarchical model from config...")
    model = build_hierarchical_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    metrics = checkpoint.get('metrics', {})
    print(f"  Checkpoint from epoch {epoch}")
    print(f"  Val metrics: mIoU_fine={metrics.get('mIoU_fine', 'N/A'):.4f}")
    
    model = model.to(device)
    model.eval()
    
    return model, config


def evaluate_model(model, dataloader, device):
    """Evaluate hierarchical model on dataloader"""
    metrics_tracker = HierarchicalMetrics()
    predictions = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for images, targets_dict in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            targets_dict = {k: v.to(device) for k, v in targets_dict.items()}
            
            # Forward pass (get all 3 heads)
            preds_dict = model(images, return_all_heads=True)
            
            # Get predicted classes
            preds_binary = torch.argmax(preds_dict['binary'], dim=1)
            preds_coarse = torch.argmax(preds_dict['coarse'], dim=1)
            preds_fine = torch.argmax(preds_dict['fine'], dim=1)
            
            preds = {
                'binary': preds_binary,
                'coarse': preds_coarse,
                'fine': preds_fine
            }
            
            # Update metrics
            metrics_tracker.update(preds, targets_dict)
            
            # Store some predictions for visualization
            if len(predictions) < 10:
                for img, mask_bin, mask_coarse, mask_fine, pred_bin, pred_coarse, pred_fine in zip(
                    images, targets_dict['binary'], targets_dict['coarse'], targets_dict['fine'],
                    preds_binary, preds_coarse, preds_fine
                ):
                    predictions.append({
                        'image': img.cpu().numpy(),
                        'mask_binary': mask_bin.cpu().numpy(),
                        'mask_coarse': mask_coarse.cpu().numpy(),
                        'mask_fine': mask_fine.cpu().numpy(),
                        'pred_binary': pred_bin.cpu().numpy(),
                        'pred_coarse': pred_coarse.cpu().numpy(),
                        'pred_fine': pred_fine.cpu().numpy(),
                    })
    
    # Compute all metrics
    metrics = metrics_tracker.get_all_metrics()
    
    return metrics, predictions


def plot_confusion_matrix(cm, class_names, output_path, title="Confusion Matrix"):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.75)))
    sns.heatmap(
        cm,
        annot=True if len(class_names) <= 4 else False,  # Only annotate if <= 4 classes
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path.name}")


def plot_hierarchical_predictions(predictions, output_dir, num_samples=6):
    """Plot hierarchical prediction visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    samples = predictions[:num_samples]
    
    for idx, pred in enumerate(samples):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Denormalize image
        image = pred['image'].transpose(1, 2, 0)  # CHW -> HWC
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        # Row 1: Ground Truth
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(pred['mask_binary'], cmap='tab10', vmin=0, vmax=1)
        axes[0, 1].set_title('GT Binary (Clean/Damage)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred['mask_coarse'], cmap='tab10', vmin=0, vmax=3)
        axes[0, 2].set_title('GT Coarse (4 Groups)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(pred['mask_fine'], cmap='tab20', vmin=0, vmax=15)
        axes[0, 3].set_title('GT Fine (16 Classes)', fontsize=12, fontweight='bold')
        axes[0, 3].axis('off')
        
        # Row 2: Predictions
        axes[1, 0].axis('off')  # Empty
        
        axes[1, 1].imshow(pred['pred_binary'], cmap='tab10', vmin=0, vmax=1)
        axes[1, 1].set_title('Pred Binary', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(pred['pred_coarse'], cmap='tab10', vmin=0, vmax=3)
        axes[1, 2].set_title('Pred Coarse', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(pred['pred_fine'], cmap='tab20', vmin=0, vmax=15)
        axes[1, 3].set_title('Pred Fine', fontsize=12, fontweight='bold')
        axes[1, 3].axis('off')
        
        plt.suptitle(f'Hierarchical Predictions - Sample {idx+1}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'hierarchical_pred_{idx+1}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved {len(samples)} hierarchical visualizations to {output_dir}")


def save_metrics(metrics, output_path):
    """Save metrics to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    metrics_serializable = convert_numpy(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"✅ Saved metrics: {output_path}")


def print_metrics_summary(metrics):
    """Print hierarchical metrics summary"""
    print("\n" + "="*100)
    print("HIERARCHICAL EVALUATION RESULTS")
    print("="*100)
    
    # Binary Head
    print("\n📊 BINARY HEAD (Clean vs Damage)")
    print("-" * 100)
    print(f"  mIoU:  {metrics['binary']['miou']:.4f}")
    print(f"  mF1:   {metrics['binary']['mf1']:.4f}")
    for i, name in enumerate(BINARY_NAMES):
        print(f"  {name:15s}: IoU={metrics['binary']['iou_per_class'][i]:.4f}, F1={metrics['binary']['f1_per_class'][i]:.4f}")
    
    # Coarse Head
    print("\n📊 COARSE HEAD (4 Damage Groups)")
    print("-" * 100)
    print(f"  mIoU:  {metrics['coarse']['miou']:.4f}")
    print(f"  mF1:   {metrics['coarse']['mf1']:.4f}")
    for i, name in enumerate(COARSE_NAMES):
        print(f"  {name:15s}: IoU={metrics['coarse']['iou_per_class'][i]:.4f}, F1={metrics['coarse']['f1_per_class'][i]:.4f}")
    
    # Fine Head
    print("\n📊 FINE HEAD (16 Classes)")
    print("-" * 100)
    print(f"  mIoU:  {metrics['fine']['miou']:.4f}")
    print(f"  mF1:   {metrics['fine']['mf1']:.4f}")
    print("\n  Per-class results:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {i:2d}. {name:20s}: IoU={metrics['fine']['iou_per_class'][i]:.4f}, F1={metrics['fine']['f1_per_class'][i]:.4f}")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description='Evaluate hierarchical model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    if args.output is None:
        exp_name = config.get('logging', {}).get('experiment_name', 'poc55_experiment')
        args.output = f"logs/{exp_name}/evaluation"
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Load model
    model, model_config = load_model(args.config, args.checkpoint, device)
    
    # Load validation dataset
    data_root = Path(config['data']['root'])
    images_dir = data_root / 'images'
    annotations_dir = data_root / 'annotations'
    
    image_files = sorted(images_dir.glob('*.png'))
    mask_files = sorted(annotations_dir.glob('*.png'))
    
    image_paths = [str(f) for f in image_files]
    mask_paths = [str(f) for f in mask_files]
    
    # Train/val split (same as training)
    train_split = config['data'].get('train_val_split', 0.8)
    split_idx = int(len(image_paths) * train_split)
    
    val_images = image_paths[split_idx:]
    val_masks = mask_paths[split_idx:]
    
    print(f"\nValidation set: {len(val_images)} images")
    
    # Create validation dataset
    val_dataset = ArtefactMulticlassDataset(
        val_images, val_masks,
        transform=get_multiclass_transforms(config, mode='val'),
        ignore_index=config['data'].get('ignore_index', 255)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    # Evaluate
    metrics, predictions = evaluate_model(model, val_loader, device)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save metrics
    metrics_path = os.path.join(args.output, 'metrics.json')
    save_metrics(metrics, metrics_path)
    
    # Plot confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(
        np.array(metrics['binary']['confusion_matrix']),
        BINARY_NAMES,
        Path(args.output) / 'confusion_matrix_binary.png',
        "Binary Head - Confusion Matrix"
    )
    
    plot_confusion_matrix(
        np.array(metrics['coarse']['confusion_matrix']),
        COARSE_NAMES,
        Path(args.output) / 'confusion_matrix_coarse.png',
        "Coarse Head - Confusion Matrix"
    )
    
    plot_confusion_matrix(
        np.array(metrics['fine']['confusion_matrix']),
        CLASS_NAMES,
        Path(args.output) / 'confusion_matrix_fine.png',
        "Fine Head - Confusion Matrix"
    )
    
    # Plot predictions
    pred_dir = os.path.join(args.output, 'predictions')
    plot_hierarchical_predictions(predictions, pred_dir, num_samples=6)
    
    print(f"\n✅ Evaluation complete! Results saved to: {args.output}")


if __name__ == '__main__':
    main()
