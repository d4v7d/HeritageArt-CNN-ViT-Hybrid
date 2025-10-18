"""
Evaluation script for POC-5: Multi-backbone ARTeFACT Damage Detection
Evaluates trained models on validation set with detailed metrics and visualizations.

Usage:
    python evaluate.py --config configs/convnext_tiny_upernet.yaml --checkpoint logs/.../best_model.pth
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import custom modules
from dataset import ArtefactDataset
from models.model_factory import UPerNetModel


class SegmentationMetrics:
    """Calculate segmentation metrics: IoU, F1, Precision, Recall, Accuracy"""
    
    def __init__(self, num_classes=2, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, preds, targets):
        """Update confusion matrix with new predictions
        
        Args:
            preds: (B, H, W) - predicted class indices
            targets: (B, H, W) - ground truth class indices
        """
        # Flatten
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # Remove ignore index
        mask = targets != self.ignore_index
        preds = preds[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        for pred, target in zip(preds, targets):
            if 0 <= target < self.num_classes and 0 <= pred < self.num_classes:
                self.confusion_matrix[target, pred] += 1
    
    def compute_iou(self):
        """Compute IoU per class and mean IoU"""
        iou_per_class = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            denominator = tp + fp + fn
            if denominator == 0:
                iou = 0.0
            else:
                iou = tp / denominator
            iou_per_class.append(iou)
        
        return np.array(iou_per_class), np.mean(iou_per_class)
    
    def compute_f1(self):
        """Compute F1 score per class and mean F1"""
        f1_per_class = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            f1_per_class.append(f1)
        
        return np.array(f1_per_class), np.mean(f1_per_class)
    
    def compute_accuracy(self):
        """Compute pixel accuracy"""
        tp = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return tp / total if total > 0 else 0.0
    
    def compute_precision_recall(self):
        """Compute precision and recall per class"""
        precision_per_class = []
        recall_per_class = []
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
        
        return np.array(precision_per_class), np.array(recall_per_class)
    
    def get_confusion_matrix(self):
        """Return confusion matrix"""
        return self.confusion_matrix


def get_val_transforms(img_size=512):
    """Get validation transforms: Resize + Normalize + ToTensor
    
    Args:
        img_size: Target image size (default 512)
    
    Returns:
        albumentations.Compose: Validation transforms
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def load_model(config_path, checkpoint_path, device):
    """Load trained model from checkpoint
    
    Args:
        config_path: Path to config YAML
        checkpoint_path: Path to model checkpoint
        device: torch device
    
    Returns:
        model: Loaded model
        config: Configuration dict
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model config
    model_cfg = config['model']
    encoder_name = model_cfg['encoder']
    num_classes = model_cfg['classes']
    
    # Get UPerNet config
    upernet_cfg = model_cfg.get('upernet', {})
    ppm_pool_scales = tuple(upernet_cfg.get('ppm_pool_scales', [1, 2, 3, 6]))
    fpn_out_channels = upernet_cfg.get('fpn_out_channels', 256)
    dropout = upernet_cfg.get('dropout', 0.1)
    
    # Encoder-specific channel configurations
    ENCODER_CHANNELS = {
        'convnext_tiny': [96, 192, 384, 768],
        'swin_tiny_patch4_window7_224': [96, 192, 384, 768],
        'maxvit_tiny_tf_512': [64, 128, 256, 512],
    }
    
    # Get encoder channels
    in_channels_list = ENCODER_CHANNELS.get(encoder_name)
    if in_channels_list is None:
        raise ValueError(f"Unknown encoder: {encoder_name}. Add to ENCODER_CHANNELS dict.")
    
    # Build model
    print(f"Building model: {model_cfg['name']}")
    print(f"  Encoder: {encoder_name}")
    print(f"  Classes: {num_classes}")
    
    model = UPerNetModel(
        encoder_name=encoder_name,
        encoder_weights=None,  # Not loading pretrained, using checkpoint
        in_channels_list=in_channels_list,
        out_channels=fpn_out_channels,
        ppm_pool_scales=ppm_pool_scales,
        dropout=dropout,
        num_classes=num_classes,
        img_size=512
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    val_miou = checkpoint.get('val_miou', 'unknown')
    print(f"  Checkpoint from epoch {epoch}, Val mIoU: {val_miou}")
    
    model = model.to(device)
    model.eval()
    
    return model, config


def evaluate_model(model, dataloader, device, class_names=['Clean', 'Damage']):
    """Evaluate model on dataloader
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for validation set
        device: torch device
        class_names: List of class names
    
    Returns:
        metrics_dict: Dictionary of computed metrics
        predictions: List of (image, mask, pred) tuples for visualization
    """
    metrics = SegmentationMetrics(num_classes=len(class_names))
    predictions = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics.update(preds, masks)
            
            # Store some predictions for visualization
            if len(predictions) < 10:  # Store first 10 batches
                for img, mask, pred in zip(images, masks, preds):
                    predictions.append((
                        img.cpu().numpy(),
                        mask.cpu().numpy(),
                        pred.cpu().numpy()
                    ))
    
    # Compute all metrics
    iou_per_class, miou = metrics.compute_iou()
    f1_per_class, mf1 = metrics.compute_f1()
    precision_per_class, recall_per_class = metrics.compute_precision_recall()
    accuracy = metrics.compute_accuracy()
    confusion_matrix = metrics.get_confusion_matrix()
    
    # Build metrics dictionary
    metrics_dict = {
        'miou': miou,
        'mf1': mf1,
        'accuracy': accuracy,
        'iou_per_class': {class_names[i]: iou_per_class[i] for i in range(len(class_names))},
        'f1_per_class': {class_names[i]: f1_per_class[i] for i in range(len(class_names))},
        'precision_per_class': {class_names[i]: precision_per_class[i] for i in range(len(class_names))},
        'recall_per_class': {class_names[i]: recall_per_class[i] for i in range(len(class_names))},
        'confusion_matrix': confusion_matrix.tolist()
    }
    
    return metrics_dict, predictions


def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix: {output_path}")


def plot_predictions(predictions, output_dir, num_samples=6):
    """Plot prediction visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select samples
    samples = predictions[:num_samples]
    
    for idx, (image, mask, pred) in enumerate(samples):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image.transpose(1, 2, 0)  # CHW -> HWC
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        # Plot image
        axes[0].imshow(image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Plot ground truth
        axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction
        axes[2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'prediction_{idx+1}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved {len(samples)} prediction visualizations to {output_dir}")


def save_metrics(metrics_dict, output_path):
    """Save metrics to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✅ Saved metrics: {output_path}")


def print_metrics_summary(metrics_dict, class_names=['Clean', 'Damage']):
    """Print metrics summary to console"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Overall Metrics:")
    print(f"  mIoU (mean IoU):  {metrics_dict['miou']:.4f}")
    print(f"  mF1 (mean F1):    {metrics_dict['mf1']:.4f}")
    print(f"  Accuracy:         {metrics_dict['accuracy']:.4f}")
    print()
    print("Per-Class Metrics:")
    for cls in class_names:
        print(f"  {cls}:")
        print(f"    IoU:       {metrics_dict['iou_per_class'][cls]:.4f}")
        print(f"    F1:        {metrics_dict['f1_per_class'][cls]:.4f}")
        print(f"    Precision: {metrics_dict['precision_per_class'][cls]:.4f}")
        print(f"    Recall:    {metrics_dict['recall_per_class'][cls]:.4f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
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
        model_name = config['model']['name']
        args.output = f"logs/{model_name}/evaluation"
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Load model
    model, model_config = load_model(args.config, args.checkpoint, device)
    
    # Load validation dataset
    # Get data paths
    data_cfg = config.get('data', {})
    root_dir = data_cfg.get('root_dir', 'data/artefact')
    train_split = data_cfg.get('train_split', 0.8)
    random_seed = data_cfg.get('random_seed', 42)
    binary_mode = data_cfg.get('binary_mode', True)
    ignore_index = data_cfg.get('ignore_index', 255)
    img_size = data_cfg.get('image_size', 512)
    
    # Get all image and mask paths
    data_root = Path(root_dir)
    images_dir = data_root / 'images'
    masks_dir = data_root / 'annotations'
    
    image_files = sorted(images_dir.glob('*.png'))
    mask_files = sorted(masks_dir.glob('*.png'))
    
    image_paths = [str(f) for f in image_files]
    mask_paths = [str(f) for f in mask_files]
    
    # Train/val split (same as training)
    _, val_images, _, val_masks = train_test_split(
        image_paths, mask_paths,
        train_size=train_split,
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"\nValidation set: {len(val_images)} images")
    
    # Create validation dataset with transforms (Resize + Normalize + ToTensor)
    val_dataset = ArtefactDataset(
        val_images, val_masks,
        transform=get_val_transforms(img_size),
        binary_mode=binary_mode,
        ignore_index=ignore_index
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"\nValidation set: {len(val_dataset)} images")
    
    # Evaluate
    class_names = ['Clean', 'Damage']  # Binary segmentation
    metrics_dict, predictions = evaluate_model(model, val_loader, device, class_names)
    
    # Print summary
    print_metrics_summary(metrics_dict, class_names)
    
    # Save metrics
    metrics_path = os.path.join(args.output, 'metrics.json')
    save_metrics(metrics_dict, metrics_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output, 'confusion_matrix.png')
    plot_confusion_matrix(
        np.array(metrics_dict['confusion_matrix']),
        class_names,
        cm_path
    )
    
    # Plot predictions
    pred_dir = os.path.join(args.output, 'predictions')
    plot_predictions(predictions, pred_dir, num_samples=6)
    
    print(f"\n✅ Evaluation complete! Results saved to: {args.output}")


if __name__ == '__main__':
    main()
