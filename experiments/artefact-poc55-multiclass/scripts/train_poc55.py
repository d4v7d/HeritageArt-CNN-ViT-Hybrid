"""
POC-5.5: Hierarchical Multi-Task Segmentation Training Script
Tests Innovation #1 (Hierarchical MTL) on laptop hardware (6GB VRAM).

Trains binary + coarse + fine segmentation heads simultaneously.
Optimized for RTX 3050/1000 Ada with 256px resolution and FP16.
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

# Disable PIL's DecompressionBomb warning (we handle large images safely with transforms)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Add current dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.hierarchical_upernet import HierarchicalUPerNet
from dataset_multiclass import ArtefactMulticlassDataset, get_multiclass_transforms
from losses import HierarchicalDiceFocalLoss, compute_class_weights


def load_config(config_path: str) -> Dict:
    """Load and merge configuration files."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load base config if specified (support both 'base' and '_base_')
    base_key = '_base_' if '_base_' in config else 'base'
    if base_key in config:
        base_path = Path(config_path).parent / config[base_key]
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Recursively load base's base if it exists
        if '_base_' in base_config or 'base' in base_config:
            base_config = load_config(str(base_path))
        
        # Merge configs (current overrides base)
        def deep_update(base, override):
            for key, value in override.items():
                if key == base_key:  # Skip the _base_ key itself
                    continue
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        
        deep_update(base_config, config)
        config = base_config
    
    return config


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Compute segmentation metrics (IoU, Dice).
    
    Args:
        predictions: (B, C, H, W) logits
        targets: (B, H, W) class indices
        num_classes: Number of classes
        ignore_index: Class to ignore
    
    Returns:
        Dict with mIoU, mDice, per-class IoU
    """
    # Get predicted classes
    preds = predictions.argmax(dim=1)  # (B, H, W)
    
    # Flatten
    preds = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    
    # Remove ignored pixels
    valid_mask = targets != ignore_index
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    
    # Compute IoU per class
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        
        if union == 0:
            # Class not present in this batch
            iou = float('nan')
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    # Compute mean IoU (ignoring NaN classes)
    ious_array = np.array(ious)
    valid_ious = ious_array[~np.isnan(ious_array)]
    miou = valid_ious.mean() if len(valid_ious) > 0 else 0.0
    
    # Dice = 2 * IoU / (1 + IoU)
    dice_scores = 2 * valid_ious / (1 + valid_ious)
    mdice = dice_scores.mean() if len(dice_scores) > 0 else 0.0
    
    metrics = {
        'mIoU': float(miou),
        'mDice': float(mdice),
        'per_class_iou': ious
    }
    
    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: Dict,
    epoch: int,
    writer: SummaryWriter
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_binary_loss = 0.0
    total_coarse_loss = 0.0
    total_fine_loss = 0.0
    
    gradient_accumulation = config['training'].get('gradient_accumulation_steps', 1)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Mixed precision forward
        with autocast(enabled=config['training']['mixed_precision']):
            predictions = model(images, return_all_heads=True)
            loss, loss_dict = criterion(predictions, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights every N steps
        if (batch_idx + 1) % gradient_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Accumulate losses
        total_loss += loss_dict['loss_total']
        total_binary_loss += loss_dict['loss_binary']
        total_coarse_loss += loss_dict['loss_coarse']
        total_fine_loss += loss_dict['loss_fine']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['loss_total']:.4f}",
            'binary': f"{loss_dict['loss_binary']:.4f}",
            'coarse': f"{loss_dict['loss_coarse']:.4f}",
            'fine': f"{loss_dict['loss_fine']:.4f}"
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('train/loss_total', loss_dict['loss_total'], global_step)
        writer.add_scalar('train/loss_binary', loss_dict['loss_binary'], global_step)
        writer.add_scalar('train/loss_coarse', loss_dict['loss_coarse'], global_step)
        writer.add_scalar('train/loss_fine', loss_dict['loss_fine'], global_step)
    
    # Average losses
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'loss_binary': total_binary_loss / num_batches,
        'loss_coarse': total_coarse_loss / num_batches,
        'loss_fine': total_fine_loss / num_batches
    }
    
    return metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict,
    epoch: int,
    writer: SummaryWriter
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_binary_loss = 0.0
    total_coarse_loss = 0.0
    total_fine_loss = 0.0
    
    # Metrics accumulators
    all_metrics_binary = []
    all_metrics_coarse = []
    all_metrics_fine = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    for images, targets in pbar:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Forward
        with autocast(enabled=config['training']['mixed_precision']):
            predictions = model(images, return_all_heads=True)
            loss, loss_dict = criterion(predictions, targets)
        
        # Accumulate losses
        total_loss += loss_dict['loss_total']
        total_binary_loss += loss_dict['loss_binary']
        total_coarse_loss += loss_dict['loss_coarse']
        total_fine_loss += loss_dict['loss_fine']
        
        # Compute metrics for each head
        metrics_binary = compute_metrics(
            predictions['binary'], targets['binary'],
            num_classes=2, ignore_index=255
        )
        metrics_coarse = compute_metrics(
            predictions['coarse'], targets['coarse'],
            num_classes=4, ignore_index=255
        )
        metrics_fine = compute_metrics(
            predictions['fine'], targets['fine'],
            num_classes=16, ignore_index=255
        )
        
        all_metrics_binary.append(metrics_binary)
        all_metrics_coarse.append(metrics_coarse)
        all_metrics_fine.append(metrics_fine)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['loss_total']:.4f}",
            'mIoU_fine': f"{metrics_fine['mIoU']:.4f}"
        })
    
    # Average losses and metrics
    num_batches = len(dataloader)
    
    avg_metrics = {
        'loss': total_loss / num_batches,
        'loss_binary': total_binary_loss / num_batches,
        'loss_coarse': total_coarse_loss / num_batches,
        'loss_fine': total_fine_loss / num_batches,
        'mIoU_binary': np.mean([m['mIoU'] for m in all_metrics_binary]),
        'mDice_binary': np.mean([m['mDice'] for m in all_metrics_binary]),
        'mIoU_coarse': np.mean([m['mIoU'] for m in all_metrics_coarse]),
        'mDice_coarse': np.mean([m['mDice'] for m in all_metrics_coarse]),
        'mIoU_fine': np.mean([m['mIoU'] for m in all_metrics_fine]),
        'mDice_fine': np.mean([m['mDice'] for m in all_metrics_fine])
    }
    
    # Log to tensorboard
    writer.add_scalar('val/loss_total', avg_metrics['loss'], epoch)
    writer.add_scalar('val/mIoU_binary', avg_metrics['mIoU_binary'], epoch)
    writer.add_scalar('val/mIoU_coarse', avg_metrics['mIoU_coarse'], epoch)
    writer.add_scalar('val/mIoU_fine', avg_metrics['mIoU_fine'], epoch)
    
    return avg_metrics


def main(args):
    """Main training function."""
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(yaml.dump(config, default_flow_style=False))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create output directories
    exp_name = config.get('logging', {}).get('experiment_name', 'poc55_experiment')
    output_dir = Path(args.output_dir) / exp_name
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)
    
    # Create datasets
    print("\nPreparing datasets...")
    data_root = Path(config['data']['root'])
    images_dir = data_root / 'images'
    annotations_dir = data_root / 'annotations'
    
    # Get all image and mask paths
    image_files = sorted(list(images_dir.glob('*.png')))
    mask_files = sorted(list(annotations_dir.glob('*.png')))
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    # Split into train/val
    split_ratio = config['data'].get('train_val_split', 0.8)
    split_idx = int(len(image_files) * split_ratio)
    
    train_images = [str(f) for f in image_files[:split_idx]]
    train_masks = [str(f) for f in mask_files[:split_idx]]
    val_images = [str(f) for f in image_files[split_idx:]]
    val_masks = [str(f) for f in mask_files[split_idx:]]
    
    print(f"Train: {len(train_images)} samples, Val: {len(val_images)} samples")
    
    train_transform = get_multiclass_transforms(config, mode='train')
    val_transform = get_multiclass_transforms(config, mode='val')
    
    train_dataset = ArtefactMulticlassDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        transform=train_transform
    )
    
    val_dataset = ArtefactMulticlassDataset(
        image_paths=val_images,
        mask_paths=val_masks,
        transform=val_transform
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Compute class weights if enabled
    class_weights_binary = None
    class_weights_coarse = None
    class_weights_fine = None
    
    class_weights_config = config['training'].get('class_weights')
    if class_weights_config and class_weights_config.get('method'):
        print("\nComputing class weights...")
        
        # Use fine-grained weights (binary/coarse derived from fine)
        weights_fine = compute_class_weights(
            train_dataset,
            num_classes=16,
            method=class_weights_config['method']
        )
        class_weights_fine = weights_fine.to(device)
        
        # Simplified weights for binary and coarse (could be computed separately)
        class_weights_binary = torch.ones(2).to(device)
        class_weights_coarse = torch.ones(4).to(device)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    # Create model
    print("\nCreating model...")
    from models.hierarchical_upernet import build_hierarchical_model
    model = build_hierarchical_model(config)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    # Create loss function
    criterion = HierarchicalDiceFocalLoss(
        binary_weight=config['loss']['weights']['binary'],
        coarse_weight=config['loss']['weights']['coarse'],
        fine_weight=config['loss']['weights']['fine'],
        dice_weight=config['loss']['dice_weight'],
        focal_weight=config['loss']['focal_weight'],
        alpha=config['loss']['focal_alpha'],
        gamma=config['loss']['focal_gamma'],
        binary_class_weights=class_weights_binary,
        coarse_class_weights=class_weights_coarse,
        fine_class_weights=class_weights_fine
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['optimizer']['lr']),
        weight_decay=float(config['training']['optimizer']['weight_decay'])
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['scheduler']['T_max']
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Training loop
    print("\nStarting training...")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['optimizer']['lr']}")
    print(f"Mixed precision: {config['training']['mixed_precision']}")
    print(f"Gradient accumulation: {config['training'].get('gradient_accumulation_steps', 1)}")
    
    best_miou = 0.0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(1, config['training']['epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, config, epoch, writer
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, config, epoch, writer
        )
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{config['training']['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}")
        print(f"         mIoU (Binary): {val_metrics['mIoU_binary']:.4f}")
        print(f"         mIoU (Coarse): {val_metrics['mIoU_coarse']:.4f}")
        print(f"         mIoU (Fine):   {val_metrics['mIoU_fine']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'config': config,
            'metrics': val_metrics
        }
        
        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # Save best model
        if val_metrics['mIoU_fine'] > best_miou:
            best_miou = val_metrics['mIoU_fine']
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f"  ✅ New best model! mIoU: {best_miou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save periodic checkpoints
        save_interval = config.get('logging', {}).get('save_interval', 5)
        if epoch % save_interval == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Early stopping
        patience = config.get('training', {}).get('early_stopping', {}).get('patience', 10)
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered (patience: {patience})")
            break
        
        # Test epoch mode (only 1 epoch)
        if args.test_epoch:
            print("\n🧪 Test epoch mode - stopping after 1 epoch")
            break
    
    # Training finished
    total_time = time.time() - start_time
    print(f"\n✅ Training finished!")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Best mIoU (fine): {best_miou:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='POC-5.5 Hierarchical Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='../logs', help='Output directory')
    parser.add_argument('--test-epoch', action='store_true', help='Test mode: run only 1 epoch')
    
    args = parser.parse_args()
    
    main(args)
