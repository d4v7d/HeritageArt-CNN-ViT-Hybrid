"""
Training script for POC-4: Minimal ARTeFACT Training
Binary segmentation: Damage vs Clean
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import segmentation_models_pytorch as smp

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
from dataset import prepare_dataloaders, verify_dataset


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal Loss for segmentation."""
    
    def __init__(
        self,
        dice_weight=0.5,
        focal_weight=0.5,
        focal_alpha=0.25,
        focal_gamma=2.0,
        ignore_index=255
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ignore_index = ignore_index
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """Soft Dice Loss."""
        pred = F.softmax(pred, dim=1)
        
        # Mask out ignore_index first
        mask = (target != self.ignore_index).float().unsqueeze(1)
        
        # Clamp target to valid range for one-hot encoding
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0  # Temporarily set to 0
        
        # Create one-hot target
        target_one_hot = F.one_hot(target_clamped, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Apply mask to exclude ignore_index pixels
        pred = pred * mask
        target_one_hot = target_one_hot * mask
        
        # Calculate dice per class
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return 1.0 - dice.mean()
    
    def focal_loss(self, pred, target):
        """Focal Loss."""
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


def compute_metrics(pred, target, ignore_index=255, num_classes=2):
    """
    Compute segmentation metrics.
    
    Args:
        pred: Model predictions (logits) [B, C, H, W]
        target: Ground truth [B, H, W]
        ignore_index: Index to ignore in metrics
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    pred_labels = torch.argmax(pred, dim=1)  # [B, H, W]
    
    # Create mask excluding ignore_index
    valid_mask = (target != ignore_index)
    
    pred_masked = pred_labels[valid_mask]
    target_masked = target[valid_mask]
    
    if len(pred_masked) == 0:
        return {'iou': 0.0, 'f1': 0.0, 'accuracy': 0.0}
    
    # Per-class metrics
    ious = []
    f1s = []
    
    for cls in range(num_classes):
        pred_cls = (pred_masked == cls)
        target_cls = (target_masked == cls)
        
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        
        if union > 0:
            iou = intersection / union
            precision = intersection / pred_cls.sum().item() if pred_cls.sum() > 0 else 0
            recall = intersection / target_cls.sum().item() if target_cls.sum() > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            iou = 1.0  # Both pred and target are empty for this class
            f1 = 1.0
        
        ious.append(iou)
        f1s.append(f1)
    
    # Mean metrics
    miou = np.mean(ious)
    mf1 = np.mean(f1s)
    accuracy = (pred_masked == target_masked).float().mean().item()
    
    return {
        'miou': miou,
        'mf1': mf1,
        'accuracy': accuracy,
        'iou_per_class': ious,
        'f1_per_class': f1s
    }


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    running_metrics = {'miou': 0.0, 'mf1': 0.0, 'accuracy': 0.0}
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(outputs, masks)
        
        running_loss += loss.item()
        for k in running_metrics:
            running_metrics[k] += metrics[k]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'miou': f'{metrics["miou"]:.4f}',
            'mf1': f'{metrics["mf1"]:.4f}'
        })
    
    # Average metrics
    n_batches = len(loader)
    avg_loss = running_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}
    
    return avg_loss, avg_metrics


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, epoch, total_epochs):
    """Validate for one epoch."""
    model.eval()
    
    running_loss = 0.0
    running_metrics = {'miou': 0.0, 'mf1': 0.0, 'accuracy': 0.0}
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [Val]')
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        metrics = compute_metrics(outputs, masks)
        
        running_loss += loss.item()
        for k in running_metrics:
            running_metrics[k] += metrics[k]
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'miou': f'{metrics["miou"]:.4f}',
            'mf1': f'{metrics["mf1"]:.4f}'
        })
    
    n_batches = len(loader)
    avg_loss = running_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}
    
    return avg_loss, avg_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train ARTeFACT segmentation model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Verify dataset
    print("\n" + "="*80)
    verify_dataset(config['data']['root'], config['data']['binary_mode'])
    print("="*80 + "\n")
    
    # Prepare dataloaders
    train_loader, val_loader = prepare_dataloaders(config)
    
    # Create model
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights=config['model']['encoder_weights'],
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes']
    )
    model = model.to(device)
    
    print(f"\nModel: {config['model']['name']}")
    print(f"  Encoder: {config['model']['encoder']}")
    print(f"  Classes: {config['model']['classes']} (binary mode)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss, optimizer, scheduler
    criterion = DiceFocalLoss(
        dice_weight=config['training']['loss']['dice_weight'],
        focal_weight=config['training']['loss']['focal_weight'],
        focal_alpha=config['training']['loss']['focal_alpha'],
        focal_gamma=config['training']['loss']['focal_gamma'],
        ignore_index=config['data']['ignore_index']
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['scheduler']['min_lr']
    )
    
    # Create output directories
    checkpoint_dir = Path(config['paths']['checkpoints'])
    logs_dir = Path(config['paths']['logs'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_miou = 0.0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0.0)
    
    # Training loop
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config['training']['epochs']
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, config['training']['epochs']
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, mIoU: {train_metrics['miou']:.4f}, mF1: {train_metrics['mf1']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, mIoU: {val_metrics['miou']:.4f}, mF1: {val_metrics['mf1']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save checkpoint
        is_best = val_metrics['miou'] > best_miou
        if is_best:
            best_miou = val_metrics['miou']
        
        if epoch % config['training']['save_freq'] == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_miou': best_miou,
                'config': config
            }
            
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            if is_best:
                best_path = checkpoint_dir / 'best_model.pth'
                torch.save(checkpoint, best_path)
                print(f"Saved best model: {best_path} (mIoU: {best_miou:.4f})")
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
