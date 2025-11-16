"""
POC-5.8: Training Script with SMP + AMP

Key features:
- Segmentation Models PyTorch (SMP) for proven architectures
- Automatic Mixed Precision (AMP) for 2x speedup
- OneCycleLR for better convergence
- Minimal code, maximum performance
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_dataloaders, compute_class_weights
from preload_dataset import create_preloaded_dataloaders
from model_factory import create_model  # Use custom factory for timm encoders


class ModelWithLoss(nn.Module):
    """
    Wrapper that combines model + loss for DataParallel.
    Each GPU computes its own loss, then they are averaged.
    """
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion
    
    def forward(self, images, masks=None):
        predictions = self.model(images)
        if masks is not None:
            # Training mode: return loss
            loss = self.criterion(predictions, masks)
            return loss, predictions
        else:
            # Inference mode: return predictions only
            return predictions


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True,
    use_dataparallel: bool = False
) -> dict:
    """
    Train one epoch with AMP
    
    Returns:
        Dict with loss, time, throughput
    """
    model.train()
    epoch_loss = 0.0
    start_time = time.time()
    
    for images, masks in train_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Forward with AMP
        if use_amp:
            with autocast():
                if use_dataparallel:
                    # Model wrapper returns (loss, predictions)
                    loss, predictions = model(images, masks)
                    loss = loss.mean()  # Average losses from all GPUs
                else:
                    predictions = model(images)
                    loss = criterion(predictions, masks)
        else:
            if use_dataparallel:
                loss, predictions = model(images, masks)
                loss = loss.mean()
            else:
                predictions = model(images)
                loss = criterion(predictions, masks)
        
        # Backward with gradient scaling
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Scheduler step (OneCycleLR steps every batch)
        scheduler.step()
        
        # Accumulate loss
        epoch_loss += loss.detach().item()
    
    elapsed = time.time() - start_time
    avg_loss = epoch_loss / len(train_loader)
    throughput = len(train_loader.dataset) / elapsed
    
    return {
        'loss': avg_loss,
        'time': elapsed,
        'throughput': throughput
    }


def validate_epoch(
    model,
    val_loader,
    criterion,
    device: torch.device,
    num_classes: int,
    use_amp: bool = True,
    use_dataparallel: bool = False
) -> dict:
    """
    Validate with IoU computation
    
    Returns:
        Dict with loss, miou, iou_per_class
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    epoch_loss = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward with AMP
            if use_amp:
                with autocast():
                    if use_dataparallel:
                        # Inference mode: no masks, returns predictions only
                        predictions = model(images)
                    else:
                        predictions = model(images)
                    loss = criterion(predictions, masks)
            else:
                if use_dataparallel:
                    predictions = model(images)
                else:
                    predictions = model(images)
                loss = criterion(predictions, masks)
            
            epoch_loss += loss.item()
            
            # Store predictions for IoU
            preds = predictions.argmax(dim=1)
            all_preds.append(preds)
            all_targets.append(masks)
    
    # Compute IoU (vectorized on GPU)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    iou_per_class = compute_iou(all_preds, all_targets, num_classes)
    miou = iou_per_class.mean().item()
    avg_loss = epoch_loss / len(val_loader)
    
    return {
        'loss': avg_loss,
        'miou': miou,
        'iou_per_class': iou_per_class.cpu().numpy()
    }


def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Compute IoU per class (GPU-accelerated)
    
    Args:
        preds: (N, H, W) predicted class indices
        targets: (N, H, W) ground truth class indices
        num_classes: Number of classes
    
    Returns:
        IoU per class (num_classes,)
    """
    ious = []
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union == 0:
            iou = torch.tensor(0.0, device=preds.device)
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return torch.stack(ious)


def get_gpu_memory() -> tuple:
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return allocated, total
    return 0, 0


def main():
    parser = argparse.ArgumentParser(description='POC-5.8: Standard Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    parser.add_argument('--test-epoch', action='store_true',
                       help='Run only 1 epoch for testing')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("POC-5.8: Standard Segmentation Pipeline")
    print("="*60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: {config['model']['architecture']}")
    print(f"Encoder: {config['model']['encoder_name']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Image size: {config['data']['image_size']}")
    print(f"Mixed Precision: {config['training']['mixed_precision']}")
    print(f"Epochs: {1 if args.test_epoch else config['training']['epochs']}")
    print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    
    use_preload = config['data'].get('use_preload', False)
    
    if use_preload:
        print("🔥 Using RAM PRE-LOADING for maximum throughput")
        train_loader, val_loader = create_preloaded_dataloaders(
            data_root=config['data']['data_dir'],
            image_size=config['data']['image_size'],
            batch_size=config['training']['batch_size'],
            num_workers=config['dataloader']['num_workers'],
            use_augmented=config['data'].get('use_augmented', False),
            preload_to_gpu=config['data'].get('preload_to_gpu', False)
        )
    else:
        print("Using standard CPU dataloader")
        train_loader, val_loader = create_dataloaders(
            data_root=config['data']['data_dir'],
            image_size=config['data']['image_size'],
            batch_size=config['training']['batch_size'],
            num_workers=config['dataloader']['num_workers'],
            pin_memory=config['dataloader']['pin_memory'],
            persistent_workers=config['dataloader']['persistent_workers'],
            prefetch_factor=config['dataloader']['prefetch_factor'],
            drop_last=config['dataloader']['drop_last'],
            use_augmented=config['data'].get('use_augmented', False)
        )
    
    # Create model
    print("Creating model...")
    model = create_model(config).to(device)
    
    # Loss function (SMP DiceLoss)
    print("Creating loss function...")
    criterion = smp.losses.DiceLoss(
        mode=config['loss']['mode'],
        smooth=config['loss'].get('smooth', 1.0)
    )
    print(f"Loss: {config['loss']['type'].upper()} ({config['loss']['mode']})")
    print()
    
    # Multi-GPU support with loss integration
    use_dataparallel = False
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        # Wrap model with loss for distributed computation
        model = ModelWithLoss(model, criterion)
        model = torch.nn.DataParallel(model)
        use_dataparallel = True
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler (OneCycleLR)
    num_epochs = 1 if args.test_epoch else config['training']['epochs']
    total_steps = len(train_loader) * num_epochs
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['scheduler']['max_lr'],
        total_steps=total_steps,
        pct_start=config['scheduler'].get('pct_start', 0.3),
        anneal_strategy=config['scheduler'].get('anneal_strategy', 'cos')
    )
    
    # Gradient scaler for AMP
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Training loop
    print("Starting training...")
    print("="*60)
    print()
    
    best_miou = 0.0
    num_classes = config['model']['classes']
    
    # Create checkpoint directory
    model_name = f"{config['model']['architecture']}_{config['model']['encoder_name']}"
    checkpoint_dir = Path(f'../logs/{model_name}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, use_amp=config['training']['mixed_precision'],
            use_dataparallel=use_dataparallel
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, num_classes,
            use_amp=config['training']['mixed_precision'],
            use_dataparallel=use_dataparallel
        )
        
        # GPU memory
        vram_used, vram_total = get_gpu_memory()
        vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0
        
        # Log
        print(f"Epoch {epoch}/{num_epochs} ({train_metrics['time']:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}")
        print(f"         mIoU: {val_metrics['miou']:.4f}")
        print(f"  Throughput: {train_metrics['throughput']:.1f} imgs/s")
        print(f"  VRAM: {vram_used:.2f}GB / {vram_total:.2f}GB ({vram_pct:.1f}%)")
        
        # Save best checkpoint
        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            print(f"  ✅ New best mIoU: {best_miou:.4f}")
            
            # Save checkpoint (extract model from DataParallel wrapper if needed)
            model_to_save = model.module.model if use_dataparallel else model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        
        print()
    
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best mIoU: {best_miou:.4f}")
    print()
    
    if args.test_epoch:
        print("🧪 Test epoch completed")
        print()
        print("Metrics:")
        print(f"  - VRAM: {vram_pct:.1f}% (target: >40%)")
        print(f"  - Throughput: {train_metrics['throughput']:.1f} imgs/s (target: >100)")
        print()
        
        if vram_pct < 40:
            print("⚠️  VRAM low - can increase batch size")
        elif vram_pct > 90:
            print("⚠️  VRAM high - consider reducing batch size")
        else:
            print("✅ VRAM usage optimal")
        
        if train_metrics['throughput'] < 100:
            print("⚠️  Throughput below target")
        else:
            print("✅ Throughput excellent")


if __name__ == '__main__':
    main()
