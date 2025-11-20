"""
POC-60: Training Script with Hierarchical MTL Support

Key features:
- Supports both standard UNet and Hierarchical UPerNet
- Hierarchical loss: 0.2*binary + 0.3*coarse + 1.0*fine
- Automatic Mixed Precision (AMP) for 2x speedup
- OneCycleLR for better convergence
"""

# Force single GPU BEFORE importing torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

from dataset import create_dataloaders, compute_class_weights, fine_to_binary, fine_to_coarse
from preload_dataset import create_preloaded_dataloaders
from model_factory import create_model
from losses import DiceFocalLoss
from hierarchical_utils import (
    create_hierarchical_criterion,
    compute_hierarchical_metrics,
    is_hierarchical_model
)


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
    use_dataparallel: bool = False,
    is_hierarchical: bool = False
) -> dict:
    """
    Train one epoch with AMP (supports hierarchical models)
    
    Returns:
        Dict with loss, time, throughput (and per-head losses if hierarchical)
    """
    model.train()
    epoch_loss = 0.0
    start_time = time.time()
    
    # Hierarchical tracking
    if is_hierarchical:
        epoch_binary_loss = 0.0
        epoch_coarse_loss = 0.0
        epoch_fine_loss = 0.0
    
    print(f"  🔄 Training: processing {len(train_loader)} batches...")
    for batch_idx, (images, masks) in enumerate(train_loader, 1):
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
                    
                    if is_hierarchical:
                        # Hierarchical: predictions = dict{'binary', 'coarse', 'fine'}
                        # Convert targets
                        binary_targets = fine_to_binary(masks)
                        coarse_targets = fine_to_coarse(masks)
                        
                        # Compute loss (criterion handles hierarchical)
                        loss_dict = criterion(predictions, masks, binary_targets, coarse_targets)
                        loss = loss_dict['total']
                    else:
                        # Standard: predictions = tensor
                        loss = criterion(predictions, masks)
        else:
            if use_dataparallel:
                loss, predictions = model(images, masks)
                loss = loss.mean()
            else:
                predictions = model(images)
                
                if is_hierarchical:
                    binary_targets = fine_to_binary(masks)
                    coarse_targets = fine_to_coarse(masks)
                    loss_dict = criterion(predictions, masks, binary_targets, coarse_targets)
                    loss = loss_dict['total']
                else:
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
        
        if is_hierarchical and not use_dataparallel:
            epoch_binary_loss += loss_dict['binary'].item()
            epoch_coarse_loss += loss_dict['coarse'].item()
            epoch_fine_loss += loss_dict['fine'].item()
        
        # Progress indicator every 5 batches
        if batch_idx % 5 == 0 or batch_idx == 1:
            if is_hierarchical and not use_dataparallel:
                print(f"    Batch {batch_idx}/{len(train_loader)} - Total: {loss.item():.4f} "
                      f"(B: {loss_dict['binary'].item():.3f}, C: {loss_dict['coarse'].item():.3f}, "
                      f"F: {loss_dict['fine'].item():.3f})")
            else:
                print(f"    Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    elapsed = time.time() - start_time
    avg_loss = epoch_loss / len(train_loader)
    throughput = len(train_loader.dataset) / elapsed
    
    result = {
        'loss': avg_loss,
        'time': elapsed,
        'throughput': throughput
    }
    
    if is_hierarchical:
        result['binary_loss'] = epoch_binary_loss / len(train_loader)
        result['coarse_loss'] = epoch_coarse_loss / len(train_loader)
        result['fine_loss'] = epoch_fine_loss / len(train_loader)
    
    return result


def validate_epoch(
    model,
    val_loader,
    criterion,
    device: torch.device,
    num_classes: int,
    use_amp: bool = True,
    use_dataparallel: bool = False,
    is_hierarchical: bool = False
) -> dict:
    """
    Validate with IoU computation (supports hierarchical models)
    
    Returns:
        Dict with loss, miou, iou_per_class (and per-head metrics if hierarchical)
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    epoch_loss = 0.0
    
    # Hierarchical tracking
    if is_hierarchical:
        all_binary_preds = []
        all_coarse_preds = []
        all_fine_preds = []
    
    print(f"  🔍 Validating: processing {len(val_loader)} batches...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward with AMP
            if use_amp:
                with autocast():
                    if use_dataparallel:
                        predictions = model(images)
                    else:
                        predictions = model(images)
                    
                    if is_hierarchical:
                        # Hierarchical: compute loss with all targets
                        binary_targets = fine_to_binary(masks)
                        coarse_targets = fine_to_coarse(masks)
                        loss_dict = criterion(predictions, masks, binary_targets, coarse_targets)
                        loss = loss_dict['total']
                    else:
                        loss = criterion(predictions, masks)
            else:
                if use_dataparallel:
                    predictions = model(images)
                else:
                    predictions = model(images)
                
                if is_hierarchical:
                    binary_targets = fine_to_binary(masks)
                    coarse_targets = fine_to_coarse(masks)
                    loss_dict = criterion(predictions, masks, binary_targets, coarse_targets)
                    loss = loss_dict['total']
                else:
                    loss = criterion(predictions, masks)
            
            epoch_loss += loss.item()
            
            # Store predictions for IoU
            if is_hierarchical:
                # Store all 3 heads
                all_binary_preds.append(predictions['binary'].argmax(dim=1))
                all_coarse_preds.append(predictions['coarse'].argmax(dim=1))
                all_fine_preds.append(predictions['fine'].argmax(dim=1))
                all_targets.append(masks)
            else:
                preds = predictions.argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(masks)
            
            # Progress indicator
            if batch_idx % 2 == 0 or batch_idx == 1:
                print(f"    Batch {batch_idx}/{len(val_loader)}")
    
    # Compute IoU (vectorized on GPU)
    all_targets = torch.cat(all_targets, dim=0)
    
    if is_hierarchical:
        # Compute IoU for all 3 heads
        all_binary_preds = torch.cat(all_binary_preds, dim=0)
        all_coarse_preds = torch.cat(all_coarse_preds, dim=0)
        all_fine_preds = torch.cat(all_fine_preds, dim=0)
        
        # Convert targets
        binary_targets = fine_to_binary(all_targets)
        coarse_targets = fine_to_coarse(all_targets)
        
        # Compute per-head IoU
        binary_iou = compute_iou(all_binary_preds, binary_targets, 2)
        coarse_iou = compute_iou(all_coarse_preds, coarse_targets, 4)
        fine_iou = compute_iou(all_fine_preds, all_targets, num_classes)
        
        binary_miou = binary_iou.mean().item()
        coarse_miou = coarse_iou.mean().item()
        fine_miou = fine_iou.mean().item()
        
        return {
            'loss': epoch_loss / len(val_loader),
            'binary_miou': binary_miou,
            'coarse_miou': coarse_miou,
            'fine_miou': fine_miou,
            'miou': fine_miou,  # Main metric (for compatibility)
            'binary_iou_per_class': binary_iou.cpu().numpy(),
            'coarse_iou_per_class': coarse_iou.cpu().numpy(),
            'fine_iou_per_class': fine_iou.cpu().numpy()
        }
    else:
        # Standard model
        all_preds = torch.cat(all_preds, dim=0)
        iou_per_class = compute_iou(all_preds, all_targets, num_classes)
        miou = iou_per_class.mean().item()
        
        return {
            'loss': epoch_loss / len(val_loader),
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
    parser = argparse.ArgumentParser(description='POC-60: Hierarchical MTL Training')
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
    
    # Detect if hierarchical model
    is_hierarchical = config['model'].get('hierarchical', False)
    
    import sys
    print("="*60)
    if is_hierarchical:
        print(f"POC-60: Hierarchical Multi-Task Learning")
    else:
        print(f"POC-59: Standard Segmentation Pipeline")
    print("="*60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Visible GPU count: {torch.cuda.device_count()}")
    print(f"Architecture: {config['model'].get('architecture', 'Unet')}")
    print(f"Encoder: {config['model'].get('encoder_name', config['model'].get('encoder', 'unknown'))}")
    print(f"Hierarchical: {'✅ YES (3 heads)' if is_hierarchical else '❌ NO (single head)'}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Image size: {config['data']['image_size']}")
    print(f"Mixed Precision: {config['training']['mixed_precision']}")
    print(f"Epochs: {1 if args.test_epoch else config['training']['epochs']}")
    print()
    sys.stdout.flush()
    
    # Create dataloaders
    print("Creating dataloaders...")
    sys.stdout.flush()
    
    if config['data'].get('use_preload', False):
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
    
    # Load pre-computed class weights from metadata (POC-5.9 optimization)
    print("Loading class weights...")
    
    # Check for custom weights file in config (POC-5.9-v2 improvement)
    weights_filename = config['loss'].get('class_weights_file', 'class_weights.json')
    class_weights_path = Path(config['data']['data_dir']) / weights_filename
    
    if class_weights_path.exists():
        import json
        with open(class_weights_path) as f:
            weights_data = json.load(f)
        class_weights = torch.tensor(weights_data['weights'], dtype=torch.float32).to(device)
        print(f"  ✅ Loaded from: {weights_filename}")
        method = weights_data.get('method', 'unknown')
        num_images = weights_data.get('num_images', weights_data.get('images', 'N/A'))
        print(f"  Method: {method}, Images: {num_images}")
    else:
        print(f"  ⚠️  Pre-computed weights not found ({weights_filename}), using uniform weights")
        class_weights = None
    
    # Loss function
    print("\nCreating loss function...")
    if is_hierarchical:
        # Hierarchical loss (POC-60)
        criterion = create_hierarchical_criterion(config, device)
        print(f"Loss: HIERARCHICAL (0.2*binary + 0.3*coarse + 1.0*fine)")
    elif config['loss']['type'] == 'dice_focal':
        # Standard DiceFocalLoss (POC-59)
        criterion = DiceFocalLoss(
            dice_weight=config['loss'].get('dice_weight', 0.5),
            focal_weight=config['loss'].get('focal_weight', 0.5),
            gamma=config['loss'].get('focal_gamma', 2.0),
            alpha=config['loss'].get('focal_alpha', 0.25),
            smooth=config['loss'].get('smooth', 1.0),
            class_weights=class_weights,
            ignore_index=255
        )
        print(f"Loss: DICE+FOCAL (weights: {config['loss']['dice_weight']}/{config['loss']['focal_weight']})")
    elif config['loss']['type'] == 'dice':
        # Fallback to DiceLoss only
        from losses import DiceLoss as CustomDiceLoss
        criterion = CustomDiceLoss(
            smooth=config['loss'].get('smooth', 1.0),
            class_weights=class_weights,
            ignore_index=255
        )
        print(f"Loss: DICE (smooth: {config['loss']['smooth']})")
    else:
        # Fallback to SMP DiceLoss
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
    
    print(f"🚀 Starting epoch loop (1 to {num_epochs})...")
    print()
    for epoch in range(1, num_epochs + 1):
        print(f"\n📅 EPOCH {epoch}/{num_epochs}")
        print("="*60)
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, use_amp=config['training']['mixed_precision'],
            use_dataparallel=use_dataparallel,
            is_hierarchical=is_hierarchical
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, num_classes,
            use_amp=config['training']['mixed_precision'],
            use_dataparallel=use_dataparallel,
            is_hierarchical=is_hierarchical
        )
        
        # GPU memory
        vram_used, vram_total = get_gpu_memory()
        vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0
        
        # Log
        print(f"Epoch {epoch}/{num_epochs} ({train_metrics['time']:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}")
        
        if is_hierarchical:
            # Hierarchical metrics
            print(f"         Binary Loss: {train_metrics.get('binary_loss', 0):.4f}")
            print(f"         Coarse Loss: {train_metrics.get('coarse_loss', 0):.4f}")
            print(f"         Fine Loss: {train_metrics.get('fine_loss', 0):.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}")
            print(f"         Binary mIoU: {val_metrics.get('binary_miou', 0):.4f}")
            print(f"         Coarse mIoU: {val_metrics.get('coarse_miou', 0):.4f}")
            print(f"         Fine mIoU: {val_metrics.get('fine_miou', 0):.4f}")
        else:
            # Standard metrics
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
