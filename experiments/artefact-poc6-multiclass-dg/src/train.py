"""
POC-5.8: Training Script with SMP + AMP

Key features:
- Segmentation Models PyTorch (SMP) for proven architectures
- Automatic Mixed Precision (AMP) for 2x speedup
- OneCycleLR for better convergence
- Minimal code, maximum performance
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
# Add parent dir to path (for src. imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import create_dataloaders, compute_class_weights
from preload_dataset import create_preloaded_dataloaders
from model_factory import create_model
from losses import DiceFocalLoss
from src.losses.hierarchical_loss import HierarchicalDiceFocalLoss


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
            # Handle tuple return from hierarchical loss
            if isinstance(loss, tuple):
                return loss[0], predictions
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
    
    print(f"  🔄 Training: processing {len(train_loader)} batches...")
    for batch_idx, (images, masks) in enumerate(train_loader, 1):
        images = images.to(device, non_blocking=True)
        
        # Handle hierarchical masks (dict)
        if isinstance(masks, dict):
            masks = {k: v.to(device, non_blocking=True) for k, v in masks.items()}
        else:
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
                    loss_out = criterion(predictions, masks)
                    if isinstance(loss_out, tuple):
                        loss = loss_out[0]
                    else:
                        loss = loss_out
        else:
            if use_dataparallel:
                loss, predictions = model(images, masks)
                loss = loss.mean()
            else:
                predictions = model(images)
                loss_out = criterion(predictions, masks)
                if isinstance(loss_out, tuple):
                    loss = loss_out[0]
                else:
                    loss = loss_out
        
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
        
        # Progress indicator every 5 batches
        if batch_idx % 5 == 0 or batch_idx == 1:
            print(f"    Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
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
    
    print(f"  🔍 Validating: processing {len(val_loader)} batches...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader, 1):
            images = images.to(device, non_blocking=True)
            
            # Handle hierarchical masks
            if isinstance(masks, dict):
                masks_gpu = {k: v.to(device, non_blocking=True) for k, v in masks.items()}
                target_masks = masks['fine'] # Use fine masks for IoU
            else:
                masks_gpu = masks.to(device, non_blocking=True)
                target_masks = masks
            
            # Forward with AMP
            if use_amp:
                with autocast():
                    if use_dataparallel:
                        # Inference mode: no masks, returns predictions only
                        predictions = model(images)
                    else:
                        predictions = model(images)
                    
                    loss_out = criterion(predictions, masks_gpu)
                    if isinstance(loss_out, tuple):
                        loss = loss_out[0]
                    else:
                        loss = loss_out
            else:
                if use_dataparallel:
                    predictions = model(images)
                else:
                    predictions = model(images)
                
                loss_out = criterion(predictions, masks_gpu)
                if isinstance(loss_out, tuple):
                    loss = loss_out[0]
                else:
                    loss = loss_out
            
            epoch_loss += loss.item()
            
            # Store predictions for IoU
            if isinstance(predictions, dict):
                preds = predictions['fine'].argmax(dim=1)
            else:
                preds = predictions.argmax(dim=1)
                
            all_preds.append(preds.cpu())
            all_targets.append(target_masks)
            
            # Progress indicator
            if batch_idx % 2 == 0 or batch_idx == 1:
                print(f"    Batch {batch_idx}/{len(val_loader)}")
    
    # Compute IoU (vectorized on GPU)
    all_preds = torch.cat(all_preds, dim=0).to(device)
    all_targets = torch.cat(all_targets, dim=0).to(device)
    
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


def update_curriculum(model, criterion, epoch, config):
    """
    Update model and loss based on curriculum stage (POC-6 Innovation #5)
    """
    if not config['training'].get('curriculum', False):
        return
    
    # Define stages (hardcoded for POC-6 as per plan)
    # Stage 1: Binary (1-20)
    # Stage 2: Coarse (21-40)
    # Stage 3: Fine (41-100)
    
    if epoch <= 20:
        stage = 'binary'
        freeze = ['head_coarse', 'head_fine']
        weights = {'binary': 1.0, 'coarse': 0.0, 'fine': 0.0}
    elif epoch <= 40:
        stage = 'coarse'
        freeze = ['head_fine']
        weights = {'binary': 0.2, 'coarse': 1.0, 'fine': 0.0}
    else:
        stage = 'fine'
        freeze = []
        weights = {'binary': 0.2, 'coarse': 0.3, 'fine': 1.0}
        
    print(f"  🎓 Curriculum Stage: {stage.upper()} (Epoch {epoch})")
    
    # Update loss weights
    if hasattr(criterion, 'binary_weight'):
        criterion.binary_weight = weights['binary']
        criterion.coarse_weight = weights['coarse']
        criterion.fine_weight = weights['fine']
        print(f"     Loss weights: {weights}")
        
    # Freeze/Unfreeze heads
    # We need to access the underlying model (handle DataParallel)
    if isinstance(model, torch.nn.DataParallel):
        real_model = model.module.model
    elif isinstance(model, ModelWithLoss):
        real_model = model.model
    else:
        real_model = model
    
    # Unfreeze everything first
    for param in real_model.parameters():
        param.requires_grad = True
        
    # Freeze specific heads
    for head_name in freeze:
        if hasattr(real_model, head_name):
            head = getattr(real_model, head_name)
            for param in head.parameters():
                param.requires_grad = False
            print(f"     ❄️  Frozen {head_name}")


def main():
    parser = argparse.ArgumentParser(description='POC-5.9: Production Segmentation Pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    parser.add_argument('--test-epoch', action='store_true',
                       help='Run only 1 epoch for testing')
    parser.add_argument('--manifest', type=str, default=None,
                       help='Path to JSON manifest for DG split')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import sys
    print("="*60)
    print(f"POC-6: Multiclass Segmentation + Domain Generalization")
    print("="*60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Visible GPU count: {torch.cuda.device_count()}")
    
    encoder_name = config['model'].get('encoder_name', config['model'].get('encoder', 'unknown'))
    print(f"Architecture: {config['model'].get('architecture', 'HierarchicalUPerNet')}")
    print(f"Encoder: {encoder_name}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Image size: {config['data']['image_size']}")
    print(f"Mixed Precision: {config['training']['mixed_precision']}")
    print(f"Epochs: {1 if args.test_epoch else config['training']['epochs']}")
    if args.manifest:
        print(f"Manifest: {args.manifest}")
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
            preload_to_gpu=config['data'].get('preload_to_gpu', False),
            manifest_path=args.manifest
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
            use_augmented=config['data'].get('use_augmented', False),
            hierarchical=config['data'].get('hierarchical', False),
            manifest_path=args.manifest
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
    
    # Loss function (POC-5.9: DiceFocalLoss with class weights)
    print("\nCreating loss function...")
    if config['loss']['type'] == 'hierarchical':
        criterion = HierarchicalDiceFocalLoss(
            binary_weight=config['loss'].get('binary_weight', 0.2),
            coarse_weight=config['loss'].get('coarse_weight', 0.3),
            fine_weight=config['loss'].get('fine_weight', 1.0),
            fine_class_weights=class_weights,
            ignore_index=255
        )
        print(f"Loss: HIERARCHICAL (weights: {config['loss']['binary_weight']}/{config['loss']['coarse_weight']}/{config['loss']['fine_weight']})")
    elif config['loss']['type'] == 'dice_focal':
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
    
    # Create checkpoint directory with clean model names
    # Map encoder names to clean model directory names
    model_name_map = {
        'tu-convnext_tiny': 'convnext_tiny',
        'mit_b3': 'segformer_b3',
        'tu-maxvit_tiny_tf_384': 'maxvit_tiny',
        'convnext_tiny': 'convnext_tiny',
        'maxvit_tiny_rw_256': 'maxvit_tiny'
    }
    simple_name = model_name_map.get(encoder_name, encoder_name.replace('tu-', '').replace('/', '_'))
    
    # Use absolute path to avoid issues when running from different directories
    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / 'logs' / 'models' / simple_name
    
    if args.manifest:
        fold_name = Path(args.manifest).stem
        checkpoint_dir = checkpoint_dir / fold_name
        
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Checkpoint directory: {checkpoint_dir}")
    print()
    
    print(f"🚀 Starting epoch loop (1 to {num_epochs})...")
    print()
    for epoch in range(1, num_epochs + 1):
        print(f"\n📅 EPOCH {epoch}/{num_epochs}")
        print("="*60)
        
        # Update curriculum (POC-6)
        update_curriculum(model, criterion, epoch, config)
        
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
