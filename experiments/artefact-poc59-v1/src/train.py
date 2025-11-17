"""
POC-5.9 Training Script
Single-task multiclass segmentation with early stopping and class weights
"""

import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from losses import DiceFocalLoss, compute_class_weights
from dataset import create_dataloaders
from model_factory import create_model


def load_config(config_path: str) -> dict:
    """Load YAML config file with base config inheritance."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle base config inheritance
    if '_base_' in config:
        base_path = Path(config_path).parent / config['_base_']
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configs (config overrides base)
        def merge_dicts(base, override):
            result = base.copy()
            for key, value in override.items():
                if key == '_base_':
                    continue
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        config = merge_dicts(base_config, config)
    
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    """
    Compute mean IoU across classes.
    
    Args:
        preds: (B, H, W) predicted class indices
        targets: (B, H, W) target class indices
        num_classes: Number of classes
        ignore_index: Index to ignore
    
    Returns:
        Mean IoU across valid classes
    """
    ious = []
    
    for cls in range(num_classes):
        # Create masks
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        valid_mask = (targets != ignore_index)
        
        # Intersection and union
        intersection = (pred_mask & target_mask & valid_mask).sum().item()
        union = (pred_mask | target_mask) & valid_mask
        union = union.sum().item()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    return np.mean(ious) if ious else 0.0


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device: str,
    epoch: int,
    config: dict,
    writer: SummaryWriter
):
    """Train for one epoch."""
    import time
    model.train()
    
    total_loss = 0.0
    total_miou = 0.0
    start_time = time.time()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward with AMP
        with autocast(enabled=config['training']['mixed_precision']):
            logits = model(images)
            loss = criterion(logits, masks)
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config['training'].get('gradient_clip'):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )
        
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            miou = compute_miou(
                preds,
                masks,
                config['data']['num_classes'],
                config['data']['ignore_index']
            )
        
        total_loss += loss.item()
        total_miou += miou
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mIoU': f"{miou:.4f}"
        })
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % config['logging']['log_interval'] == 0:
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            writer.add_scalar('train/miou_step', miou, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
    
    # Epoch metrics
    avg_loss = total_loss / len(train_loader)
    avg_miou = total_miou / len(train_loader)
    
    # Time and throughput
    elapsed = time.time() - start_time
    throughput = len(train_loader.dataset) / elapsed
    
    # Step scheduler
    if scheduler is not None:
        scheduler.step()
    
    return avg_loss, avg_miou, elapsed, throughput


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader,
    criterion,
    device: str,
    config: dict
):
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_miou = 0.0
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        logits = model(images)
        loss = criterion(logits, masks)
        
        # Metrics
        preds = logits.argmax(dim=1)
        miou = compute_miou(
            preds,
            masks,
            config['data']['num_classes'],
            config['data']['ignore_index']
        )
        
        total_loss += loss.item()
        total_miou += miou
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mIoU': f"{miou:.4f}"
        })
    
    avg_loss = total_loss / len(val_loader)
    avg_miou = total_miou / len(val_loader)
    
    return avg_loss, avg_miou


def train(config_path: str, fold: int = None, test_mode: bool = False):
    """
    Main training function.
    
    Args:
        config_path: Path to YAML config file
        fold: Fold index for K-fold CV (None for single split)
        test_mode: If True, run only 5 epochs for testing
    """
    # Load config
    config = load_config(config_path)
    
    # Override epochs for test mode
    if test_mode:
        config['training']['epochs'] = 5
        config['early_stopping']['patience'] = 3
        print("⚠️  Test mode: running only 5 epochs")
    
    # Set seeds
    set_seed(config['training']['seed'])
    
    # Setup device
    device = config['hardware']['device']
    if not torch.cuda.is_available() and device == 'cuda':
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    # Create experiment directory
    config_name = Path(config_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if fold is not None:
        exp_name = f"{config_name}_fold{fold}_{timestamp}"
    else:
        exp_name = f"{config_name}_{timestamp}"
    
    if test_mode:
        exp_name = f"{exp_name}_test"
    
    exp_dir = Path(config['logging']['log_dir']) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"\n🚀 Starting experiment: {exp_name}")
    print(f"   Device: {device}")
    print(f"   Fold: {fold if fold is not None else 'single split'}")
    print(f"   Logs: {exp_dir}")
    
    # Create dataloaders
    print("\n📊 Loading dataset...")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Num workers: {config['data']['num_workers']}")
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(config, fold=fold)
    
    # Load pre-computed class weights from metadata
    print("\n⚖️  Loading pre-computed class weights...")
    class_weights_path = Path(config['data']['data_dir']) / 'class_weights.json'
    if class_weights_path.exists():
        import json
        with open(class_weights_path) as f:
            weights_data = json.load(f)
        class_weights = torch.tensor(weights_data['weights'], dtype=torch.float32).to(device)
        print(f"   Loaded from: {class_weights_path}")
        print(f"   Method: {weights_data['method']}, Images: {weights_data['num_images']}")
    else:
        print(f"   ⚠️  Pre-computed weights not found, computing from masks...")
        class_weights = compute_class_weights(
            train_dataset.mask_paths,
            num_classes=config['data']['num_classes'],
            method=config['loss']['class_weights_method'],
            ignore_index=config['data']['ignore_index']
        )
        class_weights = class_weights.to(device)
    
    # Create model
    print("\n🏗️  Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    
    # Create loss
    criterion = DiceFocalLoss(
        dice_weight=config['loss']['dice_weight'],
        focal_weight=config['loss']['focal_weight'],
        smooth=config['loss']['smooth'],
        alpha=config['loss']['focal_alpha'],
        gamma=config['loss']['focal_gamma'],
        ignore_index=config['data']['ignore_index'],
        class_weights=class_weights
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['optimizer']['betas']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['scheduler']['max_lr'],
        epochs=config['training']['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=config['scheduler']['pct_start'],
        anneal_strategy=config['scheduler']['anneal_strategy'],
        div_factor=config['scheduler']['div_factor'],
        final_div_factor=config['scheduler']['final_div_factor']
    )
    
    # AMP scaler
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Tensorboard writer
    writer = SummaryWriter(exp_dir / 'tensorboard')
    
    # Training loop
    print(f"\n🎓 Training for {config['training']['epochs']} epochs...")
    
    best_miou = 0.0
    patience_counter = 0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_miou, train_time, throughput = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, epoch, config, writer
        )
        
        # Validate
        val_loss, val_miou = validate_epoch(
            model, val_loader, criterion, device, config
        )
        
        # GPU memory stats
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated() / 1024**3
            vram_reserved = torch.cuda.max_memory_reserved() / 1024**3
        else:
            vram_allocated = vram_reserved = 0.0
        
        # Log epoch metrics
        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        writer.add_scalar('epoch/train_miou', train_miou, epoch)
        writer.add_scalar('epoch/val_loss', val_loss, epoch)
        writer.add_scalar('epoch/val_miou', val_miou, epoch)
        writer.add_scalar('epoch/throughput', throughput, epoch)
        if torch.cuda.is_available():
            writer.add_scalar('epoch/vram_allocated', vram_allocated, epoch)
        
        print(f"\n📈 Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")
        print(f"   Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
        print(f"   Time: {train_time:.1f}s, Throughput: {throughput:.1f} imgs/s")
        if torch.cuda.is_available():
            print(f"   VRAM: {vram_allocated:.2f}GB allocated / {vram_reserved:.2f}GB peak")
        
        # Check improvement
        improved = val_miou > best_miou + config['early_stopping']['min_delta']
        
        if improved:
            print(f"   ✅ New best mIoU: {val_miou:.4f} (+{val_miou-best_miou:.4f})")
            best_miou = val_miou
            patience_counter = 0
            
            # Save best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_miou': best_miou,
                'config': config
            }, checkpoint_dir / 'best_model.pth')
        else:
            patience_counter += 1
            print(f"   ⏸️  No improvement ({patience_counter}/{config['early_stopping']['patience']})")
        
        # Save last checkpoint
        if config['logging']['save_last']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_miou': best_miou,
                'config': config
            }, checkpoint_dir / 'last_model.pth')
        
        # Early stopping
        if config['early_stopping']['enabled'] and patience_counter >= config['early_stopping']['patience']:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
            print(f"   Best mIoU: {best_miou:.4f}")
            break
    
    writer.close()
    
    print(f"\n✅ Training complete!")
    print(f"   Best mIoU: {best_miou:.4f}")
    print(f"   Checkpoints saved to: {checkpoint_dir}")
    
    return best_miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='POC-5.9 Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--fold', type=int, default=None, help='Fold index for K-fold CV')
    parser.add_argument('--test', action='store_true', help='Test mode (5 epochs only)')
    
    args = parser.parse_args()
    
    train(args.config, args.fold, args.test)
