"""
POC-60: Hierarchical Training Utilities

Helper functions for training hierarchical multi-task models.
Handles dict outputs {' binary', 'coarse', 'fine'} and hierarchical loss computation.
"""

import torch
import torch.nn as nn
from dataset import fine_to_binary, fine_to_coarse
from hierarchical_losses import HierarchicalDiceFocalLoss


def create_hierarchical_criterion(config: dict, device: torch.device):
    """
    Create hierarchical loss function
    
    Args:
        config: Config dict with loss parameters
        device: Device to place criterion on
    
    Returns:
        HierarchicalDiceFocalLoss instance
    """
    loss_config = config.get('loss', {})
    
    criterion = HierarchicalDiceFocalLoss(
        alpha=loss_config.get('alpha', 0.25),
        gamma=loss_config.get('gamma', 2.0),
        dice_weight=loss_config.get('dice_weight', 0.7),
        focal_weight=loss_config.get('focal_weight', 0.3),
        binary_weight=loss_config.get('binary_weight', 0.2),
        coarse_weight=loss_config.get('coarse_weight', 0.3),
        fine_weight=loss_config.get('fine_weight', 1.0),
        smooth=loss_config.get('smooth', 1.0),
        ignore_index=255
    )
    
    return criterion.to(device)


def compute_hierarchical_metrics(predictions: dict, targets: torch.Tensor, num_classes: int = 16):
    """
    Compute mIoU for all 3 hierarchical heads
    
    Args:
        predictions: Dict with keys 'binary', 'coarse', 'fine' (logits)
        targets: Ground truth fine labels (B, H, W)
        num_classes: Number of fine classes (16)
    
    Returns:
        Dict with binary_miou, coarse_miou, fine_miou
    """
    device = targets.device
    
    # Convert fine targets to binary and coarse
    binary_targets = fine_to_binary(targets, ignore_index=255)
    coarse_targets = fine_to_coarse(targets, ignore_index=255)
    
    # Get predictions (argmax)
    binary_preds = predictions['binary'].argmax(dim=1)  # (B, H, W)
    coarse_preds = predictions['coarse'].argmax(dim=1)
    fine_preds = predictions['fine'].argmax(dim=1)
    
    # Compute IoU for each head
    binary_miou = compute_miou(binary_preds, binary_targets, num_classes=2, ignore_index=255)
    coarse_miou = compute_miou(coarse_preds, coarse_targets, num_classes=4, ignore_index=255)
    fine_miou = compute_miou(fine_preds, targets, num_classes=num_classes, ignore_index=255)
    
    return {
        'binary_miou': binary_miou,
        'coarse_miou': coarse_miou,
        'fine_miou': fine_miou
    }


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    """
    Compute mean IoU
    
    Args:
        preds: (B, H, W) predicted class indices
        targets: (B, H, W) ground truth class indices
        num_classes: Number of classes
        ignore_index: Index to ignore (255)
    
    Returns:
        Mean IoU as float
    """
    ious = []
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Create valid mask
    valid_mask = (targets != ignore_index)
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
    
    return sum(ious) / len(ious) if len(ious) > 0 else 0.0


def is_hierarchical_model(model) -> bool:
    """
    Check if model is hierarchical (returns dict) or standard (returns tensor)
    
    Args:
        model: Model instance
    
    Returns:
        True if hierarchical, False otherwise
    """
    # Check model type or config
    if hasattr(model, 'module'):  # DataParallel wrapper
        model = model.module
    
    # Check if it's HierarchicalUPerNet
    return model.__class__.__name__ == 'HierarchicalUPerNet'
