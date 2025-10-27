"""
Hierarchical loss functions for multi-task segmentation.
Combines binary, coarse, and fine-grained predictions with class weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for segmentation tasks.
    Handles class imbalance through smooth coefficient and optional class weights.
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.class_weights = class_weights
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        
        Returns:
            Scalar loss value
        """
        # Softmax to get probabilities
        probs = F.softmax(predictions, dim=1)  # (B, C, H, W)
        
        # Create ignore mask
        valid_mask = (targets != self.ignore_index).float()  # (B, H, W)
        
        # One-hot encode targets
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(
            targets.clamp(0, num_classes - 1),
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Apply ignore mask
        probs = probs * valid_mask.unsqueeze(1)
        targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1)
        
        # Compute Dice coefficient per class
        dims = (0, 2, 3)  # Reduce over batch, height, width
        intersection = (probs * targets_one_hot).sum(dim=dims)  # (C,)
        cardinality = (probs + targets_one_hot).sum(dim=dims)  # (C,)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score  # (C,)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            dice_loss = dice_loss * self.class_weights.to(dice_loss.device)
        
        return dice_loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.class_weights = class_weights
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        
        Returns:
            Scalar loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            predictions,
            targets,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights
        )  # (B, H, W)
        
        # Compute probabilities
        probs = F.softmax(predictions, dim=1)  # (B, C, H, W)
        
        # Get probabilities for target classes
        num_classes = predictions.shape[1]
        targets_clamped = targets.clamp(0, num_classes - 1)
        pt = probs.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)  # (B, H, W)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Focal loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Only compute loss on valid pixels
        valid_mask = (targets != self.ignore_index).float()
        focal_loss = (focal_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        return focal_loss


class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal loss for single-task segmentation.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        smooth: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(
            smooth=smooth,
            ignore_index=ignore_index,
            class_weights=class_weights
        )
        
        self.focal_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            ignore_index=ignore_index,
            class_weights=class_weights
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        
        Returns:
            Scalar loss value
        """
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        
        return self.dice_weight * dice + self.focal_weight * focal


class HierarchicalDiceFocalLoss(nn.Module):
    """
    Hierarchical multi-task loss for binary + coarse + fine segmentation.
    
    Loss = w_binary * L_binary + w_coarse * L_coarse + w_fine * L_fine
    
    Each component uses Dice + Focal loss with class weighting.
    """
    
    def __init__(
        self,
        binary_weight: float = 0.2,
        coarse_weight: float = 0.3,
        fine_weight: float = 1.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        smooth: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
        binary_class_weights: Optional[torch.Tensor] = None,
        coarse_class_weights: Optional[torch.Tensor] = None,
        fine_class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.binary_weight = binary_weight
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
        
        # Binary loss (2 classes: Clean vs Damage)
        self.binary_loss = DiceFocalLoss(
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            smooth=smooth,
            alpha=alpha,
            gamma=gamma,
            ignore_index=ignore_index,
            class_weights=binary_class_weights
        )
        
        # Coarse loss (4 groups)
        self.coarse_loss = DiceFocalLoss(
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            smooth=smooth,
            alpha=alpha,
            gamma=gamma,
            ignore_index=ignore_index,
            class_weights=coarse_class_weights
        )
        
        # Fine loss (16 classes)
        self.fine_loss = DiceFocalLoss(
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            smooth=smooth,
            alpha=alpha,
            gamma=gamma,
            ignore_index=ignore_index,
            class_weights=fine_class_weights
        )
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: Dict with keys 'binary', 'coarse', 'fine'
                Each value: (B, C, H, W) logits
            targets: Dict with keys 'binary', 'coarse', 'fine'
                Each value: (B, H, W) class indices
        
        Returns:
            total_loss: Scalar weighted sum of all losses
            loss_dict: Dict with individual loss values for logging
        """
        # Compute individual losses
        loss_binary = self.binary_loss(predictions['binary'], targets['binary'])
        loss_coarse = self.coarse_loss(predictions['coarse'], targets['coarse'])
        loss_fine = self.fine_loss(predictions['fine'], targets['fine'])
        
        # Weighted sum
        total_loss = (
            self.binary_weight * loss_binary +
            self.coarse_weight * loss_coarse +
            self.fine_weight * loss_fine
        )
        
        # Return total loss and components for logging
        loss_dict = {
            'loss_binary': loss_binary.item(),
            'loss_coarse': loss_coarse.item(),
            'loss_fine': loss_fine.item(),
            'loss_total': total_loss.item()
        }
        
        return total_loss, loss_dict


def compute_class_weights(
    dataset,
    num_classes: int,
    method: str = 'inverse_sqrt',
    ignore_index: int = 255
) -> torch.Tensor:
    """
    Compute class weights from dataset to handle class imbalance.
    
    Args:
        dataset: Dataset object with targets
        num_classes: Number of classes
        method: 'inverse', 'inverse_sqrt', or 'effective_samples'
        ignore_index: Class index to ignore
    
    Returns:
        Tensor of shape (num_classes,) with normalized weights
    """
    # Count pixels per class
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    print(f"Computing class weights for {len(dataset)} samples...")
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        
        # Handle dict targets (hierarchical)
        if isinstance(target, dict):
            target = target['fine']  # Use fine-grained labels
        
        # Count valid pixels (not ignore_index)
        valid_mask = target != ignore_index
        for c in range(num_classes):
            class_counts[c] += (target[valid_mask] == c).sum()
    
    # Compute weights based on method
    if method == 'inverse':
        # Weight inversely proportional to frequency
        weights = 1.0 / (class_counts + 1e-8)
    elif method == 'inverse_sqrt':
        # Weight inversely proportional to sqrt(frequency) - less extreme
        weights = 1.0 / (np.sqrt(class_counts) + 1e-8)
    elif method == 'effective_samples':
        # Effective number of samples (from Class-Balanced Loss paper)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
    elif method is None or method == 'None':
        # No class weights
        weights = np.ones(num_classes, dtype=np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights to sum to num_classes
    weights = weights / weights.sum() * num_classes
    
    # Convert to tensor
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # Print statistics
    print(f"\nClass distribution and weights ({method}):")
    for c in range(num_classes):
        print(f"  Class {c:2d}: {int(class_counts[c]):10d} pixels, weight: {weights[c]:.4f}")
    
    return weights


if __name__ == '__main__':
    """Test hierarchical loss computation."""
    
    # Mock predictions and targets
    batch_size = 2
    height, width = 64, 64
    
    predictions = {
        'binary': torch.randn(batch_size, 2, height, width),
        'coarse': torch.randn(batch_size, 4, height, width),
        'fine': torch.randn(batch_size, 16, height, width)
    }
    
    targets = {
        'binary': torch.randint(0, 2, (batch_size, height, width)),
        'coarse': torch.randint(0, 4, (batch_size, height, width)),
        'fine': torch.randint(0, 16, (batch_size, height, width))
    }
    
    # Create loss function
    loss_fn = HierarchicalDiceFocalLoss()
    
    # Compute loss
    total_loss, loss_dict = loss_fn(predictions, targets)
    
    print("Loss computation test:")
    print(f"  Total loss: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Hierarchical loss test passed!")
