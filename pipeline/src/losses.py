"""
Single-task loss functions for POC-5.9 multiclass segmentation.
Combines Dice + Focal loss with class weighting (no multi-head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
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
        
        # Ensure targets are long tensor for one_hot
        targets = targets.long()
        
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
        # Ensure targets are long tensor for cross_entropy
        targets = targets.long()
        
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
    Combined Dice + Focal loss for single-task multiclass segmentation.
    
    POC-5.9: Single-head design (16 classes) with class weighting.
    Simpler than POC-5.5's multi-head approach, easier to interpret.
    
    Loss = dice_weight * DiceLoss + focal_weight * FocalLoss
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
        """
        Args:
            dice_weight: Weight for Dice loss component (default: 0.5)
            focal_weight: Weight for Focal loss component (default: 0.5)
            smooth: Smoothing factor for Dice loss (default: 1.0)
            alpha: Focal loss alpha parameter (default: 0.25)
            gamma: Focal loss gamma parameter (default: 2.0)
            ignore_index: Class index to ignore (default: 255 for background)
            class_weights: Per-class weights (C,) tensor (optional)
        """
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


def compute_class_weights(
    mask_paths,
    num_classes: int = 16,
    method: str = 'inverse_sqrt',
    ignore_index: int = 255
) -> torch.Tensor:
    """
    Compute class weights from mask files (NOT full dataset) to avoid RAM overload.
    
    POC-5.9 uses inverse_sqrt by default (less extreme than inverse).
    
    Args:
        mask_paths: List of paths to mask PNG files
        num_classes: Number of classes (default: 16 for ARTeFACT)
        method: 'inverse', 'inverse_sqrt', or 'effective_samples'
        ignore_index: Class index to ignore (default: 255)
    
    Returns:
        Tensor of shape (num_classes,) with normalized weights
    """
    from PIL import Image
    
    # Count pixels per class
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    print(f"Computing class weights from {len(mask_paths)} masks...")
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))
        
        # Count valid pixels (not ignore_index)
        valid_mask = mask != ignore_index
        for c in range(num_classes):
            class_counts[c] += (mask[valid_mask] == c).sum()
    
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
        # No class weights (uniform)
        weights = np.ones(num_classes, dtype=np.float32)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'inverse', 'inverse_sqrt', or 'effective_samples'")
    
    # Normalize weights to sum to num_classes
    weights = weights / weights.sum() * num_classes
    
    # Convert to tensor
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # Print statistics
    print(f"\nClass distribution and weights (method={method}):")
    print("Class | Pixel Count | Weight")
    print("------|-------------|-------")
    for c in range(num_classes):
        print(f"  {c:2d}  | {int(class_counts[c]):11d} | {weights[c]:6.4f}")
    print(f"\nTotal pixels: {int(class_counts.sum())}")
    
    return weights


# Class names for ARTeFACT dataset (POC-5.9)
CLASS_NAMES = [
    "Clean",           # 0
    "Material_loss",   # 1
    "Peel",            # 2
    "Dust",            # 3
    "Scratch",         # 4
    "Hair",            # 5
    "Dirt",            # 6
    "Fold",            # 7
    "Writing",         # 8
    "Cracks",          # 9
    "Staining",        # 10
    "Stamp",           # 11
    "Sticker",         # 12
    "Puncture",        # 13
    "Burn_marks",      # 14
    "Lightleak"        # 15
]


if __name__ == '__main__':
    """Test single-head loss computation."""
    
    # Mock predictions and targets
    batch_size = 2
    num_classes = 16
    height, width = 256, 256
    
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Add some ignore pixels
    targets[0, :10, :10] = 255
    
    # Create mock class weights
    class_weights = torch.rand(num_classes)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    # Test Dice loss
    dice_fn = DiceLoss(class_weights=class_weights)
    dice_loss = dice_fn(predictions, targets)
    print(f"Dice Loss: {dice_loss.item():.4f}")
    
    # Test Focal loss
    focal_fn = FocalLoss(class_weights=class_weights)
    focal_loss = focal_fn(predictions, targets)
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    # Test combined Dice+Focal loss
    combined_fn = DiceFocalLoss(class_weights=class_weights)
    combined_loss = combined_fn(predictions, targets)
    print(f"Dice+Focal Loss: {combined_loss.item():.4f}")
    
    print("\n✅ Single-head loss test passed!")
