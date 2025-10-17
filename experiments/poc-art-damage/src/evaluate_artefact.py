#!/usr/bin/env python3
"""Evaluate ARTeFACT-trained models on test set.

Computes IoU, Dice, and per-class metrics for trained models.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from .datasets.artefact import ArtefactDataset
from .models.convnext_fpn import ConvNeXtTinyFPN
from .models.maxvit_fpn import MaxViTTinyFPN
from .models.upernet_swin import UPerNetSwinBase16
from .utils.palette import N_CLASSES, IGNORE_INDEX


def build_model(name: str, weights_path: str | None = None) -> torch.nn.Module:
    """Build model and optionally load weights."""
    name = name.lower()
    if name == "convnext_tiny_fpn":
        model = ConvNeXtTinyFPN(num_classes=N_CLASSES)
    elif name == "maxvit_tiny_fpn":
        model = MaxViTTinyFPN(num_classes=N_CLASSES)
    elif name == "upernet_swin_base":
        model = UPerNetSwinBase16(num_classes=N_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")

    if weights_path and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path}")

    return model


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute IoU and Dice coefficients."""
    pred = pred.flatten()
    target = target.flatten()

    # Create masks for valid pixels (not ignore_index)
    valid = target != ignore_index

    ious = []
    dices = []

    for cls in range(num_classes):
        pred_cls = (pred == cls) & valid
        target_cls = (target == cls) & valid

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        dice_den = pred_cls.sum() + target_cls.sum()

        iou = intersection / (union + 1e-6)
        dice = 2 * intersection / (dice_den + 1e-6)

        ious.append(iou.item())
        dices.append(dice.item())

    return torch.tensor(ious), torch.tensor(dices)


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: str) -> dict:
    """Evaluate model on dataloader."""
    model.eval()
    model.to(device)

    all_ious = []
    all_dices = []
    total_pixels = 0
    correct_pixels = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Compute pixel accuracy (excluding ignore_index)
            valid_mask = masks != IGNORE_INDEX
            correct_pixels += ((preds == masks) & valid_mask).sum().item()
            total_pixels += valid_mask.sum().item()

            # Compute IoU and Dice per batch
            batch_ious, batch_dices = compute_iou(preds, masks, N_CLASSES, IGNORE_INDEX)
            all_ious.append(batch_ious)
            all_dices.append(batch_dices)

    # Average over all batches
    mean_ious = torch.stack(all_ious).mean(dim=0)
    mean_dices = torch.stack(all_dices).mean(dim=0)

    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0
    mean_iou = mean_ious.mean().item()
    mean_dice = mean_dices.mean().item()

    results = {
        "pixel_accuracy": pixel_acc,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "per_class_iou": mean_ious.tolist(),
        "per_class_dice": mean_dices.tolist(),
    }

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, choices=["convnext_tiny_fpn", "maxvit_tiny_fpn", "upernet_swin_base"])
    ap.add_argument("--weights", type=str, required=True, help="Path to model weights")
    ap.add_argument("--data", type=str, default="logs/data/artefact", help="Path to ARTeFACT data")
    ap.add_argument("--batch", type=int, default=4, help="Batch size")
    ap.add_argument("--workers", type=int, default=2, help="Number of workers")
    ap.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = build_model(args.model, args.weights)

    # Load test data (using validation split as test)
    metadata_path = Path(args.data) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Simple train/val split (80/20) - use val as test
    total = len(df)
    val_len = max(1, int(0.2 * total))
    train_len = total - val_len

    # Use the same split as training
    torch.manual_seed(42)
    indices = torch.randperm(total).tolist()
    val_indices = indices[train_len:]

    test_df = df.iloc[val_indices].reset_index(drop=True)
    print(f"Test set: {len(test_df)} samples")

    # Create dataset and dataloader
    test_dataset = ArtefactDataset(test_df, size=512, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    # Evaluate
    print(f"Evaluating {args.model}...")
    results = evaluate_model(model, test_loader, device)

    # Print results
    print("\n" + "="*50)
    print(f"Results for {args.model}")
    print("="*50)
    print(".4f")
    print(".4f")
    print(".4f")

    print("\nTop 5 classes by IoU:")
    class_ious = results["per_class_iou"]
    sorted_indices = np.argsort(class_ious)[::-1]
    for i, cls_idx in enumerate(sorted_indices[:5]):
        print(".4f")

    # Save results
    output_file = args.output or f"logs/evaluation/{args.model}_evaluation.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
