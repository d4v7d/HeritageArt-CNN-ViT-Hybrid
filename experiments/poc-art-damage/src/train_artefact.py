#!/usr/bin/env python3
"""Train any of the 3 models on ARTeFACT dataset. -> This must be really for all 3 models"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .datasets.artefact import ensure_data, ArtefactDataset
from .models.convnext_fpn import ConvNeXtTinyFPN
from .models.maxvit_fpn import MaxViTTinyFPN
from .models.upernet_swin import UPerNetSwinBase16
from .utils.palette import N_CLASSES, IGNORE_INDEX


def build_model(name: str) -> torch.nn.Module:
    """Build model by name."""
    name = name.lower()
    if name == "convnext_tiny_fpn":
        return ConvNeXtTinyFPN(num_classes=N_CLASSES)
    elif name == "maxvit_tiny_fpn":
        return MaxViTTinyFPN(num_classes=N_CLASSES)
    elif name == "upernet_swin_base":
        return UPerNetSwinBase16(num_classes=N_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model, loader, optim, device, criterion, max_batches: int | None = None):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        total_loss += float(loss.item())
    n = max(1, (i + 1 if max_batches is None else min(len(loader), max_batches)))
    return total_loss / n


@torch.no_grad()
def validate(model, loader, device, criterion, max_batches: int | None = None):
    model.eval()
    total_loss = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item())
    n = max(1, (i + 1 if max_batches is None else min(len(loader), max_batches)))
    return total_loss / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, choices=["convnext_tiny_fpn", "maxvit_tiny_fpn", "upernet_swin_base"])
    ap.add_argument("--out", type=str, default="logs/checkpoints", help="Output directory for checkpoints")
    ap.add_argument("--data", type=str, default="logs/data/artefact", help="ARTeFACT data root")
    ap.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    ap.add_argument("--batch", type=int, default=4, help="Batch size")
    ap.add_argument("--size", type=int, default=512, help="Image size")
    ap.add_argument("--workers", type=int, default=2, help="Number of workers")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--max_train_batches", type=int, default=None, help="Limit batches per epoch")
    ap.add_argument("--max_val_batches", type=int, default=None, help="Limit validation batches")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {args.model} on {device}")

    # Ensure data exists
    df = ensure_data(args.data)
    print(f"Dataset size: {len(df)} samples")

    # Stratified train/val split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['material']
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_ds = ArtefactDataset(train_df.reset_index(drop=True), size=args.size, train=True)
    val_ds = ArtefactDataset(val_df.reset_index(drop=True), size=args.size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = out_dir / f"{args.model}_artefact.pth"

    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optim, device, criterion, args.max_train_batches)
        va_loss = validate(model, val_loader, device, criterion, args.max_val_batches)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} - train_loss={tr_loss:.4f} val_loss={va_loss:.4f} ({dt:.1f}s)")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"  → Saved best checkpoint to {best_path}")

    # Save final too
    final_path = out_dir / f"{args.model}_artefact_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final checkpoint to {final_path}")

    print(f"\nTraining completed! Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    main()