from __future__ import annotations

"""Minimal ARTeFACT fine-tuning script for the PoC.

Trains a ConvNeXt-Tiny + FPN head for 16 classes (0..15) with ignore_index=255.
Saves a checkpoint .pth with model.state_dict(). Kept intentionally simple.
"""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ..datasets.artefact import ensure_data, ArtefactDataset
from ..models.convnext_fpn import ConvNeXtTinyFPN
from ..models.maxvit_fpn import MaxViTTinyFPN
from ..models.upernet_swin import UPerNetSwinBase16


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
    ap.add_argument("--out", type=str, default="logs/checkpoints", help="Output directory for checkpoints")
    ap.add_argument("--data", type=str, default="logs/data/artefact", help="ARTeFACT data root")
    ap.add_argument("--model", type=str, default="convnext_tiny_fpn", 
                   choices=["convnext_tiny_fpn", "maxvit_tiny_fpn", "upernet_swin_base"],
                   help="Model to train")
    ap.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    ap.add_argument("--batch", type=int, default=4, help="Batch size")
    ap.add_argument("--size", type=int, default=512, help="Image size")
    ap.add_argument("--workers", type=int, default=2, help="Number of workers")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--max_train_batches", type=int, default=None, help="Limit batches per epoch (None for all)")
    ap.add_argument("--max_val_batches", type=int, default=None, help="Limit validation batches")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure data exists
    df = ensure_data(args.data)

    # Simple random split train/val (80/20) for PoC
    total = len(df)
    val_len = max(1, int(0.2 * total))
    train_len = total - val_len
    train_df, val_df = random_split(df, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    # random_split returns Subset; recover DataFrames by iloc indices
    train_df = df.iloc[train_df.indices].reset_index(drop=True)
    val_df = df.iloc[val_df.indices].reset_index(drop=True)

    train_ds = ArtefactDataset(train_df, size=args.size, train=True)
    val_ds = ArtefactDataset(val_df, size=args.size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers)

    # Select model
    if args.model == "convnext_tiny_fpn":
        model = ConvNeXtTinyFPN(num_classes=16).to(device)
        model_name = "convnext_tiny_fpn"
    elif args.model == "maxvit_tiny_fpn":
        model = MaxViTTinyFPN(num_classes=16).to(device)
        model_name = "maxvit_tiny_fpn"
    elif args.model == "upernet_swin_base":
        model = UPerNetSwinBase16(num_classes=16).to(device)
        model_name = "upernet_swin_base"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = out_dir / f"{model_name}_artefact.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optim, device, criterion, args.max_train_batches)
        va_loss = validate(model, val_loader, device, criterion, args.max_val_batches)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} - train_loss={tr_loss:.4f} val_loss={va_loss:.4f} ({dt:.1f}s)")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best checkpoint to {best_path}")

    # Save final too
    final_path = out_dir / f"{model_name}_artefact_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
