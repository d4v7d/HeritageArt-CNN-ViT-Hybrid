#!/usr/bin/env python3
"""Complete end-to-end pipeline for ARTeFACT training and evaluation.

This script:
1. Downloads ARTeFACT dataset (only once)
2. Trains 3 models: CNN (ConvNeXt), ViT (Swin), Hybrid (MaxViT)
3. Evaluates each model and saves metrics
4. Generates segmentation maps and visualizations for each test image
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from .datasets.artefact import ensure_data, ArtefactDataset
from .models.convnext_fpn import ConvNeXtTinyFPN
from .models.maxvit_fpn import MaxViTTinyFPN
from .models.upernet_swin import UPerNetSwinBase16
from .utils.palette import N_CLASSES, IGNORE_INDEX, PALETTE


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> dict:
    """Compute F1, mIoU, and per-class metrics."""
    pred = pred.flatten()
    target = target.flatten()
    valid = target != ignore_index

    ious = []
    f1_scores = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls) & valid
        target_cls = (target == cls) & valid
        
        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()
        
        intersection = tp
        union = tp + fp + fn
        
        iou = (intersection / (union + 1e-6)).item()
        precision = (tp / (tp + fp + 1e-6)).item()
        recall = (tp / (tp + fn + 1e-6)).item()
        f1 = (2 * precision * recall / (precision + recall + 1e-6))
        
        ious.append(iou)
        f1_scores.append(f1)
    
    return {
        'mIoU': np.mean(ious),
        'mF1': np.mean(f1_scores),
        'per_class_iou': ious,
        'per_class_f1': f1_scores
    }


def visualize_prediction(image: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, save_dir: Path, image_id: str):
    """Save visualization of prediction vs ground truth."""
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image.cpu() * std + mean
    image = (image * 255).byte().permute(1, 2, 0).numpy()
    
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Create colored segmentation maps
    pred_colored = np.zeros((*pred.shape, 3), dtype=np.uint8)
    target_colored = np.zeros((*target.shape, 3), dtype=np.uint8)
    
    for cls in range(N_CLASSES):
        pred_colored[pred == cls] = PALETTE[cls]
        if cls < len(PALETTE):
            target_colored[target == cls] = PALETTE[cls]
    
    # Handle ignore index
    target_colored[target == IGNORE_INDEX] = [0, 0, 0]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(target_colored)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(pred_colored)
    axes[1, 0].set_title('Prediction')
    axes[1, 0].axis('off')
    
    # Overlay
    overlay = (image * 0.6 + pred_colored * 0.4).astype(np.uint8)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f"{image_id}_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual maps
    Image.fromarray(pred_colored).save(save_dir / f"{image_id}_pred.png")
    Image.fromarray(target_colored).save(save_dir / f"{image_id}_gt.png")
    Image.fromarray(overlay).save(save_dir / f"{image_id}_overlay.png")


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                device: str, epochs: int, lr: float, save_path: Path) -> nn.Module:
    """Train model with validation."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    return model


def evaluate_and_visualize(model: nn.Module, test_loader: DataLoader, device: str, 
                           output_dir: Path, model_name: str) -> dict:
    """Evaluate model and generate visualizations for each image."""
    model.eval()
    model.to(device)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    overall_ious = []
    overall_f1s = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            image_ids = batch['id']
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Process each image in batch (typically batch_size=1 for testing)
            for i in range(len(images)):
                pred = preds[i]
                target = masks[i]
                image = images[i]
                image_id = image_ids[i] if isinstance(image_ids, list) else image_ids[0]
                
                # Compute metrics
                metrics = compute_metrics(pred, target, N_CLASSES, IGNORE_INDEX)
                metrics['image_id'] = image_id
                all_metrics.append(metrics)
                
                overall_ious.append(metrics['mIoU'])
                overall_f1s.append(metrics['mF1'])
                
                # Visualize
                visualize_prediction(image, pred, target, output_dir, image_id)
                
                # Save metrics per image
                with open(output_dir / f"{image_id}_metrics.json", 'w') as f:
                    json.dump({
                        'image_id': image_id,
                        'mIoU': metrics['mIoU'],
                        'mF1': metrics['mF1'],
                        'per_class_iou': metrics['per_class_iou'],
                        'per_class_f1': metrics['per_class_f1']
                    }, f, indent=2)
    
    # Compute overall metrics
    overall_metrics = {
        'model_name': model_name,
        'mean_mIoU': np.mean(overall_ious),
        'std_mIoU': np.std(overall_ious),
        'mean_mF1': np.mean(overall_f1s),
        'std_mF1': np.std(overall_f1s),
        'num_images': len(all_metrics),
        'per_image_metrics': all_metrics
    }
    
    # Save overall metrics
    with open(output_dir / 'overall_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    print(f"\n{model_name} Results:")
    print(f"  Mean mIoU: {overall_metrics['mean_mIoU']:.4f} ± {overall_metrics['std_mIoU']:.4f}")
    print(f"  Mean mF1:  {overall_metrics['mean_mF1']:.4f} ± {overall_metrics['std_mF1']:.4f}")
    
    return overall_metrics


def main():
    parser = argparse.ArgumentParser(description="Complete ARTeFACT Pipeline")
    parser.add_argument('--data-dir', type=str, default='logs/data/artefact_real',
                        help='Directory for ARTeFACT dataset')
    parser.add_argument('--output-dir', type=str, default='logs/pipeline_results',
                        help='Directory for all outputs')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to download from ARTeFACT (None for all)')
    parser.add_argument('--max-eval-samples', type=int, default=20,
                        help='Maximum samples to evaluate (to avoid too many visualizations)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download if already exists')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use existing checkpoints')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download/Load Dataset
    print("\n" + "="*80)
    print("STEP 1: Loading ARTeFACT Dataset")
    print("="*80)
    
    df = ensure_data(args.data_dir, use_mock=False, max_samples=args.max_samples)
    print(f"Loaded {len(df)} samples from ARTeFACT")
    
    # Step 2: Split dataset
    print("\n" + "="*80)
    print("STEP 2: Splitting Dataset")
    print("="*80)
    
    total = len(df)
    test_len = min(args.max_eval_samples, max(1, int(0.2 * total)))
    val_len = max(1, int(0.1 * total))
    train_len = total - test_len - val_len
    
    indices = list(range(total))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:train_len + val_len + test_len]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = ArtefactDataset(train_df, size=512, train=True)
    val_dataset = ArtefactDataset(val_df, size=512, train=False)
    test_dataset = ArtefactDataset(test_df, size=512, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Step 3: Define models
    models_config = [
        ('ConvNeXt-Tiny-FPN', 'convnext_tiny_fpn', ConvNeXtTinyFPN(num_classes=N_CLASSES)),
        ('Swin-Base-UPerNet', 'upernet_swin_base', UPerNetSwinBase16(num_classes=N_CLASSES)),
        ('MaxViT-Tiny-FPN', 'maxvit_tiny_fpn', MaxViTTinyFPN(num_classes=N_CLASSES)),
    ]
    
    results_summary = []
    
    for display_name, model_name, model in models_config:
        print("\n" + "="*80)
        print(f"Processing Model: {display_name} ({model_name})")
        print("="*80)
        
        checkpoint_path = checkpoints_dir / f"{model_name}_artefact.pth"
        
        # Step 4: Training
        if not args.skip_training:
            print(f"\nTraining {display_name}...")
            model = train_model(
                model, train_loader, val_loader, device, 
                args.epochs, args.lr, checkpoint_path
            )
        else:
            if checkpoint_path.exists():
                print(f"Loading existing checkpoint from {checkpoint_path}")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            else:
                print(f"Warning: Checkpoint not found at {checkpoint_path}, using untrained model")
        
        # Step 5: Evaluation and Visualization
        print(f"\nEvaluating {display_name}...")
        eval_output_dir = output_dir / model_name
        metrics = evaluate_and_visualize(model, test_loader, device, eval_output_dir, model_name)
        results_summary.append(metrics)
    
    # Step 6: Save summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    summary_path = output_dir / 'summary_results.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\nAll models comparison:")
    for result in results_summary:
        print(f"\n{result['model_name']}:")
        print(f"  mIoU: {result['mean_mIoU']:.4f} ± {result['std_mIoU']:.4f}")
        print(f"  mF1:  {result['mean_mF1']:.4f} ± {result['std_mF1']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
