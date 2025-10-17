"""
Evaluation script for POC-4
Generates metrics and visualizations for trained model
"""

import os
import sys
from pathlib import Path
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).parent))
from dataset import prepare_dataloaders, get_transforms
from train import compute_metrics


# ARTeFACT color palette for visualization
PALETTE = {
    0: [0, 0, 0],         # Clean - Black
    1: [255, 0, 0],       # Damage - Red
    255: [128, 128, 128]  # Ignore - Gray
}


def visualize_prediction(image, mask_gt, mask_pred, save_path=None):
    """Create 4-panel visualization: Image, GT, Pred, Overlay."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Denormalize image (move to CPU first)
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)
    image_denorm = image[0] * std[:, None, None] + mean[:, None, None]
    image_denorm = torch.clip(image_denorm, 0, 1).permute(1, 2, 0).cpu().numpy()
    
    # Convert masks to RGB
    def mask_to_rgb(mask):
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in PALETTE.items():
            rgb[mask == cls] = color
        return rgb
    
    mask_gt_rgb = mask_to_rgb(mask_gt.cpu().numpy())
    mask_pred_rgb = mask_to_rgb(mask_pred.cpu().numpy())
    
    # Plot
    axes[0, 0].imshow(image_denorm)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_gt_rgb)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(mask_pred_rgb)
    axes[1, 0].set_title('Prediction')
    axes[1, 0].axis('off')
    
    # Overlay
    overlay = image_denorm.copy()
    overlay[mask_pred.cpu().numpy() == 1] = [1, 0, 0]  # Red for damage
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (Red = Damage)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate_model(model, loader, device, output_dir, max_visualizations=10):
    """Evaluate model and generate visualizations."""
    
    model.eval()
    
    all_metrics = []
    viz_count = 0
    
    viz_dir = Path(output_dir) / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nEvaluating model...")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Compute metrics
            metrics = compute_metrics(outputs, masks)
            all_metrics.append(metrics)
            
            # Visualize first N samples
            if viz_count < max_visualizations:
                for i in range(min(images.size(0), max_visualizations - viz_count)):
                    viz_path = viz_dir / f'sample_{batch_idx}_{i}.png'
                    visualize_prediction(images, masks[i], preds[i], viz_path)
                    viz_count += 1
                    
                    if viz_count >= max_visualizations:
                        break
    
    # Aggregate metrics
    final_metrics = {
        'miou': np.mean([m['miou'] for m in all_metrics]),
        'mf1': np.mean([m['mf1'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'miou_std': np.std([m['miou'] for m in all_metrics]),
        'mf1_std': np.std([m['mf1'] for m in all_metrics]),
        'num_samples': len(all_metrics)
    }
    
    return final_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate ARTeFACT segmentation model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='logs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--max-viz', type=int, default=10,
                       help='Maximum number of visualizations')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights=None,  # We load trained weights
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Best mIoU: {checkpoint.get('best_miou', 'N/A')}")
    
    # Prepare validation data
    _, val_loader = prepare_dataloaders(config)
    
    # Evaluate
    metrics = evaluate_model(model, val_loader, device, args.output, args.max_viz)
    
    # Save metrics
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"  mIoU:     {metrics['miou']:.4f} ± {metrics['miou_std']:.4f}")
    print(f"  mF1:      {metrics['mf1']:.4f} ± {metrics['mf1_std']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Samples:  {metrics['num_samples']}")
    print("="*80)
    print(f"\nMetrics saved to: {metrics_file}")
    print(f"Visualizations saved to: {output_dir / 'visualizations'}")


if __name__ == '__main__':
    main()
