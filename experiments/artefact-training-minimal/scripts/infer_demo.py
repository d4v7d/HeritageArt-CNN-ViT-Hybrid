"""
Single image inference demo
"""

import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


PALETTE = {
    0: [0, 0, 0],         # Clean - Black
    1: [255, 0, 0],       # Damage - Red  
    255: [128, 128, 128]  # Ignore - Gray
}


def load_model(checkpoint_path, config):
    """Load trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights=None,
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device


def preprocess_image(image_path):
    """Load and preprocess image."""
    image = np.array(Image.open(image_path).convert('RGB'))
    
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image, image_tensor


def mask_to_rgb(mask):
    """Convert mask to RGB using palette."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in PALETTE.items():
        rgb[mask == cls] = color
    return rgb


@torch.no_grad()
def infer_single_image(model, image_tensor, device):
    """Run inference on single image."""
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    pred = torch.argmax(output, dim=1)[0].cpu().numpy()
    return pred


def visualize_result(original_image, pred_mask, save_path=None):
    """Visualize inference result."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Prediction mask
    pred_rgb = mask_to_rgb(pred_mask)
    axes[1].imshow(pred_rgb)
    axes[1].set_title('Prediction\n(Red = Damage, Black = Clean)')
    axes[1].axis('off')
    
    # Overlay
    overlay = original_image.copy().astype(np.float32) / 255.0
    damage_mask = pred_mask == 1
    overlay[damage_mask] = [1, 0, 0]  # Red for damage
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Single image inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization (optional)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print(f"Loading model from: {args.checkpoint}")
    model, device = load_model(args.checkpoint, config)
    print(f"Using device: {device}")
    
    print(f"\nProcessing image: {args.image}")
    original_image, image_tensor = preprocess_image(args.image)
    
    print("Running inference...")
    pred_mask = infer_single_image(model, image_tensor, device)
    
    # Calculate damage percentage
    total_pixels = (pred_mask != 255).sum()
    damage_pixels = (pred_mask == 1).sum()
    damage_pct = 100 * damage_pixels / total_pixels if total_pixels > 0 else 0
    
    print(f"\nResults:")
    print(f"  Damage pixels: {damage_pixels:,} / {total_pixels:,} ({damage_pct:.2f}%)")
    
    # Visualize
    output_path = args.output or 'logs/demo_inference.png'
    visualize_result(original_image, pred_mask, output_path)


if __name__ == '__main__':
    main()
