import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
# Add parent of parent to path (for src imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_factory import create_model
from preload_dataset import create_preloaded_dataloaders

# ARTeFACT 16 classes
CLASS_NAMES = [
    'Clean', 'Material_loss', 'Peel', 'Dust', 'Scratch',
    'Hair', 'Dirt', 'Fold', 'Writing', 'Cracks',
    'Staining', 'Stamp', 'Sticker', 'Puncture', 'Burn_marks', 'Lightleak'
]

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_iou_per_class(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:
    ious = []
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union.item() == 0:
            iou = float('nan')
        else:
            iou = (intersection / union).item()
        ious.append(iou)
    return np.array(ious)

def evaluate_model(config_path: str, manifest_path: str, checkpoint_path: str = None):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model name logic
    encoder_name = config['model'].get('encoder_name', config['model'].get('encoder', 'unknown'))
    model_name_map = {
        'tu-convnext_tiny': 'convnext_tiny',
        'mit_b3': 'segformer_b3',
        'tu-maxvit_tiny_tf_384': 'maxvit_tiny',
        'convnext_tiny': 'convnext_tiny',
        'maxvit_tiny_rw_256': 'maxvit_tiny'
    }
    simple_name = model_name_map.get(encoder_name, encoder_name.replace('tu-', '').replace('/', '_'))
    
    # Checkpoint logic
    if checkpoint_path is None:
        project_root = Path(__file__).parent.parent
        checkpoint_dir = project_root / 'logs' / 'models' / simple_name
        fold_name = Path(manifest_path).stem
        checkpoint_dir = checkpoint_dir / fold_name
        checkpoint_path = checkpoint_dir / 'best_model.pth'
    
    print(f"Evaluating: {simple_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Manifest: {manifest_path}")
    
    if not Path(checkpoint_path).exists():
        print("❌ Checkpoint not found!")
        return

    # Load Model
    model = create_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Dataloader
    _, val_loader = create_preloaded_dataloaders(
        data_root=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=4,
        use_augmented=config['data'].get('use_augmented', False),
        preload_to_gpu=False,
        manifest_path=manifest_path
    )
    
    all_preds = []
    all_targets = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            if isinstance(masks, dict):
                target_masks = masks['fine']
            else:
                target_masks = masks
            
            predictions = model(images)
            if isinstance(predictions, dict):
                preds = predictions['fine'].argmax(dim=1)
            else:
                preds = predictions.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_targets.append(target_masks)
            
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    iou_per_class = compute_iou_per_class(all_preds, all_targets, config['model']['classes'])
    
    print("\nPer-class IoU:")
    print("-" * 30)
    for i, name in enumerate(CLASS_NAMES):
        if i < len(iou_per_class):
            val = iou_per_class[i]
            print(f"{name:<15} {val:.4f}")
            
    valid_mask = ~np.isnan(iou_per_class)
    miou = np.mean(iou_per_class[valid_mask])
    print("-" * 30)
    print(f"mIoU: {miou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    evaluate_model(args.config, args.manifest)
