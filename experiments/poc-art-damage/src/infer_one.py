from __future__ import annotations
"""CLI to run one model on a single image with Hann-blended sliding-window inference.

Outputs:
- logs/{model}/<stem>_mask.png (indexed palette)
- logs/{model}/<stem>_overlay.png
- logs/{model}/<stem>_entropy.png
- logs/{model}/<stem>_stats.json

Usage:
  python -m src.infer_one --config configs/convnext_tiny_fpn.yaml --image samples/demo.jpg
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf

from .data.tiling import sliding_window_logits
from .models.convnext_fpn import ConvNeXtTinyFPN
from .models.maxvit_fpn import MaxViTTinyFPN
from .models.upernet_swin import UPerNetSwinBase16
from .utils.io import load_rgb_image, make_overlay, save_entropy, save_indexed_mask, save_stats
from .utils.palette import IGNORE_INDEX, N_CLASSES
from .utils.seed import set_seed


def build_model(name: str) -> torch.nn.Module:
    name = name.lower()
    if name == "convnext_tiny_fpn":
        return ConvNeXtTinyFPN(num_classes=N_CLASSES)
    if name == "maxvit_tiny_fpn":
        return MaxViTTinyFPN(num_classes=N_CLASSES)
    if name == "upernet_swin_base":
        return UPerNetSwinBase16(num_classes=N_CLASSES)
    raise ValueError(f"Unknown model: {name}")


def preprocess(img: Image.Image, size_hw: tuple[int, int] | None = None) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std
    x = torch.from_numpy(arr).unsqueeze(0)  # [1,3,H,W]
    return x


def softmax_entropy(logits: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    p = torch.softmax(logits, dim=dim)
    ent = -(p * torch.clamp(p.log(), min=-1e9)).sum(dim=dim)
    return ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--weights", type=str, default=None, help="Optional path to .pth weights for the model")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(int(cfg.get("seed", 42)))

    model_name = cfg.model.name
    tile = int(cfg.common.tile)
    overlap = float(cfg.common.overlap)
    stride = int(tile * (1 - overlap))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name).to(device)
    if args.weights and Path(args.weights).exists():
        sd = torch.load(args.weights, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded weights from {args.weights}")
    model.eval()

    # Load image
    img = load_rgb_image(args.image)
    x = preprocess(img).to(device)

    # Define prediction function
    def predict(tile_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = model(tile_tensor)
        return logits

    logits = sliding_window_logits(x, predict, tile=tile, stride=stride, num_classes=N_CLASSES, device=device)

    # Final outputs
    mask = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    # Clamp to 0..15 just in case (no 255 prediction in this PoC)
    mask = np.clip(mask, 0, N_CLASSES - 1)

    ent = softmax_entropy(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)

    # Paths
    stem = Path(args.image).stem
    out_dir = Path("logs") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_indexed_mask(mask, out_dir / f"{stem}_mask.png")
    overlay = make_overlay(img, mask)
    overlay.save(out_dir / f"{stem}_overlay.png")
    save_entropy(ent, out_dir / f"{stem}_entropy.png")

    # Stats: per-class pixel counts
    counts = {i: int((mask == i).sum()) for i in range(N_CLASSES)}
    save_stats(counts, out_dir / f"{stem}_stats.json")

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
