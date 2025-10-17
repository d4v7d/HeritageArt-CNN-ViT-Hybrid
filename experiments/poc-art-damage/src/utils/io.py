"""I/O helpers: load image, save mask/overlay/entropy, save stats JSON.

All outputs are written under logs/{model_name}/ with files named after the input stem.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from .palette import IGNORE_INDEX, N_CLASSES, to_pil_lut


def load_rgb_image(path: str | Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def save_indexed_mask(mask: np.ndarray, out_path: str | Path) -> None:
    """Save uint8 mask with PIL palette (0..15 colored, 255 ignored remains index-only)."""
    mask_img = Image.fromarray(mask.astype(np.uint8), mode="P")
    mask_img.putpalette(to_pil_lut())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(out_path)


def make_overlay(rgb: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    rgb_arr = np.array(rgb).astype(np.float32)
    pal = np.array(to_pil_lut(), dtype=np.uint8).reshape(256, 3)
    color = pal[mask]
    overlay = (1 - alpha) * rgb_arr + alpha * color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def save_entropy(entropy: np.ndarray, out_path: str | Path) -> None:
    """Save entropy map as 8-bit grayscale (rescaled 0..255)."""
    v = entropy
    v = v - v.min()
    if v.max() > 0:
        v = v / (v.max() + 1e-9)
    v = (v * 255.0).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(v, mode="L")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def save_stats(counts: Dict[int, int], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({str(k): int(v) for k, v in counts.items()}, f, indent=2)
