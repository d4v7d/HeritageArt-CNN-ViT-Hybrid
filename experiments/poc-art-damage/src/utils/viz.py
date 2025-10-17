"""Visualization helpers."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .palette import to_pil_lut


def overlay_on_image(rgb: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    rgb_arr = np.array(rgb).astype(np.float32)
    pal = np.array(to_pil_lut(), dtype=np.uint8).reshape(256, 3)
    color = pal[mask]
    out = (1 - alpha) * rgb_arr + alpha * color
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
