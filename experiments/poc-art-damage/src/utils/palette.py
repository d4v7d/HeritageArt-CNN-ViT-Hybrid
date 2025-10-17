"""Palette and class mapping utilities for ARTeFACT-style 16-class masks.

- Classes 0..15 are valid; 255 is reserved for Background/Ignore.
- We expose a fixed RGB palette for visualization and PIL indexed PNGs.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

N_CLASSES = 16
IGNORE_INDEX = 255

# A visually distinct palette of 16 colors (RGB tuples 0..255)
# You can tune colors later to match paper figures
PALETTE: List[Tuple[int, int, int]] = [
    (0, 0, 0),        # 0 Clean (black)
    (230, 25, 75),    # 1 Material loss
    (60, 180, 75),    # 2 Peel
    (255, 225, 25),   # 3 Dust
    (0, 130, 200),    # 4 Scratch
    (245, 130, 48),   # 5 Hair
    (145, 30, 180),   # 6 Dirt
    (70, 240, 240),   # 7 Fold
    (240, 50, 230),   # 8 Writing
    (210, 245, 60),   # 9 Cracks
    (250, 190, 212),  # 10 Staining
    (0, 128, 128),    # 11 Stamp
    (220, 190, 255),  # 12 Sticker
    (170, 110, 40),   # 13 Puncture
    (255, 250, 200),  # 14 Burn marks
    (128, 0, 0),      # 15 Lightleak
]

assert len(PALETTE) == N_CLASSES


def to_pil_lut() -> list:
    """Return a 768-length list for PIL putpalette (256*3)."""
    pal = np.zeros((256, 3), dtype=np.uint8)
    for i, (r, g, b) in enumerate(PALETTE):
        pal[i] = (r, g, b)
    return pal.flatten().tolist()
