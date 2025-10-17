from __future__ import annotations
"""Run all three models sequentially on a single input image.

Usage:
    python -m src.run_all --image samples/demo.jpg

This will produce outputs under logs/{model_name}/.
If the provided image is missing or invalid, a demo image is downloaded to logs/demo_input.jpg.
"""

import argparse
import subprocess
from pathlib import Path
import urllib.request
from PIL import Image

MODELS = [
    ("configs/upernet_swin_base.yaml", "upernet_swin_base"),
    ("configs/convnext_tiny_fpn.yaml", "convnext_tiny_fpn"),
    ("configs/maxvit_tiny_fpn.yaml", "maxvit_tiny_fpn"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    args = ap.parse_args()

    def is_image_valid(p: Path) -> bool:
        try:
            with Image.open(p) as im:
                im.verify()
            return True
        except Exception:
            return False

    # Resolve path and validate; fallback to logs/demo_input.jpg
    img_path = Path(args.image)
    if not img_path.exists() or not is_image_valid(img_path):
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        fallback = logs_dir / "demo_input.jpg"
        if not fallback.exists() or not is_image_valid(fallback):
            url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
            print(f"Input image missing/invalid. Downloading demo image to {fallback}...")
            urllib.request.urlretrieve(url, fallback)
        img_path = fallback
    img = str(img_path)
    for cfg, name in MODELS:
        print(f"=== Running {name} ===")
        subprocess.run(["python", "-m", "src.infer_one", "--config", cfg, "--image", img], check=True)

    print("Done. Check logs/ for outputs.")


if __name__ == "__main__":
    main()
