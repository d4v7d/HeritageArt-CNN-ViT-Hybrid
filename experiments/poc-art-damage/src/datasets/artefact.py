"""ARTeFACT dataset utilities for the PoC.

Capabilities:
- Load dataset from Hugging Face (datasets) or local clone
- Save images/masks to disk with a metadata.csv
- Simple LOOCV splits by 'content' or 'material'
- PyTorch Dataset that remaps background 255 -> 16 and normalizes ImageNet

This module is intentionally minimal and self-contained for the PoC.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

Image.MAX_IMAGE_PIXELS = 243748701


# ------------------------------ I/O -------------------------------------------

def load_hf(split: str = "train", max_retries: int = 5):
    """Load ARTeFACT dataset from Hugging Face with retry logic."""
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError("Install 'datasets' to load ARTeFACT: pip install datasets") from e
    
    for attempt in range(max_retries):
        try:
            print(f"Loading ARTeFACT dataset (attempt {attempt + 1}/{max_retries})...")
            # Use streaming to reduce memory usage
            return load_dataset("danielaivanova/damaged-media", split=split, streaming=False)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to load dataset after {max_retries} attempts: {e}")
                raise
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            import time
            time.sleep(5 ** attempt)  # Exponential backoff


def save_to_disk(dataset, target_dir: str) -> pd.DataFrame:
    os.makedirs(target_dir, exist_ok=True)
    csv_path = os.path.join(target_dir, "metadata.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    image_dir = os.path.join(target_dir, "image")
    ann_dir = os.path.join(target_dir, "annotation")
    ann_rgb_dir = os.path.join(target_dir, "annotation_rgb")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(ann_rgb_dir, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for i in range(len(dataset)):
        data = dataset[i]
        id_str = data["id"]
        material = data["material"]
        content = data["content"]

        image_path = os.path.join(image_dir, f"{id_str}.png")
        ann_path = os.path.join(ann_dir, f"{id_str}.png")
        ann_rgb_path = os.path.join(ann_rgb_dir, f"{id_str}.png")

        Image.fromarray(np.uint8(data["image"])) .save(image_path)
        Image.fromarray(np.uint8(data["annotation"]), "L").save(ann_path)
        Image.fromarray(np.uint8(data["annotation_rgb"])) .save(ann_rgb_path)

        rows.append({
            "id": id_str,
            "material": material,
            "content": content,
            "image_path": image_path,
            "annotation_path": ann_path,
            "annotation_rgb_path": ann_rgb_path,
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def save_to_disk_streaming(dataset, target_dir: str, max_samples: int) -> pd.DataFrame:
    """Save dataset to disk using streaming to avoid memory issues."""
    os.makedirs(target_dir, exist_ok=True)
    csv_path = os.path.join(target_dir, "metadata.csv")
    
    # If CSV already exists, just return it
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    image_dir = os.path.join(target_dir, "image")
    ann_dir = os.path.join(target_dir, "annotation")
    ann_rgb_dir = os.path.join(target_dir, "annotation_rgb")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(ann_rgb_dir, exist_ok=True)

    rows: List[Dict[str, str]] = []
    
    print(f"Saving {max_samples} samples to {target_dir}...")
    for i, data in enumerate(dataset):
        if i >= max_samples:
            break
            
        try:
            id_str = data["id"]
            material = data.get("material", "unknown")
            content = data.get("content", "unknown")

            image_path = os.path.join(image_dir, f"{id_str}.png")
            ann_path = os.path.join(ann_dir, f"{id_str}.png")
            ann_rgb_path = os.path.join(ann_rgb_dir, f"{id_str}.png")

            # Save images
            Image.fromarray(np.uint8(data["image"])).save(image_path)
            Image.fromarray(np.uint8(data["annotation"]), "L").save(ann_path)
            Image.fromarray(np.uint8(data["annotation_rgb"])).save(ann_rgb_path)

            rows.append({
                "id": id_str,
                "material": material,
                "content": content,
                "image_path": image_path,
                "annotation_path": ann_path,
                "annotation_rgb_path": ann_rgb_path,
            })
            
            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"  Saved {i + 1}/{max_samples} samples")
                
        except Exception as e:
            print(f"Warning: Failed to save sample {i}: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples to {csv_path}")
    return df


# ------------------------------ Splits ---------------------------------------

def loocv_splits(df: pd.DataFrame, by: str = "content") -> Dict[str, Dict[str, pd.DataFrame]]:
    if by not in df.columns:
        raise KeyError(f"Column '{by}' not found in DataFrame")
    groups = {name: g for name, g in df.groupby(by)}
    keys = list(groups.keys())
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for k in keys:
        val = groups[k]
        train = pd.concat([groups[x] for x in keys if x != k], axis=0)
        out[k] = {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True)}
    return out


# ------------------------------ Dataset --------------------------------------

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ArtefactDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int = 512, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.train = train
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=MEAN, std=STD)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        ann = Image.open(row["annotation_path"]).convert("L")

        # Simple center-crop/resize to square for PoC
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        box = (left, top, left + s, top + s)
        img = img.crop(box).resize((self.size, self.size), Image.BILINEAR)
        ann = ann.crop(box).resize((self.size, self.size), Image.NEAREST)

        x = self.normalize(self.to_tensor(img))
        y = torch.from_numpy(np.array(ann, dtype=np.uint8)).long()
        # Keep 255 as ignore_index to match PoC's 16-class setup (0..15 valid)

        return {"image": x, "mask": y, "id": row["id"]}


# ------------------------------ Helpers --------------------------------------

def ensure_data(target_dir: str, use_mock: bool = False, max_samples: int | None = None) -> pd.DataFrame:
    """Ensure ARTeFACT data exists, downloading if needed or using mock data."""
    metadata_path = os.path.join(target_dir, "metadata.csv")
    
    if os.path.exists(metadata_path):
        print(f"ARTeFACT data already exists at {target_dir}")
        df = pd.read_csv(metadata_path)
        
        # Verify that image files exist
        if len(df) > 0:
            # Check in the image subdirectory
            image_dir = os.path.join(target_dir, "image")
            if os.path.exists(image_dir):
                image_files = os.listdir(image_dir)
                if len(image_files) > 0:
                    print(f"Loaded {len(df)} samples from existing dataset ({len(image_files)} image files found)")
                    return df
            
            print("Warning: Metadata exists but image files not found, re-downloading...")
    
    if use_mock:
        print("Using mock ARTeFACT data for testing...")
        return save_mock_data(target_dir, max_samples or 50)
    
    try:
        print("Attempting to download ARTeFACT dataset from Hugging Face...")
        ds = load_hf("train")
        
        # Limit samples if specified  
        if max_samples is not None:
            print(f"Limiting to {max_samples} samples")
            # Use streaming to avoid loading all data into memory
            return save_to_disk_streaming(ds, target_dir, max_samples)
        
        return save_to_disk_streaming(ds, target_dir, len(ds))
    except Exception as e:
        print(f"Failed to download ARTeFACT dataset: {e}")
        print("Falling back to mock data for testing...")
        return save_mock_data(target_dir, max_samples or 50)


def save_mock_data(target_dir: str, num_samples: int = 20) -> pd.DataFrame:
    """Create mock ARTeFACT data for testing when download fails."""
    print(f"Creating mock ARTeFACT data in {target_dir} with {num_samples} samples")
    
    target_path = Path(target_dir)
    image_dir = target_path / "image"
    ann_dir = target_path / "annotation"
    ann_rgb_dir = target_path / "annotation_rgb"
    
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_rgb_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    materials = ["painting", "photo", "print", "drawing"]
    contents = ["portrait", "landscape", "abstract", "still-life"]
    
    for i in range(num_samples):
        id_str = f"mock_{i:04d}"
        material = materials[i % len(materials)]
        content = contents[i % len(contents)]
        
        # Create mock image (512x512 RGB with some variation)
        base_color = np.random.randint(100, 200, 3)
        noise = np.random.randint(-50, 50, (512, 512, 3))
        img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        
        # Create mock annotation (some random classes 0-15 and background 255)
        ann = np.random.randint(0, 16, (512, 512), dtype=np.uint8)
        # Add some background pixels (about 20%)
        mask = np.random.rand(512, 512) < 0.2
        ann[mask] = 255
        
        # Create mock RGB annotation
        ann_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        for cls in range(16):
            ann_rgb[ann == cls] = [cls * 15, 128, 255 - cls * 15]
        
        image_path = image_dir / f"{id_str}.png"
        ann_path = ann_dir / f"{id_str}.png"
        ann_rgb_path = ann_rgb_dir / f"{id_str}.png"
        
        Image.fromarray(img).save(image_path)
        Image.fromarray(ann).save(ann_path)
        Image.fromarray(ann_rgb).save(ann_rgb_path)
        
        rows.append({
            "id": id_str,
            "material": material,
            "content": content,
            "image_path": str(image_path),
            "annotation_path": str(ann_path),
            "annotation_rgb_path": str(ann_rgb_path),
        })
    
    df = pd.DataFrame(rows)
    metadata_path = target_path / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Created {num_samples} mock samples")
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/artefact", help="Target directory to save ARTeFACT")
    ap.add_argument("--mock", action="store_true", help="Use mock data instead of downloading from Hugging Face")
    ap.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to download (for testing)")
    args = ap.parse_args()

    df = ensure_data(args.out, use_mock=args.mock, max_samples=args.max_samples)
    print(f"Saved ARTeFACT to {args.out} | {len(df)} items")
