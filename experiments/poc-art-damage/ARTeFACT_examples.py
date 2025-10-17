"""
ARTeFACT examples – cleaned Python module
=========================================

This file refactors the original notebook-style script from Hugging Face
into a single, organized Python module. All markdown titles were converted
to comments, notebook magics were removed, and code was grouped into
logical sections. Keep everything in one file, but import and run only
what you need.

Sections
--------
- Imports and global settings
- Label mappings and utilities
- Dataset I/O (Hugging Face load + save to disk)
- Dataset splits (LOOCV by content/material)
- Cropping and tiling utilities
- PyTorch dataset and datamodule
- Visualization helpers
- Optional main demo (guarded, does not run by default)

Notes
-----
- Requires: datasets, pandas, pillow, numpy, matplotlib, torch, torchvision,
  pytorch-lightning (optional, only for the DataModule), albumentations (optional).
- Background class in the annotations is 255. For training we remap it to 16
  to have a contiguous range [0..16] where 0 = Clean (no-damage).
"""

# --- Imports and global settings -------------------------------------------------
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Pytorch Lightning is optional; only needed for the DataModule
try:
    import pytorch_lightning as pl  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pl = None  # allows using Dataset independently

# Allow very large images
Image.MAX_IMAGE_PIXELS = 243748701

# Optional: set serif font for plots
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


# --- Label mappings and utilities ------------------------------------------------
# Segmentation labels are saved as a PNG image, where each number from 1 to 15
# corresponds to a damage class like Peel, Scratch etc; the Background class is
# set to 255, and the Clean class (no damage) is set to 0.

name_color_mapping: Dict[str, str] = {
    "Material loss": "#1CE6FF",
    "Peel": "#FF34FF",
    "Dust": "#FF4A46",
    "Scratch": "#008941",
    "Hair": "#006FA6",
    "Dirt": "#A30059",
    "Fold": "#FFA500",
    "Writing": "#7A4900",
    "Cracks": "#0000A6",
    "Staining": "#63FFAC",
    "Stamp": "#004D43",
    "Sticker": "#8FB0FF",
    "Puncture": "#997D87",
    "Background": "#5A0007",
    "Burn marks": "#809693",
    "Lightleak": "#f6ff1b",
}

class_names: List[str] = [
    "Material loss",
    "Peel",
    "Dust",
    "Scratch",
    "Hair",
    "Dirt",
    "Fold",
    "Writing",
    "Cracks",
    "Staining",
    "Stamp",
    "Sticker",
    "Puncture",
    "Burn marks",
    "Lightleak",
    "Background",
]

class_to_id: Dict[str, int] = {class_name: idx + 1 for idx, class_name in enumerate(class_names)}
class_to_id["Background"] = 255  # Set Background ID to 255 per dataset


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


id_to_rgb: Dict[int, Tuple[int, int, int]] = {
    class_to_id[name]: hex_to_rgb(color) for name, color in name_color_mapping.items()
}
id_to_rgb[0] = (0, 0, 0)  # Clean class as black for visualization

# Create id2label and label2id mappings
id2label: Dict[int, str] = {idx: name for name, idx in class_to_id.items()}
label2id: Dict[str, int] = class_to_id.copy()

# Non-damaged pixels
id2label[0] = "Clean"
label2id["Clean"] = 0

# If you remap background 255 -> 16 for training, you have 17 classes in total
NUM_CLASSES = 17


# --- Dataset I/O (Hugging Face load + save to disk) ------------------------------
def load_damaged_media_dataset(split: str = "train"):
    """Load the ARTeFACT dataset from Hugging Face.

    Returns the datasets.Dataset object. Requires `datasets` package.
    """
    try:
        from datasets import load_dataset  # local import to keep optional
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "The 'datasets' package is required to load the ARTeFACT dataset.\n"
            "Install it with: pip install datasets"
        ) from e

    return load_dataset("danielaivanova/damaged-media", split=split)


def save_dataset_to_disk(dataset, target_dir: str) -> pd.DataFrame:
    """Save Hugging Face dataset images and annotations to disk as PNGs.

    Creates a metadata.csv with columns: id, material, content, image_path,
    annotation_path, annotation_rgb_path.
    """
    os.makedirs(target_dir, exist_ok=True)
    csv_path = os.path.join(target_dir, "metadata.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    image_dir = os.path.join(target_dir, "image")
    annotation_dir = os.path.join(target_dir, "annotation")
    annotation_rgb_dir = os.path.join(target_dir, "annotation_rgb")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(annotation_rgb_dir, exist_ok=True)

    rows: List[Dict[str, str]] = []

    for i in range(len(dataset)):
        data = dataset[i]
        id_str = data["id"]
        material_str = data["material"]
        content_str = data["content"]

        image_path = os.path.join(image_dir, f"{id_str}.png")
        annotation_path = os.path.join(annotation_dir, f"{id_str}.png")
        annotation_rgb_path = os.path.join(annotation_rgb_dir, f"{id_str}.png")

        Image.fromarray(np.uint8(data["image"])) .save(image_path)
        Image.fromarray(np.uint8(data["annotation"]), "L").save(annotation_path)
        Image.fromarray(np.uint8(data["annotation_rgb"])) .save(annotation_rgb_path)

        rows.append(
            {
                "id": id_str,
                "material": material_str,
                "content": content_str,
                "image_path": image_path,
                "annotation_path": annotation_path,
                "annotation_rgb_path": annotation_rgb_path,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


# --- Dataset splits (LOOCV by content/material) ---------------------------------
def create_loocv_splits(df: pd.DataFrame, by: str = "content") -> Dict[str, Dict[str, pd.DataFrame]]:
    """Create leave-one-out cross-validation splits grouped by a column.

    Returns a dict of {group_name: {train_set: DataFrame, val_set: DataFrame}}.
    """
    if by not in df.columns:
        raise KeyError(f"Column '{by}' not found in DataFrame")

    grouped = df.groupby(by)
    groups = {name: group for name, group in grouped}
    unique_keys = df[by].unique()

    loocv_splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    for key in unique_keys:
        val_set = groups[key]
        train_set = pd.concat([groups[k] for k in unique_keys if k != key], axis=0)
        loocv_splits[key] = {"train_set": train_set, "val_set": val_set}

    return loocv_splits


# --- Cropping and tiling utilities ----------------------------------------------
def random_square_crop_params(image: Image.Image, target_size: int) -> Tuple[int, int, int, int]:
    width, height = image.size
    min_edge = min(width, height)

    lower_bound = min(min_edge, target_size)
    upper_bound = max(min_edge, target_size)
    crop_size = random.randint(lower_bound, upper_bound)

    if crop_size > width or crop_size > height:
        crop_size = min(width, height)

    x = random.randint(0, width - crop_size)
    y = random.randint(0, height - crop_size)
    return (x, y, x + crop_size, y + crop_size)


def apply_crop_and_resize(image: Image.Image, coords: Tuple[int, int, int, int], target_size: int) -> Image.Image:
    image_crop = image.crop(coords)
    image_crop = image_crop.resize((target_size, target_size), Image.NEAREST)
    return image_crop


def sliding_window(image: Image.Image, target_size: int, overlap: float = 0.2):
    width, height = image.size
    stride = int(target_size * (1 - overlap))
    patches: List[Image.Image] = []
    coordinates: List[Tuple[int, int, int, int]] = []

    for y in range(0, height - target_size + 1, stride):
        for x in range(0, width - target_size + 1, stride):
            coords = (x, y, x + target_size, y + target_size)
            patch = image.crop(coords)
            patches.append(patch)
            coordinates.append(coords)

    return patches, coordinates


# --- PyTorch dataset and datamodule ---------------------------------------------
class CustomDataset(Dataset):
    """Dataset that randomly square-crops training samples; validation keeps full images.

    For validation, images are downsized so the longest edge is <= 1024.
    Background label 255 is remapped to 16 for training convenience.
    """

    def __init__(self, dataframe: pd.DataFrame, target_size: int, is_train: bool = True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.target_size = target_size
        self.is_train = is_train

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        annotation = Image.open(row["annotation_path"]).convert("L")
        annotation_rgb = Image.open(row["annotation_rgb_path"]).convert("RGB")
        sample_id = row["id"]
        material = row["material"]
        content = row["content"]

        if self.is_train:
            crop_coords = random_square_crop_params(image, self.target_size)
            image = apply_crop_and_resize(image, crop_coords, self.target_size)
            annotation = apply_crop_and_resize(annotation, crop_coords, self.target_size)
            annotation_rgb = apply_crop_and_resize(annotation_rgb, crop_coords, self.target_size)
        else:
            max_edge = max(image.size)
            if max_edge > 1024:
                downsample_ratio = 1024 / max_edge
                new_size = tuple(int(dim * downsample_ratio) for dim in image.size)
                image = image.resize(new_size, Image.BILINEAR)
                annotation = annotation.resize(new_size, Image.NEAREST)
                annotation_rgb = annotation_rgb.resize(new_size, Image.BILINEAR)

        image_t = self.to_tensor(image)
        ann_np = np.array(annotation, dtype=np.uint8)
        ann_t = torch.tensor(ann_np, dtype=torch.long)
        ann_rgb_t = self.to_tensor(annotation_rgb)

        # Normalize image
        image_t = self.normalize(image_t)

        # Remap background 255 -> 16
        ann_t[ann_t == 255] = 16

        return {
            "image": image_t,
            "annotation": ann_t,
            "annotation_rgb": ann_rgb_t,
            "id": sample_id,
            "material": material,
            "content": content,
        }


class CustomDataModule(pl.LightningDataModule if pl is not None else object):  # type: ignore[misc]
    """Lightning DataModule wrapping train/val DataLoaders.

    If PyTorch Lightning isn't installed, this class will still construct
    dataloaders but won't integrate with Lightning's Trainer.
    """

    def __init__(
        self,
        loocv_splits: Dict[str, Dict[str, pd.DataFrame]],
        current_content: str,
        target_size: int,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        if pl is not None:
            super().__init__()
        self.loocv_splits = loocv_splits
        self.current_content = current_content
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: CustomDataset
        self.val_dataset: CustomDataset

    def prepare_data(self):  # Lightning hook - no-op here
        pass

    def setup(self, stage: Optional[str] = None):
        train_df = self.loocv_splits[self.current_content]["train_set"]
        val_df = self.loocv_splits[self.current_content]["val_set"].sample(frac=1).reset_index(drop=True)

        self.train_dataset = CustomDataset(dataframe=train_df, target_size=self.target_size, is_train=True)
        self.val_dataset = CustomDataset(dataframe=val_df, target_size=self.target_size, is_train=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):  # not used in this example
        return None


# --- Visualization helpers -------------------------------------------------------
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def denormalize(image: np.ndarray, mean: List[float] = MEAN, std: List[float] = STD) -> np.ndarray:
    img = image.copy()
    for i in range(3):
        img[..., i] = img[..., i] * std[i] + mean[i]
    return img


def visualize_training_batch(train_loader: DataLoader, num_samples: int = 4):
    batch = next(iter(train_loader))
    images = batch["image"]
    anns = batch["annotation"]
    anns_rgb = batch["annotation_rgb"]

    N = min(num_samples, len(images))
    fig, axes = plt.subplots(N, 3, figsize=(15, 5 * N))
    for ax, col in zip(axes[0], ["Image", "Annotation", "Overlay"]):
        ax.set_title(col, fontsize=18)

    for i in range(N):
        img = denormalize(images[i].numpy().transpose(1, 2, 0))
        ann = anns[i].numpy().astype(np.uint8)
        ann_rgb = anns_rgb[i].numpy().transpose(1, 2, 0)

        alpha_channel = np.all(ann_rgb == [0, 0, 0], axis=-1)
        ann_rgba = np.dstack((ann_rgb, np.where(alpha_channel, 0, 1)))

        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(ann_rgb)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(ann_rgba)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_validation_samples(val_loader: DataLoader, num_samples: int = 4):
    val_iter = iter(val_loader)
    batches = []
    for _ in range(num_samples):
        try:
            batches.append(next(val_iter))
        except StopIteration:
            break

    images, anns, anns_rgb, materials, contents = [], [], [], [], []
    for batch in batches:
        images.append(batch["image"].squeeze(0))
        anns.append(batch["annotation"].squeeze(0))
        anns_rgb.append(batch["annotation_rgb"].squeeze(0))
        materials.append(batch["material"][0])
        contents.append(batch["content"][0])

    N = len(images)
    if N == 0:
        print("No validation samples to visualize.")
        return

    fig, axes = plt.subplots(N, 3, figsize=(15, 5 * N))
    for ax, col in zip(axes[0], ["Image", "Annotation", "Overlay"]):
        ax.set_title(col, fontsize=18)

    for i in range(N):
        img = denormalize(images[i].numpy().transpose(1, 2, 0))
        ann_rgb = anns_rgb[i].numpy().transpose(1, 2, 0)

        alpha_channel = np.all(ann_rgb == [0, 0, 0], axis=-1)
        ann_rgba = np.dstack((ann_rgb, np.where(alpha_channel, 0, 1)))

        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(ann_rgb)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(ann_rgba)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


# --- Optional main demo ----------------------------------------------------------
if __name__ == "__main__":
    # Set to True to run a quick end-to-end demo (download, save, split, visualize).
    RUN_DEMO = False

    if not RUN_DEMO:
        print(
            "ARTeFACT_examples.py loaded. Set RUN_DEMO=True in __main__ to run the demo.\n"
            "Tip: install 'datasets' first (pip install datasets) to load the dataset."
        )
    else:
        # 1) Load dataset (requires internet and 'datasets')
        ds = load_damaged_media_dataset(split="train")

        # 2) Save to disk (update the path as needed)
        target_dir = os.path.join(
            os.path.dirname(__file__),
            "data",
            "damage_analogue",
        )
        df = save_dataset_to_disk(ds, target_dir)
        print(f"Saved dataset to: {target_dir} | {len(df)} items")

        # 3) Build LOOCV splits by content
        loocv_splits = create_loocv_splits(df, by="content")
        first_key = next(iter(loocv_splits))

        # 4) Create DataModule and loaders
        dm = CustomDataModule(loocv_splits=loocv_splits, current_content=first_key, target_size=512, batch_size=4)
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        # 5) Visualize a few samples
        visualize_training_batch(train_loader, num_samples=4)
        visualize_validation_samples(val_loader, num_samples=4)
