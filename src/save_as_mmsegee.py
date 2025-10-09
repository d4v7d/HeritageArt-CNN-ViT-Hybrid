from datasets import load_dataset
import PIL
from PIL import Image
import numpy as np, os, pandas as pd 
from pathlib import Path

PIL.Image.MAX_IMAGE_PIXELS = 243748701

# Resolve paths relative to this script, not terminal
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CKPTS = ROOT / "checkpoints"
OUT = ROOT / "src/data" # Temporary staging before split

print("Loading dataset...")
print("cwd:", Path.cwd())
print("script dir:", HERE)
print("root:", ROOT)
print("output folder:", OUT)

os.makedirs(f"{OUT}/artefact_raw/images", exist_ok = True)
os.makedirs(f"{OUT}artefact_raw/masks", exist_ok = True)

# This only needs to be run once, if you have downloaded the dataset just set it to None
dataset = load_dataset("danielaivanova/damaged-media", split="train")
print("\nFinished loading dataset\n")

def save_dataset_to_disk(dataset, target_dir):
    csv_path = os.path.join(target_dir, 'metadata.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    # Create the directories for saving images and annotations
    image_dir = os.path.join(target_dir, 'image')
    annotation_dir = os.path.join(target_dir, 'annotation')
    annotation_rgb_dir = os.path.join(target_dir, 'annotation_rgb')

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(annotation_rgb_dir, exist_ok=True)

    # Initialize an empty DataFrame to store metadata
    df = pd.DataFrame(columns=['id', 'material', 'content', 'image_path', 'annotation_path', 'annotation_rgb_path'])

    for i in range(len(dataset)):
        data = dataset[i]
        id_str = data['id']
        material_str = data['material']
        content_str = data['content']

        # Create the file paths
        image_path = os.path.join(image_dir, f"{id_str}.png")
        annotation_path = os.path.join(annotation_dir, f"{id_str}.png")
        annotation_rgb_path = os.path.join(annotation_rgb_dir, f"{id_str}.png")

        # Save the images in high quality
        Image.fromarray(np.uint8(data['image'])).save(image_path)
        Image.fromarray(np.uint8(data['annotation']), 'L').save(annotation_path)
        Image.fromarray(np.uint8(data['annotation_rgb'])).save(annotation_rgb_path)

        # # Append the data to DataFrame
        # df = df.append({
        #     'id': id_str,
        #     'material': material_str,
        #     'content': content_str,
        #     'image_path': image_path,
        #     'annotation_path': annotation_path,
        #     'annotation_rgb_path': annotation_rgb_path
        # }, ignore_index=True)

        new_row = pd.DataFrame([{
        'id': id_str,
        'material': material_str,
        'content': content_str,
        'image_path': image_path,
        'annotation_path': annotation_path,
        'annotation_rgb_path': annotation_rgb_path
        }])

        # Concatenate with the existing DataFrame
        df = pd.concat([df, new_row], ignore_index=True)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)
    return df

target_dir = OUT / "artefact_dataframe"
df = save_dataset_to_disk(dataset, str(target_dir))
print("\nFinished saving dataset to disk\n")


# rows = []
# for i in range(len(dataset)):
#     print(f"Processing image {i}/418")
#     ex = dataset[i]
#     img = Image.fromarray(np.uint8(ex["image"])).convert("RGB")
#     ann = Image.fromarray(np.uint8(ex["annotation"]))  # single-channel IDs

#     img_path = f"{OUT}/artefact_raw/images/{ex['id']}.png"
#     mask_path = f"{OUT}/artefact_raw/masks/{ex['id']}.png"

#     img.save(img_path)
#     # IMPORTANT: keep label IDs intact (0=Clean, 1..15 damages, 255=Background)
#     ann.save(mask_path)

#     rows.append({"id": ex["id"], "material": ex["material"], "content": ex["content"],
#                  "image_path": img_path, "mask_path": mask_path})

# pd.DataFrame(rows).to_csv(f"{OUT}/artefact_raw/metadata.csv", index=False)
# print("Wrote", len(rows), "files to", OUT + "/artefact_raw")
