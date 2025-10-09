import pandas as pd, os, shutil
from sklearn.model_selection import train_test_split

RAW = "data/artefact_raw"
DST = "data/artefact"
os.makedirs(f"{DST}/images/train", exist_ok=True)
os.makedirs(f"{DST}/images/val", exist_ok=True)
os.makedirs(f"{DST}/images/test", exist_ok=True)
os.makedirs(f"{DST}/masks/train", exist_ok=True)
os.makedirs(f"{DST}/masks/val", exist_ok=True)
os.makedirs(f"{DST}/masks/test", exist_ok=True)

df = pd.read_csv(f"{RAW}/metadata.csv")

# Simple split (e.g., 70/15/15). Replace with content-LOOCV later
train_df, hold = train_test_split(df, test_size=0.30, random_state=42, stratify=df["content"])
val_df, test_df = train_test_split(hold, test_size=0.50, random_state=42, stratify=hold["content"])

def copy_rows(rows, split):
    for _, r in rows.iterrows():
        shutil.copy(r["image_path"], f"{DST}/images/{split}/{r['id']}.png")
        shutil.copy(r["mask_path"],  f"{DST}/masks/{split}/{r['id']}.png")

copy_rows(train_df, "train")
copy_rows(val_df,   "val")
copy_rows(test_df,  "test")
print("Done. Train/Val/Test =", len(train_df), len(val_df), len(test_df))
