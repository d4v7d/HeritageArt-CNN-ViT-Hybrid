#!/usr/bin/env python3
"""
Quick script to inspect parquet file structure without heavy dependencies.
"""

import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pandas as pd
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("WARNING: pyarrow not installed, will use pandas only")

parquet_file = Path("artefact_repo/data/train-00000-of-00028.parquet")

if not parquet_file.exists():
    print(f"ERROR: File not found: {parquet_file}")
    sys.exit(1)

print("="*80)
print("PARQUET FILE INSPECTION")
print("="*80)
print(f"File: {parquet_file}")
print(f"Size: {parquet_file.stat().st_size / (1024**2):.1f} MB")
print()

if HAS_PYARROW:
    print("Reading with PyArrow...")
    table = pq.read_table(parquet_file)
    
    print("\nSchema:")
    print(table.schema)
    
    print(f"\nRows: {table.num_rows}")
    print(f"Columns: {table.num_columns}")
    
    print("\nColumn Names:")
    for col in table.column_names:
        print(f"  - {col}")
    
    print("\nFirst row sample (without image data):")
    df = table.to_pandas()
    first_row = df.iloc[0]
    for col in df.columns:
        if col in ['image', 'annotation', 'annotation_rgb']:
            val = first_row[col]
            if hasattr(val, 'size'):
                print(f"  {col}: PIL.Image {val.size if hasattr(val, 'size') else 'unknown size'}")
            else:
                print(f"  {col}: {type(val).__name__}")
        else:
            print(f"  {col}: {first_row[col]}")
    
    print(f"\n📸 Sample image info:")
    if 'image' in df.columns:
        img = df.iloc[0]['image']
        print(f"  Type: {type(img)}")
        if hasattr(img, 'size'):
            print(f"  Size: {img.size}")
            print(f"  Mode: {img.mode if hasattr(img, 'mode') else 'N/A'}")
            print(f"  Pixels: {img.size[0] * img.size[1]:,}")

else:
    print("Reading with pandas...")
    df = pd.read_parquet(parquet_file)
    
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nInfo:")
    print(df.info())
    print(f"\nFirst row (excluding images):")
    print(df.drop(columns=['image', 'annotation', 'annotation_rgb'], errors='ignore').head(1))

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)
