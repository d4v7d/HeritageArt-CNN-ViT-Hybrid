#!/usr/bin/env python3
"""Download a small subset of ARTeFACT dataset efficiently."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.artefact import ensure_data

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--samples', type=int, default=30, help='Number of samples to download')
    parser.add_argument('--use-mock', action='store_true', help='Use mock data instead of real')
    args = parser.parse_args()
    
    print(f"Downloading {args.samples} samples to {args.out}")
    
    df = ensure_data(args.out, use_mock=args.use_mock, max_samples=args.samples)
    print(f"\nSuccess! Downloaded {len(df)} samples")
    print(f"Metadata saved to: {args.out}/metadata.csv")
    
    # Verify files exist
    missing = 0
    for _, row in df.iterrows():
        if not os.path.exists(row['image_path']):
            missing += 1
    
    if missing > 0:
        print(f"Warning: {missing} image files are missing!")
    else:
        print("All image files verified successfully")

if __name__ == '__main__':
    main()
