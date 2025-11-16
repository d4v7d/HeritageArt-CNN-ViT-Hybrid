#!/usr/bin/env python3
"""
Pre-load ARTeFACT dataset into shared memory (/dev/shm)
This allows multiple training jobs to read from RAM without duplicating the preload step.

Usage:
    python preload_shared_dataset.py --data-dir ../data/artefact --output-dir /dev/shm/artefact_cache
"""

import argparse
import shutil
import time
from pathlib import Path
from tqdm import tqdm


def preload_dataset(data_dir: Path, output_dir: Path, use_augmented: bool = True):
    """
    Copy dataset to shared memory location.
    
    Args:
        data_dir: Source dataset directory
        output_dir: Destination in /dev/shm
        use_augmented: Use augmented dataset
    """
    
    # Determine source
    if use_augmented:
        source = data_dir / "artefact_augmented"
        if not source.exists():
            source = data_dir.parent.parent / "artefact-poc55-multiclass" / "data" / "artefact_augmented"
    else:
        source = data_dir / "artefact"
    
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    
    print(f"📦 Source: {source}")
    print(f"💾 Destination: {output_dir}")
    print()
    
    # Check available space in /dev/shm
    shm_stat = shutil.disk_usage("/dev/shm")
    print(f"💽 /dev/shm space:")
    print(f"   Total: {shm_stat.total / 1e9:.1f} GB")
    print(f"   Used:  {shm_stat.used / 1e9:.1f} GB")
    print(f"   Free:  {shm_stat.free / 1e9:.1f} GB")
    print()
    
    # Estimate dataset size
    total_size = sum(f.stat().st_size for f in source.rglob('*') if f.is_file())
    print(f"📊 Dataset size: {total_size / 1e9:.2f} GB")
    
    if total_size > shm_stat.free:
        raise RuntimeError(f"Not enough space in /dev/shm! Need {total_size/1e9:.2f} GB, have {shm_stat.free/1e9:.2f} GB")
    
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images and masks
    for subdir in ['images', 'annotations']:
        src_subdir = source / subdir
        dst_subdir = output_dir / subdir
        
        if not src_subdir.exists():
            print(f"⚠️  Skipping {subdir} (not found)")
            continue
        
        dst_subdir.mkdir(parents=True, exist_ok=True)
        
        files = list(src_subdir.glob('*'))
        print(f"🔥 Copying {len(files)} files from {subdir}/")
        
        start_time = time.time()
        
        for src_file in tqdm(files, desc=f"  {subdir}", unit="file"):
            dst_file = dst_subdir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
        
        elapsed = time.time() - start_time
        print(f"   ✅ Done in {elapsed:.1f}s ({len(files)/elapsed:.1f} files/s)")
        print()
    
    # Verify
    cached_files = sum(1 for _ in output_dir.rglob('*') if _.is_file())
    print(f"✅ Pre-loaded {cached_files} files to shared memory")
    print(f"💾 Path: {output_dir}")
    print()
    print("🚀 Training jobs can now use:")
    print(f"   data_dir: {output_dir}")
    print()


def cleanup_cache(output_dir: Path):
    """Remove cached dataset from /dev/shm"""
    if output_dir.exists() and str(output_dir).startswith('/dev/shm'):
        print(f"🗑️  Removing cache: {output_dir}")
        shutil.rmtree(output_dir)
        print("✅ Cache cleaned up")
    else:
        print(f"⚠️  Path not in /dev/shm or doesn't exist: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pre-load dataset to shared memory")
    parser.add_argument('--data-dir', type=Path, default=Path('../data/artefact'),
                        help='Source dataset directory')
    parser.add_argument('--output-dir', type=Path, default=Path('/dev/shm/artefact_cache'),
                        help='Destination in /dev/shm')
    parser.add_argument('--use-augmented', action='store_true', default=True,
                        help='Use augmented dataset')
    parser.add_argument('--cleanup', action='store_true',
                        help='Cleanup cache instead of preloading')
    
    args = parser.parse_args()
    
    print("="*60)
    print("POC-5.8: Shared Memory Dataset Pre-loader")
    print("="*60)
    print()
    
    if args.cleanup:
        cleanup_cache(args.output_dir)
    else:
        preload_dataset(args.data_dir, args.output_dir, args.use_augmented)
    
    print("="*60)


if __name__ == '__main__':
    main()
