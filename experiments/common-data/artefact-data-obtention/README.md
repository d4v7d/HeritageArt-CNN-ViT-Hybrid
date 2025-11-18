# ARTeFACT Data Obtention - HuggingFace Streaming Approach

**Code example** demonstrating how to download ARTeFACT using HuggingFace `datasets` library with streaming mode.

**Note**: This approach has memory limitations with very large images.

## Project Structure

```
common-data/
├── artefact-data-obtention/   # This directory (download scripts)
│   ├── README.md
│   ├── docker/
│   ├── scripts/
│   │   └── download_artefact.py
│   └── Makefile
├── artefact/                  # Downloaded original dataset (417 samples)
│   ├── images/
│   ├── annotations/
│   ├── annotations_rgb/
│   └── metadata.csv
└── artefact_augmented/        # Augmented dataset (1458 samples)
    ├── images/
    ├── annotations/
    └── class_weights_balanced.json
```

## Purpose

Provides clean, documented code showing how to integrate ARTeFACT with HuggingFace ecosystem for training pipelines.

## Approach: HuggingFace Datasets Library

- **Streaming mode** - Downloads samples incrementally without loading entire dataset  
- **Automatic image resizing** - Reduces memory footprint (max 512-1024px)  
- **Progress tracking** - Real-time progress with tqdm  
- **Rich metadata** - Exports metadata, statistics, and visualizations  
- **Docker containerized** - Fully isolated environment  
- **16GB RAM tested** - Successfully processes 417/418 samples (99.8%)

## Quick Start

### Using Docker (Recommended)

```bash
# Build image (first time only)
make build

# Download full dataset to ../artefact/ (417 samples, requires 16GB RAM)
make download-full

# Or download specific amounts:
make download-small   # 10 samples for testing
make download-medium  # 50 samples

# Or use docker-compose directly:
cd docker
docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py \
  --output ../artefact \
  --all
```

### Custom Commands

```bash
# Download specific number of samples to custom location
docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py \
  --output ../artefact_test \
  --max-samples 100

# Download all samples to ../artefact/ (default location)
docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py \
  --output ../artefact \
  --all
```

## Files

```
artefact-data-obtention/
├── docker/
│   ├── Dockerfile           # Container definition
│   ├── docker-compose.yml   # Docker compose config
│   └── requirements.txt     # Python dependencies
├── scripts/
│   └── download_artefact.py # Main script with streaming mode
├── Makefile                # Convenience commands
├── .gitignore             # Git ignore rules
└── data/                  # Output directory (generated)
    └── demo/              # Demo dataset (10 samples)
```

### download_artefact.py

**Features:**
- Streaming download (incremental processing)
- Automatic image resizing (max 512px)
- Progress tracking with tqdm
- Metadata and statistics export
- Sample visualizations (4-panel)
- Docker containerized for isolation

**Limitations:**
- Crashes on extremely large images (>50M pixels)
- Successfully processes 417/418 samples with 16GB RAM
- Memory constraints with very large images

## Output Structure

Original ARTeFACT dataset downloaded to `../artefact/`:
```
../artefact/
├── images/              # Resized original images (PNG, max 512px)
├── annotations/         # Grayscale damage masks (0-15 classes, 255=background)
├── annotations_rgb/     # Colored damage visualizations
├── visualizations/      # 4-panel sample visualizations (first 10)
├── metadata.csv         # Sample IDs, paths, descriptions, sizes
└── statistics.json      # Dataset statistics (materials, contents, class distribution)
```

This becomes the source for generating `../artefact_augmented/` (1458 samples).

## Test Dataset

The `data/demo/` directory contains a clean end-to-end test run of 10 samples demonstrating the complete output structure:
- 10 samples successfully processed (1 skipped due to 133M pixel size)
- Materials: Parchment (6), Film emulsion (3), Glass (1)
- Content: Artistic (5), Photographic (4), Line art (1)
- Complete visualizations and statistics included

## Known Limitations

### Memory Constraints with Large Images

**Problem:** The HuggingFace `datasets` library decodes images fully in memory when accessing `sample['image']`, BEFORE any resizing can occur. The ARTeFACT dataset contains extremely large images (up to 7680×4877 = 37M pixels, some even >100M pixels) that can exceed available system memory.

**What happens:**
- Small-medium images (< 20M pixels): Process successfully
- Large images (20-50M pixels): May work depending on available RAM
- Huge images (> 50M pixels): Crash WSL/system (OOM)

**Solutions attempted:**
1. Streaming mode (`streaming=True`) - Still decodes individual images fully
2. PIL thumbnail/resize - Called AFTER image already loaded in memory
3. Explicit memory cleanup - Too late, memory already consumed
4. Running outside Docker - Same fundamental issue
5. PIL draft mode - Not supported by datasets library's image loader

**Current status:** 
- **Suitable for pipeline integration** where data is pre-processed
- **Successfully downloads 417/418 samples** with 16GB RAM in Docker
- **Production-ready for training pipelines** when using pre-processed data

**Recommendation:** 
Use `../artefact-repo-analysis/` (git-lfs + parquet processing) for initial full dataset obtention. This approach gives you:
- Full control over image decoding
- Can check size BEFORE loading
- Memory-efficient processing
- Handles extremely large images

**Recommendation:** 
Use this approach for downloading original ARTeFACT dataset (417 samples).

**For training pipeline integration:**

See `../artefact_augmented/` (1458 augmented samples) - this is what all POC training scripts use.

## Workflow

1. **Download original dataset** (this directory):
   ```bash
   cd common-data/artefact-data-obtention
   make download-full  # → ../artefact/ (417 samples)
   ```

2. **Generate augmentations** (see `../README.md` for augmentation script):
   ```bash
   cd ..
   python generate_augmentations.py  # → artefact_augmented/ (1458 samples)
   ```

3. **Train models** (POC-5.5, 5.8, 5.9):
   ```bash
   cd ../artefact-poc59-multiarch-benchmark
   # All configs point to ../common-data/artefact_augmented/
   ```

## Dataset Structure

Expected output:
```
../artefact/
├── images/              # Original images (resized to max 512px)
├── annotations/         # Grayscale damage masks
├── annotations_rgb/     # Colored damage visualizations
├── visualizations/      # Sample 4-panel visualizations
├── metadata.csv         # Sample metadata
└── statistics.json      # Dataset statistics
```

**Note**: Successfully downloads 417/418 samples (one 133M pixel image skipped).

## Related

- **[`../artefact_augmented/`](../artefact_augmented/)** - Augmented dataset (1458 samples) used by all POC training scripts
- **[`../../artefact-poc55-multiclass/`](../../artefact-poc55-multiclass/)** - Hierarchical MTL training
- **[`../../artefact-poc58-standard/`](../../artefact-poc58-standard/)** - Standard segmentation with optimizations
- **[`../../artefact-poc59-multiarch-benchmark/`](../../artefact-poc59-multiarch-benchmark/)** - Production multi-architecture benchmark
