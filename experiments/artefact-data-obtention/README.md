# ARTeFACT Data Obtention - HuggingFace Streaming Approach

**Code example** demonstrating how to download ARTeFACT using HuggingFace `datasets` library with streaming mode.

⚠️ **Note**: This approach has memory limitations with very large images.

## � Project Structure

```
artefact-data-obtention/
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── docker/                # Docker configuration
│   ├── Dockerfile         # Image definition
│   ├── docker-compose.yml # Compose config
│   └── requirements.txt   # Python dependencies
├── scripts/               # Python scripts
│   └── download_artefact.py  # Main download script
├── Makefile               # Convenience commands (make help)
├── setup.sh               # venv setup script
├── venv/                  # Python virtual environment (optional)
└── data/                  # Downloaded datasets (generated)
```

## Purpose

Provides clean, documented code showing how to integrate ARTeFACT with HuggingFace ecosystem for training pipelines.

## Approach: HuggingFace Datasets Library

✅ **Streaming mode** - Downloads samples incrementally without loading entire dataset  
✅ **Automatic image resizing** - Reduces memory footprint (max 512-1024px)  
✅ **Progress tracking** - Real-time progress with tqdm  
✅ **Rich metadata** - Exports metadata, statistics, and visualizations  
✅ **Docker ready** - Fully containerized, no venv conflicts  
✅ **16GB RAM** - Successfully processes 417/418 samples (99.8%)

## Quick Start

### Option 1: Docker (Recommended - Isolated)

```bash
# Build image (first time only)
make build

# Download 10 samples (quick test)
make download-small

# Download 50 samples
make download-medium

# Download full dataset (417/418 samples, requires 16GB RAM)
make download-full

# Or use docker-compose directly:
cd docker
docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py \
  --output data/test \
  --max-samples 10
```

### Option 2: Python venv

#### 1. Setup Environment (First Time Only)

```bash
./setup.sh
```

This creates a venv and installs all dependencies.

#### 2. Activate Environment

```bash
source venv/bin/activate
```

#### 3. Download Samples

```bash
# Download 10 samples (quick test)
python scripts/download_artefact.py --max-samples 10 --output ./data/test

# Download 50 samples (recommended for POC)
python scripts/download_artefact.py --max-samples 50 --output ./data/poc

# Download 100+ samples (for real training)
python scripts/download_artefact.py --max-samples 100 --output ./data/train

# Download ALL samples (~420 images, requires 16GB RAM)
python scripts/download_artefact.py --all --output ./data/full
```
python download_artefact.py --max-samples 100 --output ./data/artefact_train

# Download ALL samples (~420 images, requires 16GB RAM)
python download_artefact.py --all --output ./data/artefact_full
```

## Files

```
artefact-data-obtention/
├── download_artefact.py      # Main script with streaming mode
├── requirements.txt          # Python dependencies
├── setup.sh                 # Setup script (creates venv)
├── .gitignore              # Git ignore rules
└── data/
    └── artefact_venv_test/  # Sample output (5 successful downloads)
```

### download_artefact.py

**Features:**
- ✅ Streaming download (incremental processing)
- ✅ Automatic image resizing (max 512px)
- ✅ Progress tracking with tqdm
- ✅ Metadata and statistics export
- ✅ Sample visualizations (4-panel)

**Limitations:**
- ⚠️ Crashes on extremely large images (>50M pixels)
- ⚠️ Successfully processes ~5-9 samples before hitting memory limits
- ⚠️ Not suitable for full dataset download from scratch

## Output Structure

```
data/artefact_*/
├── images/              # Resized original images (PNG)
├── annotations/         # Grayscale damage masks (0-15 classes, 255=background)
├── annotations_rgb/     # Colored damage visualizations
├── visualizations/      # 4-panel sample visualizations (first 10)
├── metadata.csv         # Sample IDs, paths, descriptions, sizes
└── statistics.json      # Dataset statistics (materials, contents, class distribution)
```

## Example: Sample Dataset

The `data/artefact_venv_test/` directory contains a successful download of 5 samples demonstrating the output structure.

## Known Limitations

### Memory Constraints with Large Images

**Problem:** The HuggingFace `datasets` library decodes images fully in memory when accessing `sample['image']`, BEFORE any resizing can occur. The ARTeFACT dataset contains extremely large images (up to 7680×4877 = 37M pixels, some even >100M pixels) that can exceed available system memory.

**What happens:**
- ✅ Small-medium images (< 20M pixels): Process successfully
- ⚠️ Large images (20-50M pixels): May work depending on available RAM
- ❌ Huge images (> 50M pixels): Crash WSL/system (OOM)

**Solutions attempted:**
1. ✅ Streaming mode (`streaming=True`) - Still decodes individual images fully
2. ✅ PIL thumbnail/resize - Called AFTER image already loaded in memory
3. ✅ Explicit memory cleanup - Too late, memory already consumed
4. ✅ Running outside Docker - Same fundamental issue
5. ❌ PIL draft mode - Not supported by datasets library's image loader

**Current status:** 
- **Suitable for pipeline integration** where data is pre-processed
- **Successfully downloads 5-9 samples** before hitting problematic large images
- **NOT suitable for full dataset download** from scratch

**Recommendation:** 
Use `../artefact-repo-analysis/` (git-lfs + parquet processing) for full dataset obtention. This approach gives you:
- ✅ Full control over image decoding
- ✅ Can check size BEFORE loading
- ✅ Memory-efficient processing
- ✅ Successfully processed 5 samples without issues

**Use this approach ONLY IF:**
- Data is already pre-processed and available
- You need HuggingFace ecosystem integration
- Working with subset of smaller images
- For pipeline training code examples

## ⚠️ For Production Use

**This approach is for learning/examples only.** For actual dataset obtention:

👉 **Use [`../artefact-repo-analysis/`](../artefact-repo-analysis/)** 
- ✅ Handles all images including 133M pixel files
- ✅ Memory efficient
- ✅ Production-ready
- ✅ Successfully processed 15/15 samples from first parquet file

## Dataset Structure

Expected output:
```
data/artefact_custom/
├── images/              # Original images (resized)
├── annotations/         # Grayscale damage masks
├── annotations_rgb/     # Colored damage visualizations
├── visualizations/      # Sample 4-panel visualizations
├── metadata.csv         # Sample metadata
└── statistics.json      # Dataset statistics
```

## Related Experiments

- **[`../artefact-repo-analysis/`](../artefact-repo-analysis/)** - Production data obtention (git-lfs + parquet)
- **[`../poc-art-damage/`](../poc-art-damage/)** - Main training pipeline with CNN/ViT models
- **[`../ARTEFACT_EXPERIMENTS.md`](../ARTEFACT_EXPERIMENTS.md)** - Overview of both approaches
