# ARTeFACT Data Obtention - HuggingFace Streaming Approach

**Code example** demonstrating how to download ARTeFACT using HuggingFace `datasets` library with streaming mode.

**Note**: This approach has memory limitations with very large images.

## Project Structure

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
└── data/                  # Downloaded datasets (generated)
    └── demo/              # Demo dataset (10 samples)
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

### Custom Commands

```bash
# Download specific number of samples
docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py \
  --output data/custom \
  --max-samples 100

# Download all samples
docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py \
  --output data/full \
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

```
data/artefact_*/
├── images/              # Resized original images (PNG)
├── annotations/         # Grayscale damage masks (0-15 classes, 255=background)
├── annotations_rgb/     # Colored damage visualizations
├── visualizations/      # 4-panel sample visualizations (first 10)
├── metadata.csv         # Sample IDs, paths, descriptions, sizes
└── statistics.json      # Dataset statistics (materials, contents, class distribution)
```

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

**Use this Docker approach when:**
- Integrating with HuggingFace ecosystem
- Working with pre-processed datasets
- Building training pipeline examples
- Need automatic streaming capabilities

## Recommended Workflow

**For complete dataset obtention:**

**Use [`../artefact-repo-analysis/`](../artefact-repo-analysis/)** 
- Handles all images including 133M pixel files
- Memory efficient parquet processing
- Production-ready with Docker
- Git-LFS repository management

**For training pipeline integration:**

**Use this approach (artefact-data-obtention)** 
- HuggingFace datasets integration
- Streaming capabilities
- 417/418 samples success rate
- Docker containerized

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
