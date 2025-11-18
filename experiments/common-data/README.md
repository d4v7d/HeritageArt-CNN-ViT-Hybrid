# Common Data Directory

Shared datasets for all HeritageArt POC experiments.

## Structure

```
common-data/
├── README.md                  # This file
├── artefact-data-obtention/   # Download + augmentation scripts
│   ├── docker/
│   ├── scripts/
│   │   ├── download_artefact.py      # Download from HuggingFace
│   │   └── generate_augmentations.py # Generate augmented dataset
│   └── Makefile
├── artefact/                  # Original ARTeFACT dataset (417 samples)
│   ├── images/                # NOT IN GIT - Download locally
│   ├── annotations/
│   └── metadata.csv
└── artefact_augmented/        # Augmented dataset (1458 samples) ✅ IN GIT
    ├── images/                # 1458 augmented images
    ├── annotations/           # 1458 augmented masks
    └── class_weights_balanced.json
```

## Quick Start

### For Training (Already Done ✅)

You already have everything needed:
```bash
cd ../artefact-poc59-multiarch-benchmark
python src/train.py --config configs/segformer_b3.yaml
```

All configs point to `../common-data/artefact_augmented/` - no download needed!

### For Reproducing Dataset (Optional)

If you want to regenerate `artefact_augmented/` from scratch:

```bash
cd artefact-data-obtention

# 1. Download original 417 samples
make download-full  # → ../artefact/

# 2. Generate 1458 augmented samples
make generate-augmentations  # → ../artefact_augmented/
```

## Datasets

### `artefact/` - Original (NOT IN GIT)

**Source**: HuggingFace `danielaivanova/damaged-media`  
**Size**: ~2-3 GB  
**Samples**: 417 images + masks  
**Purpose**: Source for augmentation

### `artefact_augmented/` - Augmented (IN GIT ✅)

**Size**: 9.7 GB  
**Samples**: 1458 images + masks  
**Purpose**: Training dataset for all POCs  
**Generation**: 417 original + 1041 augmented (3x per image)

**Used by**:
- POC-5.5, POC-5.8, POC-5.9 (all configs)

## Reproducibility

Complete workflow documented in `artefact-data-obtention/`:

1. **Download**: `download_artefact.py` - Get 417 samples from HuggingFace
2. **Augment**: `generate_augmentations.py` - Generate 1458 samples

Both scripts are tested and ready to use.

## See Also

- [artefact-data-obtention/README.md](artefact-data-obtention/README.md) - Download/augmentation details
- [../artefact-poc59-multiarch-benchmark/](../artefact-poc59-multiarch-benchmark/) - Production training
