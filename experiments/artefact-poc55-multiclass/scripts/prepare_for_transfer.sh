#!/usr/bin/env bash
set -euo pipefail

# Prepare a minimal archive for transferring POC-5.5 to another machine.
# Usage: ./scripts/prepare_for_transfer.sh

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

echo "Cleaning logs, checkpoints and caches..."
rm -rf logs/* || true
rm -rf .cache/* || true

# Do NOT include the dataset (data/artefact) to keep the archive small.
# The receiving machine can run `make download` to fetch the dataset.

ARCHIVE_NAME="poc55_transfer.tar.gz"
echo "Creating archive: $ARCHIVE_NAME"

tar -czf "$ARCHIVE_NAME" \
    Makefile \
    README.md \
    .gitignore \
    configs/ \
    docker/ \
    scripts/ \
    --exclude='scripts/*.log' \
    --exclude='scripts/__pycache__' \
    --exclude='configs/**/checkpoints' \
    --exclude='**/logs' \
    --exclude='data/artefact' \
    --exclude='.cache'

echo "Archive created at: $ROOT_DIR/$ARCHIVE_NAME"

echo "Transfer checklist:"
echo "  - On target machine: unzip archive and run 'make build' then 'make up' and 'make download'"
echo "  - After container is up: run 'make test-epoch' or 'make train-convnext' inside this directory"

exit 0
