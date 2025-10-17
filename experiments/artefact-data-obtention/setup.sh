#!/bin/bash
# Setup script for ARTeFACT data obtention (venv approach)

set -e

echo "=================================="
echo "ARTeFACT Data Obtention Setup"
echo "=================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo ""
    echo "⚠️  Virtual environment already exists"
fi

# Activate and install dependencies
echo ""
echo "Installing dependencies..."
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "To use the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To download samples:"
echo "  python download_artefact.py --max-samples 10"
echo ""
