# ARTeFACT Repository Analysis - Docker Image
# Production-ready POC for downloading and processing ARTeFACT dataset via git-lfs

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Configure git-lfs
RUN git lfs install

# Create working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY docker/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all scripts
COPY scripts/*.py ./scripts/

# Create output directories
RUN mkdir -p artefact_repo data

# Set up git configuration for large repo operations
RUN git config --global http.postBuffer 524288000 && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 999999

# Default command: show help
CMD ["python3", "scripts/process_parquet.py", "--help"]
