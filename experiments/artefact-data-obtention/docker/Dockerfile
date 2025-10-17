# ARTeFACT Data Obtention - Docker Image
# Educational POC for downloading ARTeFACT dataset via HuggingFace datasets library

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY docker/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy main script
COPY scripts/download_artefact.py ./scripts/

# Create output directory
RUN mkdir -p data

# Default command: show help
CMD ["python3", "scripts/download_artefact.py", "--help"]
