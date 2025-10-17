FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install only the extras actually required by this PoC; torch/vision/audio already in base image
RUN pip install --no-cache-dir \
    timm==1.0.9 \
    transformers==4.44.2 \
    numpy==1.26.4 \
    omegaconf==2.3.0 \
    pillow==10.4.0 \
    datasets==2.20.0 \
    pandas==2.2.2 \
    matplotlib==3.9.0 \
    tqdm==4.66.4

CMD ["bash", "-lc", "python -m src.run_all --help"]
