# MedGemma entity resolution pipeline.
# Build: docker build -t medgemma .
# Run:   docker run --gpus all -e HF_TOKEN=hf_... medgemma python <script>.py [args]

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL org.opencontainers.image.source=https://github.com/abicyclerider/SyntheticMass

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        gcc g++ libc6-dev ninja-build && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.4 support (needs custom index URL, so separate from requirements)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn from source (--no-build-isolation because setup.py imports torch).
# Pre-install all its build-time deps: packaging, psutil, ninja.
RUN pip install --no-cache-dir packaging psutil ninja && \
    pip install --no-cache-dir --no-build-isolation "flash-attn>=2.5.0"

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY _runpod.py \
     prepare_base_model.py \
     train_classifier.py \
     export_model.py \
     infer_classifier.py \
     ./

CMD ["sleep", "infinity"]
