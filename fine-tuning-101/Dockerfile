FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Python 3.10 + pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch (CUDA 11.8) + inference dependencies
COPY requirements-inference.txt .
RUN pip install --no-cache-dir torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements-inference.txt

WORKDIR /app
COPY inference_classifier.py .

# HF token passed at runtime via env var
# docker run --gpus all -e HF_TOKEN=hf_... medgemma-inference --dataset
ENTRYPOINT ["python3", "inference_classifier.py"]
