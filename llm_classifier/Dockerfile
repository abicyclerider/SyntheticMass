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

# Install all Python dependencies (flash-attn compiles CUDA kernels, needs nvcc + ninja above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prepare_base_model.py \
     train_classifier.py \
     export_model.py \
     infer_classifier.py \
     ./

CMD ["sleep", "infinity"]
