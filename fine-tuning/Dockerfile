# Multi-target Dockerfile for the MedGemma entity resolution pipeline.
# Build a specific stage: docker build --target <stage> -t medgemma-<stage> .
# Stages: prepare, train, export, infer, interactive

FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.4 support (needs custom index URL, so separate from requirements)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ---------- Stage: prepare (strip vision tower, one-time CPU job) ----------
FROM base AS prepare

COPY requirements-prepare.txt .
RUN pip install --no-cache-dir -r requirements-prepare.txt

COPY prepare_base_model.py .

ENTRYPOINT ["python", "prepare_base_model.py"]

# ---------- Stage: train (QLoRA fine-tuning, GPU) ----------
FROM base AS train

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY train_classifier_on_gpu.py .

ENTRYPOINT ["python", "train_classifier_on_gpu.py"]

# ---------- Stage: export (merge LoRA adapter, GPU) ----------
FROM base AS export

COPY requirements-export.txt .
RUN pip install --no-cache-dir -r requirements-export.txt

COPY export_text_only_model.py .

ENTRYPOINT ["python", "export_text_only_model.py"]

# ---------- Stage: infer (batch inference, GPU) ----------
FROM base AS infer

COPY requirements-infer.txt .
RUN pip install --no-cache-dir -r requirements-infer.txt

COPY inference_classifier.py .

ENTRYPOINT ["python", "inference_classifier.py"]

# ---------- Stage: interactive (RunPod SSH, all scripts) ----------
FROM base AS interactive

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prepare_base_model.py \
     train_classifier_on_gpu.py \
     export_text_only_model.py \
     inference_classifier.py \
     ./

# No ENTRYPOINT — RunPod manages the container lifecycle and SSH.
CMD ["sleep", "infinity"]
