# syntax=docker/dockerfile:1.7
# LTX-2.3 worker for RunPod Serverless on RTX 5090 (Blackwell / sm_120).
# Requires CUDA 12.8+ and PyTorch with cu128 wheels for Blackwell support.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_DIR=/models/ltx

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Blackwell (sm_120) needs cu128 wheels. Pin a stable version known to ship sm_120 kernels.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0 torchvision==0.22.0

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt hf_transfer

# ---- Weight bake stage ---------------------------------------------------
# HF token passed via buildx secret — NOT stored in image:
#   docker buildx build --secret id=hf_token,env=HF_TOKEN ...
COPY download_weights.py /app/download_weights.py
ARG HF_REPO_ID=Lightricks/LTX-2.3
ENV HF_REPO_ID=${HF_REPO_ID}
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN_FILE=/run/secrets/hf_token python /app/download_weights.py

# ---- App code ------------------------------------------------------------
COPY pipeline.py handler.py s3_upload.py /app/

CMD ["python", "-u", "/app/handler.py"]
