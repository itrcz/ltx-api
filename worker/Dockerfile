# syntax=docker/dockerfile:1.7
# Thin LTX-2.3 worker for RunPod Serverless on RTX 5090 (Blackwell / sm_120).
# Weights are NOT baked in — they live on a RunPod Network Volume mounted at
# /runpod-volume on Serverless (hard-coded by RunPod).

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    MODELS_DIR=/runpod-volume/models \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates git ffmpeg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# torch with cu128 (Blackwell / sm_120). torchaudio must also come from cu128
# index, otherwise pip falls back to cu130 build that needs libcudart.so.13.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

# Clone LTX-2 and install its packages (editable) into system site-packages.
# Pin to a commit for reproducibility — bump ARG on upgrade.
ARG LTX2_REF=main
RUN git clone --depth 1 --branch ${LTX2_REF} \
        https://github.com/Lightricks/LTX-2.git /opt/LTX-2
WORKDIR /opt/LTX-2
RUN python -m pip install -e packages/ltx-core -e packages/ltx-pipelines

# App deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

WORKDIR /app
COPY handler.py s3_upload.py /app/

# RunPod Serverless entrypoint
CMD ["python", "-u", "/app/handler.py"]
