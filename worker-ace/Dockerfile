# syntax=docker/dockerfile:1.7
# ACE-Step 1.5 XL-SFT worker — RunPod Serverless, RTX 4090 (24 GB) target.
# ComfyUI + jeankassio/ComfyUI-AceStep_SFT + XL-SFT 4B DiT.
# Weights live on a RunPod Network Volume mounted at /runpod-volume/, NOT baked in.

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_ENABLE_HF_TRANSFER=1

# No apt step: runpod/pytorch base ships with git + ca-certificates already,
# and ACE-Step doesn't need ffmpeg/opencv (audio only). Adding apt-get on top
# tries to upgrade existing packages which blows the overlay disk on Docker
# Desktop. We sanity-check git is present below; if a future base image drops
# it, the build will fail with a clear `git: command not found` rather than
# a confusing apt cache error.
RUN command -v git >/dev/null 2>&1 || { echo "FATAL: git missing in base image" >&2; exit 1; }

# ComfyUI — pin to a validated revision via ARG when bumping
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui && git checkout ${COMFYUI_REF}
WORKDIR /comfyui
RUN pip install -r requirements.txt

# jeankassio ACE-Step SFT custom node (provides AceStepSFTGenerate / TextEncode /
# ModelLoader / SaveAudio nodes used by the hi-quality 4-pass workflow)
ARG ACE_NODES_REF=main
RUN git clone https://github.com/jeankassio/ComfyUI-AceStep_SFT.git \
        /comfyui/custom_nodes/ComfyUI-AceStep_SFT \
    && cd /comfyui/custom_nodes/ComfyUI-AceStep_SFT && git checkout ${ACE_NODES_REF} \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Audio + S3 deps for the handler + workflow_builder
RUN pip install 'huggingface_hub>=0.26' 'hf_transfer' 'boto3' 'soundfile' 'pydub'

# Worker payload (handler, builder, template, system files)
COPY src/ /worker/src/
COPY start.sh /start.sh
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
RUN chmod +x /start.sh

EXPOSE 22 8188
CMD ["/start.sh"]
