# syntax=docker/dockerfile:1.7
# LTX-2.3 ComfyUI worker for RunPod Serverless on RTX 5090 (Blackwell / sm_120).
# Stack matches the validated EU-RO-1 pod where Full HD 10s worked.
# Weights live on RunPod Network Volume (/runpod-volume/models/), NOT baked in.

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI — validated working revision can be pinned via ARG
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui && git checkout ${COMFYUI_REF}
WORKDIR /comfyui
RUN pip install -r requirements.txt

# Lightricks LTX-2 custom nodes (provides LTXAV/LTX* nodes used by workflow)
ARG LTX_NODES_REF=master
RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /comfyui/custom_nodes/ComfyUI-LTXVideo \
    && cd /comfyui/custom_nodes/ComfyUI-LTXVideo && git checkout ${LTX_NODES_REF} \
    && pip install -r requirements.txt

# Pin transformers to a 4.x known-compatible release.
# Reason: LTX-2 + ComfyUI-LTXVideo access SiglipVisionModel.vision_model and
# Gemma3ForConditionalGeneration.multi_modal_projector at the top level, which
# transformers 5.x removed. Validated on 4.57.6. huggingface_hub must stay <1.0.
RUN pip install --force-reinstall 'transformers==4.57.6' 'huggingface_hub<1.0' 'boto3'

# Patch comfy's LTXAV text-encoder memory estimator to tolerate empty token lists.
# LTXVGemmaEnhancePrompt calls clip.load_model() before any tokenize(), which
# makes memory_estimation_function receive {"gemma3_12b": []} → min([]) ValueError.
RUN python3 - <<'PY'
p = "/comfyui/comfy/text_encoders/lt.py"
s = open(p).read()
old = '        token_weight_pairs = token_weight_pairs.get("gemma3_12b", [])\n        m = min([sum(1 for _ in itertools.takewhile(lambda x: x[0] == 0, sub)) for sub in token_weight_pairs])'
new = '        token_weight_pairs = token_weight_pairs.get("gemma3_12b", [])\n        if not token_weight_pairs:\n            return 642 * constant * 1024 * 1024\n        m = min([sum(1 for _ in itertools.takewhile(lambda x: x[0] == 0, sub)) for sub in token_weight_pairs])'
assert old in s, "lt.py patch target not found — upstream may have changed"
open(p, "w").write(s.replace(old, new))
print("lt.py patched")
PY

# Handler dependencies
RUN pip install runpod requests websocket-client pillow

# Map /runpod-volume/models/* into ComfyUI's expected dir layout
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml

# System prompts for Gemma enhancement (I2V / T2V)
COPY system_prompts /system_prompts

# Handler + startup
COPY src/handler.py /handler.py
COPY src/workflow_builder.py /workflow_builder.py
COPY src/workflow_template_api.json /workflow_template_api.json
COPY src/s3_upload.py /s3_upload.py
COPY src/warmup.py /warmup.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
