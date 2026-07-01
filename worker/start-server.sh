#!/usr/bin/env bash
# Self-hosted (non-serverless) entrypoint for the LTX-2.3 universal server.
# Pairs with Dockerfile.server. Same engine as the vast image — weights live on
# Cloudflare R2 (s3.unne.ai, public bucket) and are fetched into /opt/models/ on
# first boot, then cached on the host's persistent disk (mount a volume at
# /opt/models to keep them across container restarts).
#
# Differs from start-vast.sh only in the final exec: instead of the vast
# PyWorker it launches server.py — a persistent FastAPI worker with an in-memory
# queue, sync/async, webhooks, Prometheus metrics, and a QUEUE_MAX capacity gate.
#
# Intended for a box you own (e.g. a bare-metal RTX 5090 VDS). Default QUEUE_MAX=2
# (one render in flight + one queued); a 3rd concurrent submit gets 429.
set -e

echo "ltx-server: GPU check..."
python3 -c "import torch; torch.cuda.init(); print('GPU OK:', torch.cuda.get_device_name(0))" || {
    echo "GPU not available, failing fast"; exit 1; }

# --- Fetch weights from R2 if missing ---
R2_BASE="${R2_PUBLIC_URL:-https://s3.unne.ai}"

# spec: dst_relpath|filename|expected_size
WEIGHTS=(
    "ltx23/ltx-2.3-22b-dev-fp8.safetensors|ltx-2.3-22b-dev-fp8.safetensors|29145431166"
    "ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|995743560"
    "ltx23/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors|ltx-2.3-22b-distilled-lora-384-1.1.safetensors|7605507256"
    "gemma-fp8/gemma_3_12B_it_fp8_e4m3fn.safetensors|gemma_3_12B_it_fp8_e4m3fn.safetensors|13210008986"
)

for spec in "${WEIGHTS[@]}"; do
    rel="${spec%%|*}"
    rest="${spec#*|}"
    fname="${rest%%|*}"
    expected_size="${rest##*|}"
    dst="/opt/models/$rel"

    if [ -f "$dst" ]; then
        actual=$(stat -c %s "$dst" 2>/dev/null || stat -f %z "$dst" 2>/dev/null || echo 0)
        if [ "$actual" = "$expected_size" ]; then
            echo "[weights] $rel: cached ($actual B)"
            continue
        fi
        echo "[weights] $rel: size mismatch ($actual vs $expected_size), re-downloading"
        rm -f "$dst"
    fi

    mkdir -p "$(dirname "$dst")"
    echo "[weights] $rel: downloading from $R2_BASE/$fname"
    curl -fL --retry 10 --retry-delay 5 --connect-timeout 30 \
         --continue-at - \
         --speed-time 60 --speed-limit 1000000 \
         -o "$dst" "$R2_BASE/$fname"

    actual=$(stat -c %s "$dst" 2>/dev/null || stat -f %z "$dst" 2>/dev/null)
    if [ "$actual" != "$expected_size" ]; then
        echo "[weights] $rel: FAILED size check ($actual vs $expected_size)"
        exit 1
    fi
    echo "[weights] $rel: OK ($actual B)"
done

# TalkVid ID-LoRA for the custom-audio lip-sync path (audio_url). Required for
# any request with audio_url — fetched by default. Tries R2 first (free egress,
# once mirrored there) then falls back to the public HuggingFace weight. Set
# WITH_TALKVID=0 to skip if you never use the audio path.
if [ "${WITH_TALKVID:-1}" = "1" ]; then
    tv_dst="/opt/models/ltx23/loras/ltx-2.3-id-lora-talkvid-3k.safetensors"
    tv_exp="1157884304"
    tv_hf="https://huggingface.co/Comfy-Org/ltx-2.3/resolve/main/split_files/loras/ltx-2.3-id-lora-talkvid-3k.safetensors"
    if [ ! -f "$tv_dst" ] || [ "$(stat -c %s "$tv_dst" 2>/dev/null || echo 0)" != "$tv_exp" ]; then
        mkdir -p "$(dirname "$tv_dst")"
        echo "[weights] talkvid: fetching (R2 → HF fallback)"
        curl -fL --retry 6 --retry-delay 5 --connect-timeout 30 --continue-at - \
             --speed-time 60 --speed-limit 1000000 \
             -o "$tv_dst" "$R2_BASE/ltx-2.3-id-lora-talkvid-3k.safetensors" \
          || curl -fL --retry 10 --retry-delay 5 --connect-timeout 30 --continue-at - \
                  --speed-time 60 --speed-limit 1000000 \
                  -o "$tv_dst" "$tv_hf" \
          || { echo "[weights] talkvid: download failed — audio lip-sync unavailable"; rm -f "$tv_dst"; }
    fi
    if [ -f "$tv_dst" ]; then
        mkdir -p /opt/models/loras/ltxv/ltx2
        ln -sf "$tv_dst" /opt/models/loras/ltxv/ltx2/ltx-2.3-id-lora-talkvid-3k.safetensors
        echo "[weights] talkvid: ready"
    fi
fi

# Multiple-Subject-Reference IC-LoRA for reference_image_urls. R2 only (no
# public HF fallback needed — R2 is the source of truth). Set WITH_MSR=0 to
# skip if you never use the reference-image path.
if [ "${WITH_MSR:-1}" = "1" ]; then
    msr_dst="/opt/models/ltx23/loras/ltx-2.3-licon-msr-v1.safetensors"
    msr_exp="654443424"
    if [ ! -f "$msr_dst" ] || [ "$(stat -c %s "$msr_dst" 2>/dev/null || echo 0)" != "$msr_exp" ]; then
        mkdir -p "$(dirname "$msr_dst")"
        echo "[weights] msr: fetching from R2"
        curl -fL --retry 10 --retry-delay 5 --connect-timeout 30 --continue-at - \
             --speed-time 60 --speed-limit 1000000 \
             -o "$msr_dst" "$R2_BASE/ltx-2.3-licon-msr-v1.safetensors" \
          || { echo "[weights] msr: download failed — reference_image_urls unavailable"; rm -f "$msr_dst"; }
    fi
    if [ -f "$msr_dst" ]; then
        mkdir -p /opt/models/loras/ltxv/ltx2
        ln -sf "$msr_dst" /opt/models/loras/ltxv/ltx2/ltx-2.3-licon-msr-v1.safetensors
        echo "[weights] msr: ready"
    fi
fi

# Rebuild the ComfyUI symlink farm under /opt/models. The Dockerfile bakes this
# farm into the image, but a `-v <host>:/opt/models` volume mount MASKS the
# baked dirs — the host volume only holds ltx23/ + gemma-fp8/ (the downloaded
# files), so checkpoints/, latent_upscale_models/, loras/ltxv/ltx2/ and
# text_encoders/ would be empty and ComfyUI's loaders see no models
# ("ckpt_name ... not in []"). Recreating the symlinks here (after the mount is
# in place + weights are present) fixes it. Idempotent; persists on the volume.
mkdir -p /opt/models/checkpoints /opt/models/text_encoders \
         /opt/models/latent_upscale_models /opt/models/loras/ltxv/ltx2
ln -sf /opt/models/ltx23/ltx-2.3-22b-dev-fp8.safetensors \
       /opt/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors
ln -sf /opt/models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
       /opt/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
ln -sf /opt/models/ltx23/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
       /opt/models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors
ln -sf /opt/models/gemma-fp8/gemma_3_12B_it_fp8_e4m3fn.safetensors \
       /opt/models/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors

# Belt-and-braces gemma symlink (mirrors RunPod start.sh) in case
# extra_model_paths quirks don't catch the text_encoders/ path.
for f in gemma_3_12B_it_fp8_e4m3fn gemma_3_12B_it_fp8_scaled gemma_3_12B_it; do
    src="/opt/models/gemma-fp8/${f}.safetensors"
    dst="/comfyui/models/text_encoders/${f}.safetensors"
    if [ -e "$src" ] && [ ! -e "$dst" ]; then
        mkdir -p /comfyui/models/text_encoders
        ln -sf "$src" "$dst"
    fi
done

: "${PORT:=8000}"
: "${QUEUE_MAX:=2}"
export PORT QUEUE_MAX

: "${COMFY_LOG_LEVEL:=INFO}"
: "${COMFY_LOG_FILE:=/tmp/comfyui.log}"
export COMFY_LOG_FILE

python3 -u /comfyui/main.py --disable-auto-launch --disable-metadata \
    --verbose "${COMFY_LOG_LEVEL}" --log-stdout \
    > "${COMFY_LOG_FILE}" 2>&1 &

exec python3 -u /server.py
