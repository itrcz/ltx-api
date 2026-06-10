#!/usr/bin/env bash
# Yotta Labs Serverless entrypoint. Pairs with Dockerfile.yotta.
#
# Boot sequence:
#   1. GPU sanity check
#   2. Download weights from R2 (s3.unne.ai) into /opt/models/ if not cached
#   3. Launch ComfyUI on 127.0.0.1:8188 in background
#   4. exec uvicorn yotta_worker:app on :$WORKER_PORT (default 8000)
#
# Yotta expects an HTTP server on $WORKER_PORT (declared in `expose` on the
# endpoint). When a task is submitted, Yotta POSTs to the worker's
# $processUri (default /run) with the taskData JSON.
set -e

echo "gr-tv-yotta: GPU check..."
python3 -c "import torch; torch.cuda.init(); print('GPU OK:', torch.cuda.get_device_name(0))" || {
    echo "GPU not available, failing fast"; exit 1; }

# --- R2 weight download ---
R2_BASE="${R2_PUBLIC_URL:-https://s3.unne.ai}"

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
    echo "[weights] $rel: $R2_BASE/$fname"
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

# TalkVid ID-LoRA for the custom-audio lip-sync path (audio_url). Off by default;
# requires the weight mirrored to R2 first, then WITH_TALKVID=1 in the env.
if [ "${WITH_TALKVID:-0}" = "1" ]; then
    tv_dst="/opt/models/ltx23/loras/ltx-2.3-id-lora-talkvid-3k.safetensors"
    tv_exp="1157884304"
    if [ ! -f "$tv_dst" ] || [ "$(stat -c %s "$tv_dst" 2>/dev/null || echo 0)" != "$tv_exp" ]; then
        mkdir -p "$(dirname "$tv_dst")"
        echo "[weights] talkvid: downloading from $R2_BASE/ltx-2.3-id-lora-talkvid-3k.safetensors"
        curl -fL --retry 10 --retry-delay 5 --connect-timeout 30 --continue-at - \
             --speed-time 60 --speed-limit 1000000 \
             -o "$tv_dst" "$R2_BASE/ltx-2.3-id-lora-talkvid-3k.safetensors" \
          || { echo "[weights] talkvid: download failed — lip-sync unavailable"; rm -f "$tv_dst"; }
    fi
    if [ -f "$tv_dst" ]; then
        mkdir -p /opt/models/loras/ltxv/ltx2
        ln -sf "$tv_dst" /opt/models/loras/ltxv/ltx2/ltx-2.3-id-lora-talkvid-3k.safetensors
        echo "[weights] talkvid: ready"
    fi
fi

# Belt-and-braces gemma symlink (same quirk as RunPod).
for f in gemma_3_12B_it_fp8_e4m3fn gemma_3_12B_it_fp8_scaled gemma_3_12B_it; do
    src="/opt/models/gemma-fp8/${f}.safetensors"
    dst="/comfyui/models/text_encoders/${f}.safetensors"
    if [ -e "$src" ] && [ ! -e "$dst" ]; then
        mkdir -p /comfyui/models/text_encoders
        ln -sf "$src" "$dst"
    fi
done

: "${WORKER_PORT:=8000}"
export WORKER_PORT

: "${COMFY_LOG_LEVEL:=INFO}"
: "${COMFY_LOG_FILE:=/tmp/comfyui.log}"
export COMFY_LOG_FILE

# ComfyUI in bg
python3 -u /comfyui/main.py --disable-auto-launch --disable-metadata \
    --verbose "${COMFY_LOG_LEVEL}" --log-stdout \
    > "${COMFY_LOG_FILE}" 2>&1 &

# Wait for ComfyUI's HTTP server to come up before accepting requests —
# Yotta may start sending tasks the moment FastAPI is bindable.
for i in $(seq 1 120); do
    if curl -fsS -m 1 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
        echo "[start-yotta] ComfyUI ready after ${i}s"
        break
    fi
    sleep 1
done

cd /
exec python3 -u /yotta_worker.py
