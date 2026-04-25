#!/usr/bin/env bash
set -e

# Optional SSH for in-pod debugging via RunPod "pod mode"
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
    # Start sshd if available; the base runpod/pytorch image ships openssh-server
    # but we override CMD, so we must start it ourselves.
    if command -v sshd >/dev/null 2>&1; then
        if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
            ssh-keygen -A >/dev/null 2>&1 || true
        fi
        mkdir -p /var/run/sshd
        /usr/sbin/sshd -E /var/log/sshd.log
        echo "sshd started on :22"
    fi
fi

echo "worker-comfyui: GPU check..."
python3 -c "import torch; torch.cuda.init(); print('GPU OK:', torch.cuda.get_device_name(0))" || {
    echo "GPU not available, failing fast"; exit 1; }

# Ensure network-volume text encoders are symlinked into comfy's models/text_encoders/
# so folder_paths picks them up. (extra_model_paths.yaml config does not always win
# when it comes to directories outside comfy's root.)
for f in gemma_3_12B_it gemma_3_12B_it_fp8_e4m3fn gemma_3_12B_it_fp8_scaled; do
    src="/runpod-volume/models/text_encoders/${f}.safetensors"
    dst="/comfyui/models/text_encoders/${f}.safetensors"
    if [ -e "$src" ] && [ ! -e "$dst" ]; then
        ln -sf "$src" "$dst"
    fi
done

# Launch ComfyUI in background
: "${COMFY_LOG_LEVEL:=INFO}"
COMFY_PID_FILE="/tmp/comfyui.pid"

if [ "$SERVE_API_LOCALLY" = "true" ]; then
    python3 -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen \
        --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"
    python3 -u /warmup.py || true
    python3 -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python3 -u /comfyui/main.py --disable-auto-launch --disable-metadata \
        --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"
    python3 -u /warmup.py || true
    python3 -u /handler.py
fi
