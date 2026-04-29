#!/usr/bin/env bash
set -e

# Optional sshd for in-pod debugging when running this image as a pod
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
    if command -v sshd >/dev/null 2>&1; then
        if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
            ssh-keygen -A >/dev/null 2>&1 || true
        fi
        mkdir -p /var/run/sshd
        /usr/sbin/sshd -E /var/log/sshd.log
        echo "sshd started on :22"
    fi
fi

echo "ace-worker: GPU check..."
python3 -c "import torch; torch.cuda.init(); n=torch.cuda.get_device_name(0); m=torch.cuda.get_device_capability(0); print(f'GPU OK: {n} sm_{m[0]}{m[1]}')" || {
    echo "GPU not available, failing fast"; exit 1; }

# jeankassio's custom node looks for models under /comfyui/models/Audio/.
# extra_model_paths.yaml maps it to /runpod-volume/models/Audio/, but as belt-
# and-suspenders we also make sure the symlinks land where the node expects.
mkdir -p /comfyui/models/Audio /runpod-volume/models/Audio
for f in acestep_v1.5_sft_xl.safetensors \
         qwen_0.6b_ace15.safetensors \
         qwen_4b_ace15.safetensors \
         ace_1.5_vae.safetensors; do
    [ -e "/comfyui/models/Audio/$f" ] || \
        ln -sfn "/runpod-volume/models/Audio/$f" "/comfyui/models/Audio/$f" 2>/dev/null || true
done

# Launch ComfyUI in background; handler will wait for /system_stats.
cd /comfyui
python -u main.py --listen 127.0.0.1 --port 8188 --enable-cors-header \
    > /tmp/comfy.log 2>&1 &
COMFY_PID=$!
echo "comfyui pid=$COMFY_PID, log=/tmp/comfy.log"

# Hand off to handler. RunPod's serverless SDK takes over; on /run it calls
# handler.handler(event), polls comfy, uploads result.
exec python -u /worker/src/handler.py
