#!/usr/bin/env bash
# Bootstrap ComfyUI + ComfyUI-LTXVideo on a RunPod pod with the LTX network
# volume mounted at /workspace. Mirrors worker/Dockerfile + worker/start.sh
# but writes everything into /workspace so the install survives pod restarts.
#
# Usage (on the pod):
#   bash /workspace/comfyui-pod-setup.sh
#
# Idempotent: re-running skips already-installed pieces.
set -euo pipefail

MOUNT="${MOUNT:-/workspace}"
COMFY_DIR="${MOUNT}/ComfyUI"
PORT="${PORT:-8888}"

echo "==> 1. Free port $PORT (kill jupyter / anything stock-image started)"
fuser -k -n tcp "$PORT" 2>/dev/null || true
pkill -f jupyter 2>/dev/null || true
sleep 1

echo "==> 2. apt deps"
apt-get update -qq
apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 ca-certificates >/dev/null

echo "==> 3. ComfyUI"
if [ ! -d "$COMFY_DIR" ]; then
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi
cd "$COMFY_DIR"
pip install --quiet -r requirements.txt

echo "==> 4. ComfyUI-LTXVideo custom nodes"
LTX_DIR="$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo"
if [ ! -d "$LTX_DIR" ]; then
    git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git "$LTX_DIR"
fi
cd "$LTX_DIR"
pip install --quiet -r requirements.txt

echo "==> 5. Pin transformers (LTX-2 + custom-node compat)"
pip install --quiet --force-reinstall 'transformers==4.57.6' 'huggingface_hub<1.0'

echo "==> 6. Patch lt.py (empty token list tolerance)"
LT_PY="$COMFY_DIR/comfy/text_encoders/lt.py"
python3 - "$LT_PY" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = '        token_weight_pairs = token_weight_pairs.get("gemma3_12b", [])\n        m = min([sum(1 for _ in itertools.takewhile(lambda x: x[0] == 0, sub)) for sub in token_weight_pairs])'
new = '        token_weight_pairs = token_weight_pairs.get("gemma3_12b", [])\n        if not token_weight_pairs:\n            return 642 * constant * 1024 * 1024\n        m = min([sum(1 for _ in itertools.takewhile(lambda x: x[0] == 0, sub)) for sub in token_weight_pairs])'
if old in s:
    p.write_text(s.replace(old, new))
    print("lt.py patched")
elif new in s:
    print("lt.py already patched")
else:
    print("lt.py patch target not found — upstream may have changed")
PY

echo "==> 7. extra_model_paths.yaml → /workspace/models"
cat > "$COMFY_DIR/extra_model_paths.yaml" <<'YAML'
ltx_worker:
  base_path: /workspace/models
  checkpoints: checkpoints/
  text_encoders: text_encoders/
  latent_upscale_models: latent_upscale_models/
  loras: loras/
YAML

echo "==> 8. Symlink gemma into models/text_encoders/ (extra_model_paths can't always reach it)"
mkdir -p "$COMFY_DIR/models/text_encoders"
for f in gemma_3_12B_it gemma_3_12B_it_fp8_e4m3fn gemma_3_12B_it_fp8_scaled; do
    src="$MOUNT/models/text_encoders/${f}.safetensors"
    dst="$COMFY_DIR/models/text_encoders/${f}.safetensors"
    if [ -e "$src" ] && [ ! -e "$dst" ]; then
        ln -sf "$src" "$dst"
        echo "    linked $f"
    fi
done

echo "==> 9. Install our workflow as a saved workflow"
mkdir -p "$COMFY_DIR/user/default/workflows"
if [ -f "$MOUNT/workflow_template_api.json" ]; then
    cp "$MOUNT/workflow_template_api.json" "$COMFY_DIR/user/default/workflows/ltx-api.json"
    echo "    workflow → user/default/workflows/ltx-api.json"
else
    echo "    NOTE: $MOUNT/workflow_template_api.json not found — scp it after setup"
fi

echo "==> 10. Sanity check models"
ls -la "$MOUNT/models/checkpoints/" 2>&1 | head -5

echo
echo "==> 11. Launch ComfyUI on 0.0.0.0:$PORT (background)"
cd "$COMFY_DIR"
nohup python3 -u main.py --listen 0.0.0.0 --port "$PORT" --disable-auto-launch \
    > "$MOUNT/comfyui.log" 2>&1 &
COMFY_PID=$!
echo "    pid=$COMFY_PID"
echo "    log: tail -f $MOUNT/comfyui.log"

echo
echo "==> 12. Wait for /system_stats to come up (60s)"
for i in $(seq 1 30); do
    if curl -sSf "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
        echo "    READY"
        break
    fi
    sleep 2
done

echo
echo "ComfyUI ready on container :$PORT"
echo "RunPod proxy URL (use this from your browser):"
HOSTNAME=$(hostname)
echo "  https://${HOSTNAME%-*}-${PORT}.proxy.runpod.net/"
