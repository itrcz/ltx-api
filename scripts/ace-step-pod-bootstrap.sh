#!/usr/bin/env bash
# Bootstrap for ACE-Step 1.5 XL on a RunPod pod with ComfyUI.
#
# Design rules learned from earlier failed runs:
#   1. NEVER let PID 1 die. If anything fails, drop into `sleep infinity`.
#      Otherwise RunPod restarts the container in a loop and we lose all
#      progress + visibility.
#   2. Two independent visibility surfaces are up BEFORE any heavy work:
#        :8188 → tiny http server streaming /workspace/bootstrap.log
#        :8888 → JupyterLab (pre-installed in runpod/pytorch image)
#      If anything goes wrong we always have a way in.
#   3. Don't trust ComfyUI's requirements.txt — install a curated minimal
#      set instead. The pytorch base already has torch/numpy/pillow/etc.
#   4. Every step uses `|| true` so failures are logged but don't abort.
#      At the end we test for ComfyUI's main.py existing + key models present;
#      if all green → exec main.py; if not → sleep infinity, user can SSH/Jupyter.
#
# Visible URLs (after pod is RUNNING, replace <pod-id>):
#   https://<pod-id>-8188.proxy.runpod.net/log    → live bootstrap log
#   https://<pod-id>-8888.proxy.runpod.net/       → JupyterLab (token printed below)
#   https://<pod-id>-8188.proxy.runpod.net/       → ComfyUI UI (after success)

set -uo pipefail   # NB: NO -e — partial failures must not kill PID 1.

WORKSPACE=${WORKSPACE:-/workspace}
COMFY_DIR=$WORKSPACE/ComfyUI
MODELS=$COMFY_DIR/models
LOG=$WORKSPACE/bootstrap.log
mkdir -p "$WORKSPACE"

exec > >(tee -a "$LOG") 2>&1
echo "=== ace-step bootstrap v3 @ $(date -Iseconds) ==="
echo "[i] $(uname -a)"
echo "[i] python: $(python3 --version 2>&1)"
echo "[i] disk: $(df -h /workspace | tail -1)"
echo "[i] mem: $(free -h | awk '/Mem:/ {print "total "$2" used "$3" free "$4}')"

# ---------- 0. log server on :8188 (always-on diagnostic) ----------
cat > /tmp/logserver.py <<'PYEOF'
import http.server, socketserver, os
LOG = '/workspace/bootstrap.log'
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **kw): pass
    def do_GET(self):
        try:
            data = open(LOG, 'rb').read() if os.path.exists(LOG) else b'(log not yet created)\n'
        except Exception as e:
            data = f'log read error: {e}'.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)
class TS(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
with TS(('0.0.0.0', 8188), H) as httpd:
    print('[logserver] :8188 ready', flush=True)
    httpd.serve_forever()
PYEOF
nohup python3 /tmp/logserver.py >/tmp/logserver.out 2>&1 &
LOG_PID=$!
echo "[+] log server pid=$LOG_PID"
sleep 1

# ---------- 0b. JupyterLab on :8888 (fallback visibility / shell access) ----------
JUPY_TOKEN="ace-$(date +%s | tail -c 6)"
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --ServerApp.token="$JUPY_TOKEN" --ServerApp.password='' --ServerApp.disable_check_xsrf=True \
  --ServerApp.allow_origin='*' --ServerApp.root_dir=/workspace \
  >/tmp/jupy.out 2>&1 &
JUPY_PID=$!
echo "[+] jupyter pid=$JUPY_PID, token=$JUPY_TOKEN  →  visit /lab?token=$JUPY_TOKEN"
sleep 1

# ---------- main work in a function so any failure jumps to keep-alive ----------
run_setup() {
  echo ""
  echo "[+] step 1: minimal apt deps"
  apt-get install -y -qq git ffmpeg ca-certificates 2>&1 | tail -2 || echo "[!] apt skipped"

  echo "[+] step 2: hf_transfer for parallel downloads"
  pip install --no-cache-dir --quiet "huggingface_hub[cli,hf_transfer]>=0.26" hf_transfer 2>&1 | tail -2
  export HF_HUB_ENABLE_HF_TRANSFER=1

  echo "[+] step 3: curated ComfyUI deps (skipping ComfyUI's full requirements.txt)"
  # Subset of what ComfyUI actually needs that may not be in runpod/pytorch image.
  # Torch / torchvision / torchaudio / numpy / pillow / pyyaml / scipy / tqdm / psutil are pre-installed.
  pip install --no-cache-dir --quiet \
    aiohttp aiohttp-cors yarl \
    safetensors einops kornia spandrel \
    soundfile av \
    transformers sentencepiece tokenizers \
    pydantic pydantic-settings \
    websocket-client cachetools \
    alembic SQLAlchemy \
    torchsde \
    2>&1 | tail -3

  echo "[+] step 4: ComfyUI clone"
  if [ ! -d "$COMFY_DIR/.git" ]; then
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR" 2>&1 | tail -3
  fi

  echo "[+] step 5: ComfyUI-AceStep_SFT custom node"
  ACE_NODE=$COMFY_DIR/custom_nodes/ComfyUI-AceStep_SFT
  if [ ! -d "$ACE_NODE/.git" ]; then
    git clone --depth 1 https://github.com/jeankassio/ComfyUI-AceStep_SFT.git "$ACE_NODE" 2>&1 | tail -3
  fi
  if [ -f "$ACE_NODE/requirements.txt" ]; then
    pip install --no-cache-dir --quiet -r "$ACE_NODE/requirements.txt" 2>&1 | tail -3
  fi

  echo "[+] step 6: weights — sequential, with progress markers"
  mkdir -p "$MODELS/diffusion_models" "$MODELS/vae" "$MODELS/text_encoders" "$MODELS/loras"

  dl() {
    local repo=$1 dest=$2
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
      echo "[=] $repo already at $(du -sh "$dest" | cut -f1), skip"
      return 0
    fi
    echo "[+] downloading $repo"
    local t0=$(date +%s)
    huggingface-cli download "$repo" --local-dir "$dest" 2>&1 | tail -5
    local rc=$?
    local dt=$(($(date +%s) - t0))
    if [ "$rc" = "0" ]; then
      echo "[=] $repo done in ${dt}s, $(du -sh "$dest" | cut -f1)"
    else
      echo "[!] $repo FAILED rc=$rc after ${dt}s"
    fi
    return $rc
  }

  dl "ACE-Step/acestep-v15-xl-sft" "$MODELS/diffusion_models/acestep-v15-xl-sft"
  dl "ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format" "$MODELS/vae/ace_v15_vae"
  dl "ACE-Step/acestep-5Hz-lm-4B" "$MODELS/text_encoders/acestep-lm-4B"
  dl "Qwen/Qwen3-Embedding-0.6B" "$MODELS/text_encoders/qwen3-embed-0.6b"

  echo "[+] models on disk:"
  du -sh "$MODELS"/*/* 2>/dev/null | sort -h

  echo "=== bootstrap done @ $(date -Iseconds) ==="
}

run_setup
SETUP_RC=$?

# Validate we can launch ComfyUI
echo ""
if [ "$SETUP_RC" = "0" ] && [ -f "$COMFY_DIR/main.py" ]; then
  echo "[+] all green — killing log server, launching ComfyUI on :8188"
  kill "$LOG_PID" 2>/dev/null || true
  sleep 2
  cd "$COMFY_DIR"
  exec python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header
fi

# If we got here, setup failed. Stay alive so user can debug via /log + Jupyter + SSH.
echo ""
echo "=== SETUP INCOMPLETE — keeping container alive for diagnosis ==="
echo "  /log endpoint on :8188 still serving bootstrap.log"
echo "  Jupyter on :8888  with token=$JUPY_TOKEN"
echo "  SSH key already injected by RunPod"
exec sleep infinity
