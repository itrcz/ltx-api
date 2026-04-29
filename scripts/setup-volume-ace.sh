#!/usr/bin/env bash
# Provision ACE-Step 1.5 XL-SFT weights onto a RunPod Network Volume.
#
# Usage:
#   REGION=EUR-NO-1 VOLUME_ID=azs9bp5b96 ./scripts/setup-volume-ace.sh
#
# What it does:
#   1. Spawns a tiny CPU-light pod in REGION mounting VOLUME_ID at /runpod-volume
#   2. The pod runs a robust bootstrap (log server on :8888, never crashes,
#      always sleep-infinity at end so we have visibility)
#   3. Bootstrap downloads 4 split files from Comfy-Org/ace_step_1.5_ComfyUI_files
#      directly onto the volume, alongside existing LTX weights — no overwrites
#   4. Creates Audio/ symlinks (jeankassio's custom node convention)
#   5. Touches /runpod-volume/.ace-setup-done as the success sentinel
#   6. Caller polls the log URL until the sentinel is mentioned, then terminates
#
# Required env (load from .env):
#   RUNPOD_API_KEY
#   HF_TOKEN
#   REGION                 — e.g. EUR-NO-1 / EUR-IS-1 / EU-RO-1
#   VOLUME_ID              — existing network volume in that region
# Optional:
#   GPU_TYPE_ID            — default "NVIDIA RTX A6000" (cheap, plentiful, ~$0.49/hr)
#   IMAGE_NAME             — default runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "${ROOT}/.env" ]]; then set -a; source "${ROOT}/.env"; set +a; fi

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
: "${HF_TOKEN:?HF_TOKEN not set}"
: "${REGION:?REGION not set, e.g. EUR-NO-1}"
: "${VOLUME_ID:?VOLUME_ID not set}"

GPU_TYPE_ID="${GPU_TYPE_ID:-NVIDIA RTX A6000}"
IMAGE_NAME="${IMAGE_NAME:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
API="https://rest.runpod.io/v1"
H_AUTH="Authorization: Bearer ${RUNPOD_API_KEY}"
H_JSON="Content-Type: application/json"

# Bootstrap script that runs inside the pod. Robust against failure.
BOOT_SCRIPT='
#!/usr/bin/env bash
set -uo pipefail   # NB: NO -e — partial failures must not kill PID 1
LOG=/workspace/setup.log
mkdir -p /workspace
exec > >(tee -a "$LOG") 2>&1
echo "=== ace volume setup @ $(date -Iseconds) ==="
echo "[i] $(uname -a)"
echo "[i] disk: $(df -h /runpod-volume | tail -1)"
echo "[i] existing models on volume:"
ls -la /runpod-volume/models/ 2>/dev/null | head -20

# Always-on log server on :8888 — visible at https://<pod>-8888.proxy.runpod.net/
cat > /tmp/logserver.py <<PYEOF
import http.server, socketserver, os
LOG = "/workspace/setup.log"
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self,*a,**k): pass
    def do_GET(self):
        try:
            data = open(LOG,"rb").read() if os.path.exists(LOG) else b"(log not yet created)"
        except Exception as e:
            data = f"log err: {e}".encode()
        self.send_response(200)
        self.send_header("Content-Type","text/plain; charset=utf-8")
        self.send_header("Cache-Control","no-store"); self.end_headers()
        self.wfile.write(data)
class TS(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True; daemon_threads = True
with TS(("0.0.0.0",8888),H) as h:
    print("[logserver] :8888 ready", flush=True); h.serve_forever()
PYEOF
nohup python3 /tmp/logserver.py >/tmp/logserver.out 2>&1 &
echo "[+] log server pid=$! on :8888"
sleep 1

apt-get update -qq && apt-get install -y -qq curl ca-certificates 2>&1 | tail -2 || echo "[!] apt skip"
pip install --no-cache-dir --quiet "huggingface_hub[hf_transfer]>=0.26" hf_transfer 2>&1 | tail -2
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p /runpod-volume/models/{diffusion_models,text_encoders,vae,loras,Audio}

python3 - <<PY
import os, shutil
from huggingface_hub import hf_hub_download
REPO = "Comfy-Org/ace_step_1.5_ComfyUI_files"
TARGETS = [
    ("split_files/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors",
     "/runpod-volume/models/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors"),
    ("split_files/text_encoders/qwen_0.6b_ace15.safetensors",
     "/runpod-volume/models/text_encoders/qwen_0.6b_ace15.safetensors"),
    ("split_files/text_encoders/qwen_4b_ace15.safetensors",
     "/runpod-volume/models/text_encoders/qwen_4b_ace15.safetensors"),
    ("split_files/vae/ace_1.5_vae.safetensors",
     "/runpod-volume/models/vae/ace_1.5_vae.safetensors"),
]
for src, dst in TARGETS:
    if os.path.exists(dst) and os.path.getsize(dst) > 1e8:
        print(f"[=] {dst} ({os.path.getsize(dst)/1e9:.2f} GB) — skip", flush=True); continue
    print(f"[+] {src}", flush=True)
    p = hf_hub_download(REPO, src, local_dir="/tmp/hf_dl")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(p, dst)
    print(f"    → {dst} ({os.path.getsize(dst)/1e9:.2f} GB)", flush=True)
PY

# Audio/ symlinks (jeankassio custom-node naming convention)
ln -sfn /runpod-volume/models/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors \
        /runpod-volume/models/Audio/acestep_v1.5_sft_xl.safetensors
ln -sfn /runpod-volume/models/text_encoders/qwen_0.6b_ace15.safetensors \
        /runpod-volume/models/Audio/qwen_0.6b_ace15.safetensors
ln -sfn /runpod-volume/models/text_encoders/qwen_4b_ace15.safetensors \
        /runpod-volume/models/Audio/qwen_4b_ace15.safetensors
ln -sfn /runpod-volume/models/vae/ace_1.5_vae.safetensors \
        /runpod-volume/models/Audio/ace_1.5_vae.safetensors

echo ""
echo "[+] final layout (ACE additions only):"
ls -lh /runpod-volume/models/Audio/
ls -lh /runpod-volume/models/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors 2>/dev/null
ls -lh /runpod-volume/models/text_encoders/qwen_*.safetensors 2>/dev/null
ls -lh /runpod-volume/models/vae/ace_*.safetensors 2>/dev/null
df -h /runpod-volume

# Sentinel — picked up by the orchestrator to know we are done
date -Iseconds > /runpod-volume/.ace-setup-done
echo ""
echo "=== ACE-SETUP-DONE @ $(date -Iseconds) ==="
exec sleep infinity   # keep log server reachable, dont let container restart
'

BOOT_B64=$(printf '%s' "${BOOT_SCRIPT}" | base64 | tr -d '\n')
INLINE_CMD="echo ${BOOT_B64} | base64 -d > /tmp/boot.sh && chmod +x /tmp/boot.sh && exec /tmp/boot.sh"

PAYLOAD=$(GPU_TYPE_ID="${GPU_TYPE_ID}" IMAGE_NAME="${IMAGE_NAME}" \
  VOLUME_ID="${VOLUME_ID}" REGION="${REGION}" \
  HF_TOKEN="${HF_TOKEN}" INLINE_CMD="${INLINE_CMD}" \
  python3 -c "
import json, os
print(json.dumps({
  'name': f'ace-setup-{os.environ[\"REGION\"].lower()}',
  'imageName': os.environ['IMAGE_NAME'],
  'gpuTypeIds': [os.environ['GPU_TYPE_ID']],
  'cloudType': 'SECURE',
  'gpuCount': 1,
  'containerDiskInGb': 20,
  'networkVolumeId': os.environ['VOLUME_ID'],
  'volumeMountPath': '/runpod-volume',
  'ports': ['8888/http', '22/tcp'],
  'env': {'HF_TOKEN': os.environ['HF_TOKEN'], 'HUGGING_FACE_HUB_TOKEN': os.environ['HF_TOKEN']},
  'dockerStartCmd': ['bash', '-c', os.environ['INLINE_CMD']],
}))
")

echo "[+] spawning setup pod for ${REGION} (volume=${VOLUME_ID})..."
RESP=$(curl -sS -X POST "${API}/pods" -H "${H_AUTH}" -H "${H_JSON}" -d "${PAYLOAD}")
POD_ID=$(echo "${RESP}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('id') or '')")
[[ -n "${POD_ID}" ]] || { echo "pod create failed: ${RESP}"; exit 1; }
echo "  → pod id: ${POD_ID}"
echo "  → log:    https://${POD_ID}-8888.proxy.runpod.net/"
echo ""
echo "Watch progress with:"
echo "  while true; do clear; curl -sS https://${POD_ID}-8888.proxy.runpod.net/ | tail -30; sleep 10; done"
echo ""
echo "When log shows 'ACE-SETUP-DONE', terminate:"
echo "  curl -X DELETE -H \"${H_AUTH}\" ${API}/pods/${POD_ID}"
