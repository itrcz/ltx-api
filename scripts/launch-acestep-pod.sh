#!/usr/bin/env bash
# Launch a RunPod GPU pod running ComfyUI + ACE-Step 1.5 XL-SFT.
#
# Reads bootstrap script from scripts/ace-step-pod-bootstrap.sh, base64-encodes
# it into the pod's dockerStartCmd, POSTs to RunPod REST API, polls until
# RUNNING, and prints the public ComfyUI URL.
#
# Required env (load from .env):
#   RUNPOD_API_KEY
#   HF_TOKEN  (optional — only if any model is gated)
#
# Optional overrides:
#   POD_NAME        default: ace-step-test
#   GPU_TYPE_ID     default: NVIDIA GeForce RTX 5090
#   CLOUD_TYPE      default: COMMUNITY
#   IMAGE_NAME      default: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   DISK_GB         default: 100
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${ROOT}/.env" ]]; then
  set -a; source "${ROOT}/.env"; set +a
fi

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"

POD_NAME="${POD_NAME:-ace-step-test}"
GPU_TYPE_ID="${GPU_TYPE_ID:-NVIDIA GeForce RTX 5090}"
CLOUD_TYPE="${CLOUD_TYPE:-COMMUNITY}"
IMAGE_NAME="${IMAGE_NAME:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-100}"

BOOT_SCRIPT="${ROOT}/scripts/ace-step-pod-bootstrap.sh"
[[ -f "$BOOT_SCRIPT" ]] || { echo "missing $BOOT_SCRIPT"; exit 1; }
B64=$(base64 < "$BOOT_SCRIPT" | tr -d '\n')

# dockerStartCmd: decode the bootstrap script and exec it as PID 1.
# Using `exec` keeps signals flowing (so RunPod can stop the pod cleanly).
INLINE_CMD="echo $B64 | base64 -d > /tmp/boot.sh && chmod +x /tmp/boot.sh && exec /tmp/boot.sh"

ENV_JSON='{}'
if [[ -n "${HF_TOKEN:-}" ]]; then
  ENV_JSON=$(python3 -c "import json,os; print(json.dumps({'HF_TOKEN': os.environ['HF_TOKEN'], 'HUGGING_FACE_HUB_TOKEN': os.environ['HF_TOKEN']}))")
fi

PAYLOAD=$(python3 -c "
import json, sys, os
print(json.dumps({
  'name': os.environ['POD_NAME'],
  'imageName': os.environ['IMAGE_NAME'],
  'gpuTypeIds': [os.environ['GPU_TYPE_ID']],
  'cloudType': os.environ['CLOUD_TYPE'],
  'gpuCount': 1,
  'containerDiskInGb': int(os.environ['DISK_GB']),
  'volumeInGb': 0,
  'ports': '8188/http,8888/http,22/tcp',
  'env': json.loads(os.environ['ENV_JSON']),
  'dockerStartCmd': ['bash', '-c', os.environ['INLINE_CMD']],
}))
" POD_NAME="$POD_NAME" IMAGE_NAME="$IMAGE_NAME" GPU_TYPE_ID="$GPU_TYPE_ID" \
  CLOUD_TYPE="$CLOUD_TYPE" DISK_GB="$DISK_GB" ENV_JSON="$ENV_JSON" INLINE_CMD="$INLINE_CMD")

echo "[+] creating pod $POD_NAME on $GPU_TYPE_ID ($CLOUD_TYPE)..."
RESP=$(curl -sS -X POST "https://rest.runpod.io/v1/pods" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

echo "$RESP" | python3 -m json.tool

POD_ID=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('id',''))")
[[ -n "$POD_ID" ]] || { echo "no pod id in response — bailing"; exit 1; }

echo ""
echo "[+] pod created: $POD_ID"
echo "[+] ComfyUI will be at: https://${POD_ID}-8188.proxy.runpod.net"
echo "[+] tail bootstrap log:  ssh into pod, then: tail -f /workspace/bootstrap.log"
echo ""
echo "[+] polling status (Ctrl-C to stop polling, pod keeps running)..."
while true; do
  S=$(curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$POD_ID" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('desiredStatus',''), '|', d.get('lastStatusChange',''))" 2>/dev/null || echo "?")
  echo "  $(date +%H:%M:%S) $S"
  case "$S" in
    RUNNING*) break ;;
    EXITED*|TERMINATED*) echo "pod exited"; exit 1 ;;
  esac
  sleep 10
done

echo ""
echo "[+] pod RUNNING. ComfyUI URL: https://${POD_ID}-8188.proxy.runpod.net"
echo "[+] note: bootstrap still downloading weights (~50 GB, ~10 min). UI loads after main.py starts."
