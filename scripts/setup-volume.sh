#!/usr/bin/env bash
# Provision a RunPod network volume with the LTX-2.3 prod model set for ComfyUI.
#
# Required env (auto-loaded from .env at repo root):
#   RUNPOD_API_KEY  — RunPod REST API key
#   HF_TOKEN        — HuggingFace token (read access to Lightricks/* and Comfy-Org/*)
#
# Args (env vars):
#   REGION       RunPod data center ID (default: EUR-IS-2). Examples:
#                EUR-IS-2, EUR-NO-1, EU-RO-1, US-CA-2, ...
#   SIZE_GB      Volume size in GB (default: 50)
#   VOLUME_NAME  Volume name (default: ltx-weights-<region-lowercase>)
#
# Usage:
#   REGION=EUR-IS-2 SIZE_GB=50 ./scripts/setup-volume.sh
#   REGION=EUR-NO-1 SIZE_GB=50 ./scripts/setup-volume.sh
#
# Idempotent: reuses an existing volume of the same name in the same region,
# and skips files already present at the expected size.
#
# Downloads (~50 GB total) + ComfyUI symlink farm:
#   ltx23/ltx-2.3-22b-dev-fp8.safetensors                       27 GB  (Lightricks/LTX-2.3-fp8)
#   ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors            1 GB  (Lightricks/LTX-2.3)
#   ltx23/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors   7 GB  (Lightricks/LTX-2.3)
#   gemma-fp8/gemma_3_12B_it_fp8_scaled.safetensors             12 GB  (Comfy-Org/ltx-2)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -f "$ROOT/.env" ]] && { set -a; source "$ROOT/.env"; set +a; }
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
: "${HF_TOKEN:?HF_TOKEN not set}"

REGION="${REGION:-EUR-IS-1}"
SIZE_GB="${SIZE_GB:-60}"  # 50 is too tight: HF caches a 12GB partial during gemma download → ~59GB peak
REGION_LC="$(echo "$REGION" | tr '[:upper:]' '[:lower:]')"
VOLUME_NAME="${VOLUME_NAME:-ltx-weights-${REGION_LC}}"
SSH_KEY="$ROOT/.runpod-ssh/key"
SSH_PUB="$ROOT/.runpod-ssh/key.pub"

if [[ ! -f "$SSH_KEY" || ! -f "$SSH_PUB" ]]; then
    echo "ERROR: SSH keys missing at $ROOT/.runpod-ssh/{key,key.pub}" >&2
    echo "  Create with:  ssh-keygen -t ed25519 -N '' -f $SSH_KEY -C ltx-volume-setup" >&2
    exit 1
fi

API="https://rest.runpod.io/v1"
auth_curl() { curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" "$@"; }

export REGION VOLUME_NAME SIZE_GB
echo "==> Region=$REGION  Size=${SIZE_GB}GB  Name=$VOLUME_NAME"

# ---- 1. Find or create network volume ----
VOLUME_ID=$(auth_curl "$API/networkvolumes" | python3 -c "
import sys, json, os
region, name = os.environ['REGION'], os.environ['VOLUME_NAME']
for v in json.load(sys.stdin):
    if v.get('dataCenterId') == region and v.get('name') == name:
        print(v['id']); break
")

if [[ -z "$VOLUME_ID" ]]; then
    echo "==> Creating volume..."
    VOLUME_ID=$(auth_curl -X POST -H "Content-Type: application/json" \
        "$API/networkvolumes" \
        -d "{\"name\":\"$VOLUME_NAME\",\"size\":$SIZE_GB,\"dataCenterId\":\"$REGION\"}" \
        | python3 -c "import sys, json; print(json.load(sys.stdin).get('id',''))")
    [[ -n "$VOLUME_ID" ]] || { echo "ERROR: volume create failed" >&2; exit 1; }
    echo "    created: $VOLUME_ID"
else
    echo "==> Reusing existing volume: $VOLUME_ID"
fi

# ---- 2. Spawn pod with volume mounted ----
# Try CPU first (cheap); fall back to a small GPU if the region has no CPU
# capacity (e.g. EUR-NO-1 was CPU-empty during initial provisioning).
PUBKEY=$(cat "$SSH_PUB")

build_pod_body() {
    # $1 = "CPU" or "GPU"; $2 = optional gpuTypeId for GPU mode.
    VOLUME_ID="$VOLUME_ID" REGION="$REGION" PUBKEY="$PUBKEY" \
        MODE="$1" GPU_TYPE="${2:-}" python3 -c "
import json, os
mode = os.environ['MODE']
body = {
    'name': f'ltx-vol-setup-{os.environ[\"REGION\"].lower()}',
    'imageName': 'runpod/base:0.6.2-cpu' if mode == 'CPU' else 'runpod/base:0.6.2-cuda12.4.1',
    'computeType': mode,
    'containerDiskInGb': 5,
    'networkVolumeId': os.environ['VOLUME_ID'],
    'volumeMountPath': '/runpod-volume',
    'ports': ['22/tcp'],
    'dataCenterIds': [os.environ['REGION']],
    'env': {'PUBLIC_KEY': os.environ['PUBKEY']},
}
if mode == 'CPU':
    body['vcpuCount'] = 2
else:
    body['gpuCount'] = 1
    body['gpuTypeIds'] = [os.environ['GPU_TYPE']]
print(json.dumps(body))
"
}

try_create_pod() {
    local body="$1"
    local resp id
    resp=$(auth_curl -X POST -H "Content-Type: application/json" "$API/pods" -d "$body")
    id=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('id','') if isinstance(d, dict) else '')
except: pass
")
    if [[ -n "$id" ]]; then
        echo "$id"
        return 0
    fi
    LAST_ERR=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    e = d if isinstance(d, dict) else (d[0] if d else {})
    print((e.get('error') or '')[:200])
except: pass
")
    return 1
}

echo "==> Spawning pod in $REGION..."
POD_ID=""
LAST_ERR=""
# Try CPU first
if POD_ID=$(try_create_pod "$(build_pod_body CPU)"); then
    echo "    pod: $POD_ID (CPU)"
else
    echo "    CPU unavailable in $REGION ($LAST_ERR)"
    # Fall back to small GPUs in availability order
    for gpu in "NVIDIA RTX A4000" "NVIDIA RTX 4000 Ada Generation" "NVIDIA RTX A2000" \
               "NVIDIA GeForce RTX 3070" "NVIDIA GeForce RTX 3080" "NVIDIA GeForce RTX 4090"; do
        echo "    trying GPU: $gpu"
        if POD_ID=$(try_create_pod "$(build_pod_body GPU "$gpu")"); then
            echo "    pod: $POD_ID (GPU=$gpu)"
            break
        fi
    done
fi
[[ -n "$POD_ID" ]] || { echo "ERROR: no capacity in $REGION (last: $LAST_ERR)" >&2; exit 1; }

cleanup() {
    echo "==> Terminating pod $POD_ID (volume persists)..."
    auth_curl -X DELETE "$API/pods/$POD_ID" -o /dev/null -w "    HTTP %{http_code}\n" || true
}
trap cleanup EXIT

# ---- 3. Wait for SSH ----
echo "==> Waiting for SSH..."
IP=""; PORT=""
for i in $(seq 1 60); do
    INFO=$(auth_curl "$API/pods/$POD_ID")
    IP=$(echo "$INFO"  | python3 -c "import sys, json; p=json.load(sys.stdin); print(p.get('publicIp') or '')")
    PORT=$(echo "$INFO" | python3 -c "import sys, json; p=json.load(sys.stdin); pm=p.get('portMappings'); print(pm.get('22','') if pm else '')")
    if [[ -n "$IP" && -n "$PORT" ]] && \
       ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
           -o ConnectTimeout=5 -i "$SSH_KEY" -p "$PORT" "root@$IP" 'true' 2>/dev/null; then
        echo "    ssh ready: root@$IP:$PORT"
        break
    fi
    sleep 5
done
[[ -n "$IP" && -n "$PORT" ]] || { echo "ERROR: pod never came up" >&2; exit 1; }

# ---- 4. Run remote download + symlink setup ----
echo "==> Downloading models + creating symlinks..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i "$SSH_KEY" -p "$PORT" "root@$IP" \
    "HF_TOKEN='$HF_TOKEN' bash -s" <<'REMOTE'
set -euo pipefail
export HF_TOKEN
export HF_HUB_DOWNLOAD_TIMEOUT=180

if ! command -v hf >/dev/null 2>&1; then
    echo "  installing huggingface_hub..."
    pip install -q --upgrade 'huggingface_hub>=0.26' 2>&1 | tail -3
fi
# hf_transfer / Xet parallel writers crash on RunPod's MFS network volume
# ("Internal Writer Error: Background writer channel closed"). Use the
# stock chunked-HTTP downloader instead — slower but reliable.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_XET_HIGH_PERFORMANCE=0

M=/runpod-volume/models
TMPROOT=/runpod-volume/.tmp
# Clear any partial downloads from previous failed runs — they eat quota.
rm -rf "$TMPROOT"
mkdir -p "$M"/{ltx23/loras,gemma-fp8,checkpoints,text_encoders,latent_upscale_models,loras/ltxv/ltx2} "$TMPROOT"
echo "  free space at start:"
df -h /runpod-volume | tail -1 | awk '{print "    " $0}'

# repo:repo_path:dest_relative_to_$M:expected_size_MB
FILES=(
    "Lightricks/LTX-2.3-fp8:ltx-2.3-22b-dev-fp8.safetensors:ltx23/ltx-2.3-22b-dev-fp8.safetensors:27750"
    "Lightricks/LTX-2.3:ltx-2.3-spatial-upscaler-x2-1.1.safetensors:ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors:949"
    "Lightricks/LTX-2.3:ltx-2.3-22b-distilled-lora-384-1.1.safetensors:ltx23/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors:7250"
    "Comfy-Org/ltx-2:split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors:gemma-fp8/gemma_3_12B_it_fp8_scaled.safetensors:12600"
)

dl_one() {
    local repo="$1" rpath="$2" dest="$3" expected_mb="$4"
    local target="$M/$dest"
    if [[ -f "$target" ]]; then
        local cur_mb=$(( $(stat -c %s "$target") / 1024 / 1024 ))
        local diff=$(( cur_mb > expected_mb ? cur_mb - expected_mb : expected_mb - cur_mb ))
        if [[ $diff -lt 200 ]]; then
            echo "  [skip]   $dest (${cur_mb}MB)"
            return
        fi
        echo "  [redl]   $dest size mismatch (${cur_mb}MB vs ~${expected_mb}MB)"
        rm -f "$target"
    fi
    echo "  [dl]     $dest from $repo"
    # Tmpdir on the network volume — container disk is only 5 GB.
    local tmp
    tmp=$(mktemp -d "$TMPROOT/hfdl.XXXXXX")
    hf download "$repo" "$rpath" --local-dir "$tmp" >/dev/null
    mv "$tmp/$rpath" "$target"
    rm -rf "$tmp"
}

for entry in "${FILES[@]}"; do
    IFS=':' read -r repo rpath dest mb <<<"$entry"
    dl_one "$repo" "$rpath" "$dest" "$mb"
done

# ComfyUI symlink farm — extra_model_paths.yaml expects:
#   checkpoints/, text_encoders/, latent_upscale_models/, loras/
ln -sfn ../ltx23/ltx-2.3-22b-dev-fp8.safetensors                            "$M/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
ln -sfn ../ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors                "$M/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
ln -sfn ../gemma-fp8/gemma_3_12B_it_fp8_scaled.safetensors                  "$M/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"
ln -sfn ../../../ltx23/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors "$M/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

# IMPORTANT: the live workflow (worker/src/workflow_template_api.json) references
# gemma_3_12B_it_fp8_e4m3fn.safetensors — a locally-quantized file NOT on HuggingFace
# and NOT drop-in compatible with the public _scaled variant. To make this volume
# usable for inference, run scripts/migrate-gemma-from-prod.sh afterwards to copy the
# real _e4m3fn (13 GB) from the original 130 GB EU-RO-1 prod volume (c25vvptq5f).
echo
echo "  NOTE: workflow uses gemma_3_12B_it_fp8_e4m3fn.safetensors; this run installed _scaled."
echo "        Run scripts/migrate-gemma-from-prod.sh REGION=$REGION VOLUME_ID=$VOLUME_ID"
echo "        to swap in the byte-compatible _e4m3fn from the existing prod volume."

echo
echo "==> Layout:"
ls -la "$M"
echo
du -sh "$M"/*/ 2>/dev/null | sort -h
echo
df -h /runpod-volume | tail -1
REMOTE

echo
echo "==> Done. Volume $VOLUME_ID ($VOLUME_NAME) in $REGION ready."
echo "    networkVolumeId for endpoint/template: $VOLUME_ID"
