#!/usr/bin/env bash
# Copy the byte-exact `gemma_3_12B_it_fp8_e4m3fn.safetensors` (13 GB) from
# the original EU-RO-1 prod volume `c25vvptq5f` to a target network volume,
# then fix the symlinks so ComfyUI workflows that reference _e4m3fn work.
#
# Why this exists: that file was locally quantized on a previous pod and is
# NOT on HuggingFace. setup-volume.sh installs the publicly available
# `gemma_3_12B_it_fp8_scaled.safetensors` from Comfy-Org/ltx-2 instead, but
# the two files are NOT drop-in compatible — _scaled has different per-layer
# dtypes, which crashes the SigLIP vision tower in inference:
#   RuntimeError: Promotion for Float8 Types is not supported, attempted to
#   promote BFloat16 and Float8_e4m3fn
#
# This script:
#   1. Spawns a source pod in EU-RO-1 with c25vvptq5f mounted.
#   2. Starts an HTTP server on the source pod exposing the gemma file.
#   3. Spawns a target pod in REGION with VOLUME_ID mounted (CPU first,
#      GPU fallback if the region has no CPU capacity).
#   4. Has the target pod curl the file, replaces the _scaled stub with the
#      real _e4m3fn, fixes the text_encoders/ symlink.
#   5. Terminates both pods.
#
# Required env (loaded from .env):
#   RUNPOD_API_KEY
#
# Args (env vars):
#   REGION       Target region (e.g. EUR-IS-1, EUR-NO-1, EU-RO-1)
#   VOLUME_ID    Target network volume ID (the one populated by setup-volume.sh)
#
# Idempotent: if the file already matches expected size, it exits early.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -f "$ROOT/.env" ]] && { set -a; source "$ROOT/.env"; set +a; }
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
: "${REGION:?REGION not set}"
: "${VOLUME_ID:?VOLUME_ID not set}"

SOURCE_VOLUME_ID="${SOURCE_VOLUME_ID:-c25vvptq5f}"
SOURCE_REGION="${SOURCE_REGION:-EU-RO-1}"
SOURCE_PATH="${SOURCE_PATH:-/runpod-volume/models/gemma-fp8/gemma_3_12B_it_fp8_e4m3fn.safetensors}"
EXPECTED_SIZE="${EXPECTED_SIZE:-13210008986}"
SSH_KEY="$ROOT/.runpod-ssh/key"
SSH_PUB="$ROOT/.runpod-ssh/key.pub"
PUBKEY=$(cat "$SSH_PUB")

API="https://rest.runpod.io/v1"
auth_curl() { curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" "$@"; }

# ---- helpers ----
build_pod() {
    # $1 mode (CPU|GPU), $2 region, $3 volume_id, $4 ports json (e.g. '["22/tcp"]'), $5 gpu (optional)
    PUBKEY="$PUBKEY" REGION="$2" VOLUME_ID="$3" PORTS="$4" MODE="$1" GPU="${5:-}" python3 -c "
import json, os
m = os.environ['MODE']
b = {
    'name': f'ltx-gemma-{os.environ[\"REGION\"].lower()}',
    'imageName': 'runpod/base:0.6.2-cpu' if m=='CPU' else 'runpod/base:0.6.2-cuda12.4.1',
    'computeType': m,
    'containerDiskInGb': 5,
    'networkVolumeId': os.environ['VOLUME_ID'],
    'volumeMountPath': '/runpod-volume',
    'ports': json.loads(os.environ['PORTS']),
    'dataCenterIds': [os.environ['REGION']],
    'env': {'PUBLIC_KEY': os.environ['PUBKEY']},
}
if m == 'CPU':
    b['vcpuCount'] = 2
else:
    b['gpuCount'] = 1
    b['gpuTypeIds'] = [os.environ['GPU']]
print(json.dumps(b))
"
}
try_pod() {
    local body="$1" resp id
    resp=$(auth_curl -X POST -H "Content-Type: application/json" "$API/pods" -d "$body")
    id=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('id','') if isinstance(d, dict) else '')
except: pass
")
    [[ -n "$id" ]] && { echo "$id"; return 0; }
    return 1
}
spawn_with_fallback() {
    # $1 region, $2 volume_id, $3 ports
    local id=""
    if id=$(try_pod "$(build_pod CPU "$1" "$2" "$3")"); then echo "$id"; return; fi
    for gpu in "NVIDIA RTX A4000" "NVIDIA RTX 4000 Ada Generation" "NVIDIA RTX A2000" \
               "NVIDIA GeForce RTX 3070" "NVIDIA GeForce RTX 3080" "NVIDIA GeForce RTX 3090" \
               "NVIDIA GeForce RTX 4090" "NVIDIA RTX A5000" "NVIDIA L4"; do
        if id=$(try_pod "$(build_pod GPU "$1" "$2" "$3" "$gpu")"); then
            echo "$id"; return
        fi
    done
    return 1
}
wait_for_ssh() {
    # $1 pod_id; sets globals IP, PORT
    local pod_id="$1"
    IP=""; PORT=""
    for i in $(seq 1 60); do
        local info; info=$(auth_curl "$API/pods/$pod_id")
        IP=$(echo "$info"   | python3 -c "import sys, json; p=json.load(sys.stdin); print(p.get('publicIp') or '')")
        PORT=$(echo "$info" | python3 -c "import sys, json; p=json.load(sys.stdin); pm=p.get('portMappings'); print(pm.get('22','') if pm else '')")
        if [[ -n "$IP" && -n "$PORT" ]] && \
           ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
               -o ConnectTimeout=5 -i "$SSH_KEY" -p "$PORT" "root@$IP" 'true' 2>/dev/null; then
            return 0
        fi
        sleep 5
    done
    return 1
}
delete_pod() {
    auth_curl -X DELETE "$API/pods/$1" -o /dev/null -w "    HTTP %{http_code}\n" || true
}

# ---- 1. Spawn source pod with the prod volume + 8080/tcp open for HTTP ----
echo "==> Spawning source pod in $SOURCE_REGION with $SOURCE_VOLUME_ID..."
SRC_POD=$(try_pod "$(build_pod CPU "$SOURCE_REGION" "$SOURCE_VOLUME_ID" '["22/tcp","8080/tcp"]')") \
    || { echo "ERROR: source pod create failed (CPU unavailable in $SOURCE_REGION)" >&2; exit 1; }
echo "    src pod: $SRC_POD"

cleanup() {
    [[ -n "${TGT_POD:-}" ]] && { echo "==> Terminating target pod $TGT_POD..."; delete_pod "$TGT_POD"; }
    [[ -n "${SRC_POD:-}" ]] && { echo "==> Terminating source pod $SRC_POD..."; delete_pod "$SRC_POD"; }
}
trap cleanup EXIT

echo "==> Waiting for source SSH..."
wait_for_ssh "$SRC_POD" || { echo "ERROR: source pod SSH never came up" >&2; exit 1; }
SRC_IP="$IP"; SRC_PORT="$PORT"
echo "    src ssh: root@$SRC_IP:$SRC_PORT"

echo "==> Starting HTTP server on source pod..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$SSH_KEY" \
    -p "$SRC_PORT" "root@$SRC_IP" "
set -e
cd \"\$(dirname '$SOURCE_PATH')\"
nohup python3 -m http.server 8080 --bind 0.0.0.0 > /tmp/http.log 2>&1 &
sleep 2
curl -fsI http://localhost:8080/\"\$(basename '$SOURCE_PATH')\" | head -3
"
# Get the public TCP port mapping for 8080
SRC_HTTP_PORT=$(auth_curl "$API/pods/$SRC_POD" | python3 -c "
import sys, json
p = json.load(sys.stdin)
pm = p.get('portMappings') or {}
print(pm.get('8080',''))
")
[[ -n "$SRC_HTTP_PORT" ]] || { echo "ERROR: 8080 port not mapped" >&2; exit 1; }
SOURCE_URL="http://$SRC_IP:$SRC_HTTP_PORT/$(basename "$SOURCE_PATH")"
echo "    serving from: $SOURCE_URL"

# ---- 2. Spawn target pod ----
echo "==> Spawning target pod in $REGION with $VOLUME_ID..."
TGT_POD=$(spawn_with_fallback "$REGION" "$VOLUME_ID" '["22/tcp"]') \
    || { echo "ERROR: no capacity in $REGION" >&2; exit 1; }
echo "    tgt pod: $TGT_POD"

echo "==> Waiting for target SSH..."
wait_for_ssh "$TGT_POD" || { echo "ERROR: target pod SSH never came up" >&2; exit 1; }
echo "    tgt ssh: root@$IP:$PORT"

# ---- 3. Run migration on target ----
echo "==> Migrating gemma on $VOLUME_ID..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i "$SSH_KEY" -p "$PORT" "root@$IP" \
    "SOURCE_URL='$SOURCE_URL' EXPECTED_SIZE='$EXPECTED_SIZE' bash -s" <<'REMOTE'
set -euo pipefail
M=/runpod-volume/models
TARGET="$M/gemma-fp8/gemma_3_12B_it_fp8_e4m3fn.safetensors"

# Idempotency: skip if already correct
if [[ -f "$TARGET" ]]; then
    GOT=$(stat -c %s "$TARGET")
    if [[ "$GOT" == "$EXPECTED_SIZE" ]]; then
        echo "  [skip] $TARGET already correct ($GOT bytes)"
        exit 0
    fi
    echo "  size mismatch on existing file: $GOT vs $EXPECTED_SIZE — replacing"
    rm -f "$TARGET"
fi

# Drop the misleading alias symlink and the _scaled stub if they exist
[[ -L "$M/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors" ]] && \
    rm "$M/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors"
rm -f "$M/gemma-fp8/gemma_3_12B_it_fp8_scaled.safetensors"
rm -f "$M/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"

echo "  downloading 13 GB from $SOURCE_URL..."
mkdir -p "$(dirname "$TARGET")"
curl -fsSL -o "$TARGET" "$SOURCE_URL"

GOT=$(stat -c %s "$TARGET")
[[ "$GOT" == "$EXPECTED_SIZE" ]] || { echo "ERROR: size mismatch got=$GOT expected=$EXPECTED_SIZE"; exit 1; }
echo "  size OK: $GOT"

ln -sfn ../gemma-fp8/gemma_3_12B_it_fp8_e4m3fn.safetensors \
        "$M/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors"

df -h /runpod-volume | tail -1 | awk '{print "    " $0}'
REMOTE

echo "==> Done. $VOLUME_ID now has the real gemma_3_12B_it_fp8_e4m3fn.safetensors."
