#!/usr/bin/env bash
# Create a Yotta Labs Serverless endpoint for the LTX-2.3 worker (QUEUE mode).
#
# Required env (load from .env):
#   YOTTA_API_KEY
#   GHCR_USER        — for image path ghcr.io/$GHCR_USER/gr-tv-vst:$IMAGE_TAG
#   IMAGE_TAG        — e.g. v0.4.0 (must already be pushed to GHCR public)
#   R2_PUBLIC_URL    — e.g. https://s3.unne.ai (passed to worker env for runtime)
#   S3_*             — for handler's video upload (Yandex S3, separate from R2)
#
# Optional:
#   ENDPOINT_NAME    default: ltx-2-3
#   WORKERS          default: 1
#   GPU_TYPE         default: RTX_5090  (Yotta's enum names)
#   REGION           default: any  (Yotta picks)
#   DISK_GB          default: 80
set -euo pipefail

: "${YOTTA_API_KEY:?YOTTA_API_KEY required}"
: "${GHCR_USER:?GHCR_USER required}"
: "${IMAGE_TAG:?IMAGE_TAG required (e.g. v0.4.0)}"
: "${R2_PUBLIC_URL:?R2_PUBLIC_URL required (e.g. https://s3.unne.ai)}"
: "${S3_ENDPOINT_URL:?S3_ENDPOINT_URL required}"
: "${S3_BUCKET:?S3_BUCKET required}"
: "${S3_ACCESS_KEY_ID:?S3_ACCESS_KEY_ID required}"
: "${S3_SECRET_ACCESS_KEY:?S3_SECRET_ACCESS_KEY required}"

ENDPOINT_NAME="${ENDPOINT_NAME:-ltx-2-3}"
WORKERS="${WORKERS:-1}"
# Yotta available GPU types (per /v2/vms/types):
#   NVIDIA_H100_80G       — Hopper sm_90, FP8 ✓ — eu-2
#   NVIDIA_H200_141G      — Hopper sm_90, FP8 ✓ — eu-2, eu-3
#   NVIDIA_RTX_6000_Ada_48G — Ada sm_89, FP8 ✓ — eu-3
#   NVIDIA_RTX_PRO_6000_96G — Blackwell, FP8 ✓ — us-central-7, eu-3
# Note: RTX 5090 / 4090 not currently offered.
GPU_TYPE="${GPU_TYPE:-NVIDIA_H100_80G}"
REGION="${REGION:-eu-2}"
DISK_GB="${DISK_GB:-80}"
IMAGE_FULL="ghcr.io/${GHCR_USER}/gr-tv-vst:${IMAGE_TAG}"

PAYLOAD=$(IMAGE_FULL="$IMAGE_FULL" \
          ENDPOINT_NAME="$ENDPOINT_NAME" \
          WORKERS="$WORKERS" \
          GPU_TYPE="$GPU_TYPE" \
          REGION="$REGION" \
          DISK_GB="$DISK_GB" \
          R2_PUBLIC_URL="$R2_PUBLIC_URL" \
          S3_ENDPOINT_URL="$S3_ENDPOINT_URL" \
          S3_BUCKET="$S3_BUCKET" \
          S3_REGION="${S3_REGION:-auto}" \
          S3_ACCESS_KEY_ID="$S3_ACCESS_KEY_ID" \
          S3_SECRET_ACCESS_KEY="$S3_SECRET_ACCESS_KEY" \
          python3 <<'PY'
import json, os
print(json.dumps({
    "name": os.environ["ENDPOINT_NAME"],
    "image": os.environ["IMAGE_FULL"],
    "imageRegistry": "ghcr.io",
    "workers": int(os.environ["WORKERS"]),
    "containerVolumeInGb": int(os.environ["DISK_GB"]),
    "serviceMode": "QUEUE",
    "minSingleCardVramInGb": 24,
    "resources": [{
        "region": os.environ["REGION"],
        "gpuType": os.environ["GPU_TYPE"],
        "gpuCount": 1,
    }],
    "expose": {"port": 8000, "protocol": "HTTP"},
    "envVars": [
        {"key": "WORKER_PORT", "value": "8000"},
        {"key": "R2_PUBLIC_URL", "value": os.environ["R2_PUBLIC_URL"]},
        {"key": "S3_ENDPOINT_URL", "value": os.environ["S3_ENDPOINT_URL"]},
        {"key": "S3_BUCKET", "value": os.environ["S3_BUCKET"]},
        {"key": "S3_REGION", "value": os.environ["S3_REGION"]},
        {"key": "S3_ACCESS_KEY_ID", "value": os.environ["S3_ACCESS_KEY_ID"]},
        {"key": "S3_SECRET_ACCESS_KEY", "value": os.environ["S3_SECRET_ACCESS_KEY"]},
        {"key": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
    ],
}, indent=2))
PY
)

echo "==> POST /v2/serverless"
echo "$PAYLOAD"
echo
resp=$(curl -sS -X POST \
    -H "X-API-Key: $YOTTA_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "https://api.yottalabs.ai/v2/serverless")

echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if d.get('code') == 10000 and d.get('data'):
    e = d['data']
    print(f\"\\n✓ Created endpoint id={e['id']}  name={e['name']}\")
    print(f\"  status={e.get('status')}  serviceMode={e.get('serviceMode')}\")
    print(f\"  perHourPrice={e.get('perHourPrice')}\")
    print(f\"  image={e.get('image')}\")
else:
    print(f'\\nERROR: code={d.get(\"code\")} msg={d.get(\"message\")}')
    print(json.dumps(d, indent=2))"
