#!/usr/bin/env bash
# Register a vast.ai template for the LTX-2.3 worker. Idempotent only in
# the sense that vast allows duplicates — re-running creates another row.
# To update an existing template use `vastai update template --hash-id ...`.
#
# The endpoint itself is NOT created here — user creates it in the UI from
# this template so all autoscaler knobs (min_load, cold_workers, ...)
# remain editable. The template just bundles image + env + disk + runtype.
#
# Required env (load from .env):
#   VAST_API_KEY
#   GHCR_USER     GitHub username/org of the registry (e.g. itrcz)
#   IMAGE_TAG     same tag pushed by build-ltx-vast-image.sh
#   S3_ENDPOINT_URL, S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY
#
# Note: ghcr.io public packages have NO anonymous pull-rate limit (unlike
# Docker Hub free), so we don't pass docker_login_* in the template.
#
# Optional:
#   S3_REGION             default: auto
#   TEMPLATE_NAME         default: "LTX-2.3 vast worker"
#   IMAGE_NAME            default: gr-tv-vst
#   DISK_GB               default: 80
#   ASYNC_PRESIGN_TTL     default: 86400 (24 h)
set -euo pipefail

: "${VAST_API_KEY:?VAST_API_KEY required}"
: "${GHCR_USER:?GHCR_USER required}"
: "${IMAGE_TAG:?IMAGE_TAG required}"
: "${S3_ENDPOINT_URL:?S3_ENDPOINT_URL required}"
: "${S3_BUCKET:?S3_BUCKET required}"
: "${S3_ACCESS_KEY_ID:?S3_ACCESS_KEY_ID required}"
: "${S3_SECRET_ACCESS_KEY:?S3_SECRET_ACCESS_KEY required}"

S3_REGION="${S3_REGION:-auto}"
TEMPLATE_NAME="${TEMPLATE_NAME:-LTX-2.3 vast worker}"
IMAGE_NAME="${IMAGE_NAME:-gr-tv-vst}"
IMAGE="ghcr.io/${GHCR_USER}/${IMAGE_NAME}"
DISK_GB="${DISK_GB:-80}"
ASYNC_PRESIGN_TTL="${ASYNC_PRESIGN_TTL:-86400}"

# Vast's `env` field is a shell-style options string: -p maps ports,
# -e injects env vars. The PyWorker reads WORKER_PORT, S3_* and friends
# from the container env.
ENV_STR="-p 3000:3000"
ENV_STR+=" -e WORKER_PORT=3000"
ENV_STR+=" -e ASYNC_PRESIGN_TTL=${ASYNC_PRESIGN_TTL}"
ENV_STR+=" -e S3_ENDPOINT_URL=${S3_ENDPOINT_URL}"
ENV_STR+=" -e S3_REGION=${S3_REGION}"
ENV_STR+=" -e S3_BUCKET=${S3_BUCKET}"
ENV_STR+=" -e S3_ACCESS_KEY_ID=${S3_ACCESS_KEY_ID}"
ENV_STR+=" -e S3_SECRET_ACCESS_KEY=${S3_SECRET_ACCESS_KEY}"
ENV_STR+=" -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

PAYLOAD=$(IMAGE="$IMAGE" IMAGE_TAG="$IMAGE_TAG" \
          TEMPLATE_NAME="$TEMPLATE_NAME" \
          ENV_STR="$ENV_STR" DISK_GB="$DISK_GB" python3 <<'PY'
import json, os
print(json.dumps({
    "name":   os.environ["TEMPLATE_NAME"],
    "image":  os.environ["IMAGE"],
    "tag":    os.environ["IMAGE_TAG"],
    "env":    os.environ["ENV_STR"],
    "runtype": "args",   # honour the Dockerfile's CMD (start-vast.sh)
    "recommended_disk_space": float(os.environ["DISK_GB"]),
    "desc":   "LTX-2.3 22B image-to-video + audio (ComfyUI). PyWorker on :3000. "
              "POST /run with {input: {...LTX schema...}, "
              "[webhook_url], [webhook_secret]}. "
              "See https://github.com/itrcz/ltx-api/blob/main/docs/vast-deploy.md",
    "readme": "## LTX-2.3 vast.ai worker\n\n"
              "Drop-in vast equivalent of the RunPod LTX endpoint.\n\n"
              "- **Sync**: `POST <worker_url>/run` with `{input: {...}}` — holds connection for full run.\n"
              "- **Async**: include `webhook_url` (and optional `webhook_secret`) → returns "
              "`{job_id, result_url, ...}` immediately; HMAC-SHA256-signed POST fires on completion.\n",
    "readme_visible": True,
    "private": False,
}))
PY
)

resp=$(curl -sS -X POST "https://console.vast.ai/api/v0/template/" \
    -H "Authorization: Bearer $VAST_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")

hash_id=$(echo "$resp" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if not d.get('success'):
    sys.stderr.write('vast API error: ' + json.dumps(d) + '\n')
    sys.exit(1)
print(d['template']['hash_id'])
" 2>/dev/null) || { echo "vast response:"; echo "$resp"; exit 1; }

echo "Template registered: hash_id=$hash_id"
echo "                      name=$TEMPLATE_NAME"
echo "                      image=$IMAGE:$IMAGE_TAG"
echo
echo "Next: open https://cloud.vast.ai/serverless/ → New Endpoint →"
echo "      pick template '$TEMPLATE_NAME' (hash $hash_id)"
echo "      Workergroup → search_params: \"gpu_name=RTX_5090 num_gpus=1 reliability>0.98\""
echo "      Autoscaler:  min_load=0  cold_workers=0  max_workers=4"
echo "                   target_util=0.85  target_queue_time=300"
echo
echo "Or via CLI:"
echo "  vastai create endpoint --endpoint-name 'ltx-2.3' --cold-workers 0 --max-workers 4"
echo "  vastai create workergroup --endpoint-name 'ltx-2.3' --template-hash $hash_id \\"
echo "      --search-params 'gpu_name=RTX_5090 num_gpus=1 reliability>0.98' --gpu-ram 24"
