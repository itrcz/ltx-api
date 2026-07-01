#!/usr/bin/env bash
# Build + push the LTX-2.3 *universal self-hosted* server image (Dockerfile.server).
#
# Engine-only (~10 GB, NO weights baked) — weights are fetched at boot from R2
# (s3.unne.ai) by start-server.sh into /opt/models/. So unlike the vast build,
# this needs no weight staging and no RunPod volume; build it anywhere with
# Docker + buildx.
#
# Required env (load from .env or export inline):
#   GHCR_USER   GitHub username/org (e.g. itrcz). Pushes to
#               ghcr.io/$GHCR_USER/$IMAGE_NAME:$IMAGE_TAG.
#   GHCR_PAT    GitHub PAT with write:packages
#   IMAGE_TAG   Version tag (e.g. v0.1.0). Bump on every push.
#
# Optional:
#   IMAGE_NAME  default ltx-server
#   COMFYUI_REF / LTX_NODES_REF / COMFY_MATH_REF  pin upstream refs (default master/main)
set -euo pipefail

: "${GHCR_USER:?GHCR_USER required}"
: "${GHCR_PAT:?GHCR_PAT required}"
: "${IMAGE_TAG:?IMAGE_TAG required (e.g. v0.1.0)}"

IMAGE_NAME="${IMAGE_NAME:-ltx-server}"
IMAGE="ghcr.io/${GHCR_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> docker login ghcr.io"
echo "$GHCR_PAT" | docker login ghcr.io -u "$GHCR_USER" --password-stdin

echo
echo "==> docker buildx build $IMAGE  (~15-25 min: base pull 10 GB + comfy/node installs)"
docker buildx build \
    --platform linux/amd64 \
    -f worker/Dockerfile.server \
    --build-arg COMFYUI_REF="${COMFYUI_REF:-master}" \
    --build-arg LTX_NODES_REF="${LTX_NODES_REF:-master}" \
    --build-arg COMFY_MATH_REF="${COMFY_MATH_REF:-main}" \
    -t "$IMAGE" \
    --push \
    .

echo
echo "==> Pushed $IMAGE"
echo
echo "Run it on your GPU box (see docs/self-host-server.md):"
echo "  docker run -d --gpus all -p 8000:8000 \\"
echo "    -v /opt/ltx-models:/opt/models \\"
echo "    -e QUEUE_MAX=2 -e API_KEY=... \\"
echo "    -e S3_ENDPOINT_URL=... -e S3_BUCKET=... \\"
echo "    -e S3_ACCESS_KEY_ID=... -e S3_SECRET_ACCESS_KEY=... \\"
echo "    $IMAGE"
