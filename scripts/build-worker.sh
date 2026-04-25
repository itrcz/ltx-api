#!/usr/bin/env bash
# Build + push the LTX ComfyUI serverless worker image to GHCR.
#
# Required env (load from .env, never hardcode secrets):
#   GHCR_USER      — GitHub user (default: itrcz)
#   GHCR_PAT       — GitHub PAT with write:packages scope
# Optional:
#   IMAGE_TAG      — default: v0.2.0 (bump on every push; RunPod caches by digest)
#   IMAGE_NAME     — default: ltx-worker-comfy
#   COMFYUI_REF    — default: master (pin to a commit SHA for reproducibility)
#   LTX_NODES_REF  — default: master
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

: "${GHCR_PAT:?GHCR_PAT not set}"
GHCR_USER="${GHCR_USER:-itrcz}"
IMAGE_TAG="${IMAGE_TAG:-v0.2.0}"
IMAGE_NAME="${IMAGE_NAME:-ltx-worker-comfy}"
COMFYUI_REF="${COMFYUI_REF:-master}"
LTX_NODES_REF="${LTX_NODES_REF:-master}"
IMAGE="ghcr.io/${GHCR_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

cd "${ROOT}/worker"

echo "${GHCR_PAT}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

docker buildx inspect ltx-builder >/dev/null 2>&1 || \
  docker buildx create --name ltx-builder --use

docker buildx build \
  --platform linux/amd64 \
  --build-arg COMFYUI_REF="${COMFYUI_REF}" \
  --build-arg LTX_NODES_REF="${LTX_NODES_REF}" \
  --tag "${IMAGE}" \
  --push \
  .

echo ""
echo "Pushed: ${IMAGE}"
echo ""
echo "Next:"
echo "  1. PATCH RunPod template with the new imageName:"
echo "     curl -X PATCH https://rest.runpod.io/v1/templates/ybom2lfy44 \\"
echo "       -H \"Authorization: Bearer \$RUNPOD_API_KEY\" \\"
echo "       -H \"Content-Type: application/json\" \\"
echo "       -d '{\"imageName\": \"${IMAGE}\"}'"
echo ""
echo "  2. Purge stale queue:"
echo "     curl -X POST https://api.runpod.ai/v2/d7kud62ob6wwtp/purge-queue \\"
echo "       -H \"Authorization: Bearer \$RUNPOD_API_KEY\""
