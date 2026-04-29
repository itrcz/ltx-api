#!/usr/bin/env bash
# Build + push the ACE-Step ComfyUI serverless worker image to GHCR.
#
# Required env (load from .env, never hardcode secrets):
#   GHCR_USER      — GitHub user (default: itrcz)
#   GHCR_PAT       — GitHub PAT with write:packages scope
# Optional:
#   IMAGE_TAG      — default: v0.1.0 (bump on every push; RunPod caches by digest)
#   IMAGE_NAME     — default: ace-worker-comfy
#   COMFYUI_REF    — default: master (pin to a commit SHA for reproducibility)
#   ACE_NODES_REF  — default: main
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

_cli_image_tag="${IMAGE_TAG:-}"
_cli_image_name="${IMAGE_NAME:-}"
_cli_comfyui_ref="${COMFYUI_REF:-}"
_cli_ace_nodes_ref="${ACE_NODES_REF:-}"
_cli_ghcr_user="${GHCR_USER:-}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

[[ -n "$_cli_image_tag"      ]] && IMAGE_TAG="$_cli_image_tag"
[[ -n "$_cli_image_name"     ]] && IMAGE_NAME="$_cli_image_name"
[[ -n "$_cli_comfyui_ref"    ]] && COMFYUI_REF="$_cli_comfyui_ref"
[[ -n "$_cli_ace_nodes_ref"  ]] && ACE_NODES_REF="$_cli_ace_nodes_ref"
[[ -n "$_cli_ghcr_user"      ]] && GHCR_USER="$_cli_ghcr_user"

: "${GHCR_PAT:?GHCR_PAT not set}"
GHCR_USER="${GHCR_USER:-itrcz}"
IMAGE_TAG="${IMAGE_TAG:-v0.1.0}"
IMAGE_NAME="${IMAGE_NAME:-ace-worker-comfy}"
COMFYUI_REF="${COMFYUI_REF:-master}"
ACE_NODES_REF="${ACE_NODES_REF:-main}"
IMAGE="ghcr.io/${GHCR_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

cd "${ROOT}/worker-ace"

echo "${GHCR_PAT}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

docker buildx inspect ace-builder >/dev/null 2>&1 || \
  docker buildx create --name ace-builder --use

docker buildx build \
  --platform linux/amd64 \
  --build-arg COMFYUI_REF="${COMFYUI_REF}" \
  --build-arg ACE_NODES_REF="${ACE_NODES_REF}" \
  --tag "${IMAGE}" \
  --push \
  .

echo ""
echo "Pushed: ${IMAGE}"
echo ""
echo "Next steps:"
echo "  1. PATCH the ACE-Step template's imageName (template id from \$ACE_TEMPLATE_ID env):"
echo "     curl -X PATCH https://rest.runpod.io/v1/templates/\$ACE_TEMPLATE_ID \\"
echo "       -H \"Authorization: Bearer \$RUNPOD_API_KEY\" \\"
echo "       -H \"Content-Type: application/json\" \\"
echo "       -d '{\"imageName\": \"${IMAGE}\"}'"
echo ""
echo "  2. Force fresh worker spawn (kill cached weights / flashboot snapshot):"
echo "     curl -X PATCH https://rest.runpod.io/v1/endpoints/\$ACE_ENDPOINT_ID \\"
echo "       -H \"Authorization: Bearer \$RUNPOD_API_KEY\" -H \"Content-Type: application/json\" \\"
echo "       -d '{\"workersStandby\": 0}'"
