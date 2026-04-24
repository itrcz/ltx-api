#!/usr/bin/env bash
# Build + push the LTX worker image to GHCR.
#
# Required env vars (load from .env, do NOT hardcode):
#   HF_TOKEN       — Hugging Face token with read access to Lightricks/LTX-2.3
#   GHCR_USER      — GitHub username (default: itrcz)
#   GHCR_PAT       — GitHub PAT with write:packages scope
# Optional:
#   IMAGE_TAG      — default: v0.1.0
#   HF_REPO_ID     — default: Lightricks/LTX-2.3
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

: "${HF_TOKEN:?HF_TOKEN not set}"
: "${GHCR_PAT:?GHCR_PAT not set}"
GHCR_USER="${GHCR_USER:-itrcz}"
IMAGE_TAG="${IMAGE_TAG:-v0.1.0}"
LTX2_REF="${LTX2_REF:-main}"
IMAGE="ghcr.io/${GHCR_USER}/ltx-worker:${IMAGE_TAG}"

cd "$(dirname "$0")/../worker"

echo "${GHCR_PAT}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

docker buildx inspect ltx-builder >/dev/null 2>&1 || \
  docker buildx create --name ltx-builder --use

docker buildx build \
  --platform linux/amd64 \
  --build-arg LTX2_REF="${LTX2_REF}" \
  --tag "${IMAGE}" \
  --push \
  .

echo "Pushed: ${IMAGE}"
