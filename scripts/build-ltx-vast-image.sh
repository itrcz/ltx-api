#!/usr/bin/env bash
# Build + push the LTX-2.3 worker image for vast.ai serverless.
#
# WHERE TO RUN:
#   - **Local Mac/Linux with Docker Desktop or stock dockerd** (recommended,
#     ~120 GB free disk required). Bring weights into worker/build-artifacts/
#     yourself (scp/rsync from a RunPod pod with the prod volume mounted —
#     see docs/vast-deploy.md).
#   - **RunPod CPU/GPU pod with `/runpod-volume` mounted**: the script will
#     auto-stage from the volume. BUT note: unprivileged RunPod pods can't
#     run docker buildx (no CAP_SYS_ADMIN → bind-mount fails). Use Mac instead.
#
# Required env (load from .env or export inline):
#   GHCR_USER     GitHub username/org (e.g. itrcz). Image will be pushed to
#                 ghcr.io/$GHCR_USER/gr-tv-vst:$IMAGE_TAG.
#   GHCR_PAT      GitHub PAT with write:packages, read:packages
#   IMAGE_TAG     Version tag (e.g. v0.3.0). Bump on every push — vast
#                 caches images by digest, same-tag re-push isn't picked up.
#
# Optional:
#   VOLUME_ROOT   default /runpod-volume (where c25vvptq5f is mounted)
#   IMAGE_NAME    default gr-tv-vst
#   KEEP_ARTIFACTS=1  keep worker/build-artifacts/ after push (default: leave)
set -euo pipefail

: "${GHCR_USER:?GHCR_USER required}"
: "${GHCR_PAT:?GHCR_PAT required}"
: "${IMAGE_TAG:?IMAGE_TAG required (e.g. v0.3.0)}"

VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
IMAGE_NAME="${IMAGE_NAME:-gr-tv-vst}"
IMAGE="ghcr.io/${GHCR_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ARTIFACTS="worker/build-artifacts"
mkdir -p "$ARTIFACTS"

# Volume → build context mapping. Source paths match the CLAUDE.md layout
# (`models/{ltx23,gemma-fp8}/...` with the canonical symlinks under
# `models/{checkpoints,text_encoders,latent_upscale_models,loras/ltxv/ltx2}/`).
declare -a FILES=(
  "$VOLUME_ROOT/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors|$ARTIFACTS/ltx-2.3-22b-dev-fp8.safetensors"
  "$VOLUME_ROOT/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors|$ARTIFACTS/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
  "$VOLUME_ROOT/models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors|$ARTIFACTS/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
  "$VOLUME_ROOT/models/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors|$ARTIFACTS/gemma_3_12B_it_fp8_e4m3fn.safetensors"
)

echo "==> Staging weights into $ARTIFACTS/"
for spec in "${FILES[@]}"; do
    src="${spec%|*}"
    dst="${spec#*|}"
    if [ -e "$dst" ]; then
        echo "  skip $(basename "$dst") (already staged at $(du -h "$dst" | cut -f1))"
        continue
    fi
    if [ ! -e "$src" ]; then
        echo "ERROR: $(basename "$dst") missing in build-artifacts AND volume not mounted." >&2
        echo "       Either run on a pod with $VOLUME_ROOT/models/ available, or" >&2
        echo "       manually copy files into $ARTIFACTS/ first." >&2
        exit 1
    fi
    real="$(readlink -f "$src")"
    echo "  cp $(basename "$real") -> $(basename "$dst")"
    cp "$real" "$dst"
done

echo
echo "==> Staged sizes:"
ls -lh "$ARTIFACTS/"

# Sanity check: gemma must be the _e4m3fn variant, not _scaled. CLAUDE.md
# gotcha #2: the public _scaled variant crashes SigLIP with
# "Promotion for Float8 Types is not supported".
gemma_size=$(stat -c %s "$ARTIFACTS/gemma_3_12B_it_fp8_e4m3fn.safetensors" 2>/dev/null \
             || stat -f %z "$ARTIFACTS/gemma_3_12B_it_fp8_e4m3fn.safetensors")
if [ "$gemma_size" -lt 12000000000 ] || [ "$gemma_size" -gt 14000000000 ]; then
    echo "WARNING: gemma file size $gemma_size B is outside expected 12-14 GB range." >&2
    echo "         If you accidentally copied the _scaled stub, SigLIP will crash." >&2
fi

echo
echo "==> docker login ghcr.io"
echo "$GHCR_PAT" | docker login ghcr.io -u "$GHCR_USER" --password-stdin

echo
echo "==> docker buildx build $IMAGE  (~30-60 min: base pull 10 GB + ~50 GB COPY layer)"
# --push streams layers to ghcr.io as they're produced, freeing local disk
# space progressively — important on Mac where Docker Desktop's disk image
# can fill up. With --load we'd need ~140 GB local; with --push, ~80 GB peak.
docker buildx build \
    --platform linux/amd64 \
    -f worker/Dockerfile.vast \
    -t "$IMAGE" \
    --push \
    .

echo
echo "==> Pushed $IMAGE"

if [ "${KEEP_ARTIFACTS:-0}" != "1" ]; then
    echo "==> Cleaning $ARTIFACTS (set KEEP_ARTIFACTS=1 to keep)"
    rm -rf "$ARTIFACTS"
fi

echo
echo "Next: GHCR_USER=$GHCR_USER IMAGE_TAG=$IMAGE_TAG VAST_API_KEY=... \\"
echo "      scripts/create-vast-template.sh"
