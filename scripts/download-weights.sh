#!/usr/bin/env bash
# Download LTX-2.3 weights into /workspace (RunPod network volume).
#
# Idempotent: skips files that are already present at the right size.
# Picks profile via env: PROFILE=hq (default) | distilled | both
#
# Usage on a fresh pod (after sourcing HF_TOKEN):
#   bash download-weights.sh                # = PROFILE=hq
#   PROFILE=both bash download-weights.sh
set -euo pipefail

PROFILE="${PROFILE:-hq}"
ROOT="${ROOT:-/workspace}"
LTX_DIR="$ROOT/models/ltx23"
GEMMA_DIR="$ROOT/models/gemma"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set" >&2
    exit 1
fi
export HF_TOKEN

mkdir -p "$LTX_DIR" "$GEMMA_DIR"

# Files needed per profile, paired as: <hf-filename>:<expected-size-MB>
HQ_FILES=(
    "ltx-2.3-22b-dev.safetensors:43009"
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors:7232"
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors:949"
)
DISTILLED_FILES=(
    "ltx-2.3-22b-distilled-1.1.safetensors:43000"
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors:949"
)

case "$PROFILE" in
    hq)        FILES=("${HQ_FILES[@]}") ;;
    distilled) FILES=("${DISTILLED_FILES[@]}") ;;
    both)      FILES=("${HQ_FILES[@]}" "${DISTILLED_FILES[@]}") ;;
    *) echo "unknown PROFILE=$PROFILE (use hq|distilled|both)" >&2; exit 1 ;;
esac

dl_ltx() {
    local fname="$1" expected_mb="$2"
    local path="$LTX_DIR/$fname"
    if [ -f "$path" ]; then
        local cur_mb=$(( $(stat -c %s "$path") / 1024 / 1024 ))
        local diff=$(( cur_mb > expected_mb ? cur_mb - expected_mb : expected_mb - cur_mb ))
        if [ "$diff" -lt 100 ]; then
            echo "  [skip] $fname (~${cur_mb}MB)"
            return
        fi
    fi
    echo "  [dl]   $fname (~${expected_mb}MB)"
    hf download Lightricks/LTX-2.3 "$fname" --local-dir "$LTX_DIR" >/dev/null
}

echo "=== LTX-2.3 weights (profile=$PROFILE) ==="
# Dedup file list (HQ+distilled share the upsampler)
seen=()
for entry in "${FILES[@]}"; do
    fname="${entry%%:*}"
    if [[ " ${seen[*]} " == *" $fname "* ]]; then continue; fi
    seen+=("$fname")
    expected="${entry##*:}"
    dl_ltx "$fname" "$expected"
done

echo "=== Gemma 3 12B IT (text encoder, ~24GB) ==="
if [ -f "$GEMMA_DIR/model-00005-of-00005.safetensors" ]; then
    echo "  [skip] gemma already present"
else
    echo "  [dl]   unsloth/gemma-3-12b-it"
    hf download unsloth/gemma-3-12b-it --local-dir "$GEMMA_DIR" >/dev/null
fi

echo
echo "=== Done. Disk usage: ==="
du -sh "$LTX_DIR" "$GEMMA_DIR"
df -h "$ROOT" | tail -1
