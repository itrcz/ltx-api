#!/usr/bin/env bash
# Create RunPod template + serverless endpoint for ACE-Step worker.
# Run AFTER scripts/build-ace-worker.sh has pushed the image to GHCR.
#
# Required env (load from .env):
#   RUNPOD_API_KEY
#   S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET, S3_ENDPOINT_URL, S3_REGION
# Optional:
#   IMAGE_TAG     — default v0.1.0 (must match what build-ace-worker.sh pushed)
#   IMAGE_NAME    — default ace-worker-comfy
#   GHCR_USER     — default itrcz
#
# Outputs (saved to .env on success):
#   ACE_TEMPLATE_ID
#   ACE_ENDPOINT_ID
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -f "${ROOT}/.env" ]] && { set -a; source "${ROOT}/.env"; set +a; }

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
: "${S3_ACCESS_KEY_ID:?}"; : "${S3_SECRET_ACCESS_KEY:?}"; : "${S3_BUCKET:?}"

GHCR_USER="${GHCR_USER:-itrcz}"
IMAGE_NAME="${IMAGE_NAME:-ace-worker-comfy}"
IMAGE_TAG="${IMAGE_TAG:-v0.1.0}"
IMAGE="ghcr.io/${GHCR_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

# Reuse the same GHCR auth that LTX endpoint uses (private image pulls).
REG_AUTH_ID="cmochn28x00chl10759279la8"

# Volumes — same 3 region-locked LTX volumes, now also populated with ACE weights.
VOL_RO="ry2gwb83q9"
VOL_IS="v53ngmp8uf"
VOL_NO="azs9bp5b96"

API="https://rest.runpod.io/v1"
H_AUTH="Authorization: Bearer ${RUNPOD_API_KEY}"
H_JSON="Content-Type: application/json"

# ---------- 1. Create template ----------
TEMPLATE_PAYLOAD=$(python3 -c "
import json, os
print(json.dumps({
  'name': f'ace-worker-${IMAGE_TAG}',
  'imageName': '${IMAGE}',
  'isServerless': True,
  'containerRegistryAuthId': '${REG_AUTH_ID}',
  'containerDiskInGb': 20,
  'volumeMountPath': '/runpod-volume',
  'ports': ['8888/http', '22/tcp'],
  'startSsh': True,
  'startJupyter': False,
  'env': {
    'S3_ACCESS_KEY_ID':     os.environ['S3_ACCESS_KEY_ID'],
    'S3_SECRET_ACCESS_KEY': os.environ['S3_SECRET_ACCESS_KEY'],
    'S3_BUCKET':            os.environ['S3_BUCKET'],
    'S3_ENDPOINT_URL':      os.environ.get('S3_ENDPOINT_URL', 'https://storage.yandexcloud.net/'),
    'S3_REGION':            os.environ.get('S3_REGION', 'ru-central'),
    'PRESIGN_TTL':          '3600',
    'JOB_TIMEOUT_S':        '1800',
  },
}))
")
echo "[+] creating template..."
TPL_RESP=$(curl -sS -X POST "${API}/templates" -H "${H_AUTH}" -H "${H_JSON}" -d "${TEMPLATE_PAYLOAD}")
TEMPLATE_ID=$(echo "${TPL_RESP}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('id') or '')")
[[ -n "${TEMPLATE_ID}" ]] || { echo "template create failed: ${TPL_RESP}"; exit 1; }
echo "  → ACE_TEMPLATE_ID=${TEMPLATE_ID}"

# ---------- 2. Create endpoint ----------
# Multi-DC layout mirrors prod LTX: one endpoint, 3 region-locked volumes,
# RunPod picks the matching volume per worker spawn location.
ENDPOINT_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
  'name': 'ACE-Step Europe (RO + IS + NO)',
  'templateId': '${TEMPLATE_ID}',
  'gpuTypeIds': [
    'NVIDIA GeForce RTX 4090',     # primary target — fits XL-SFT + LM 4B with offload
    'NVIDIA GeForce RTX 5090',     # fallback if 4090 not available
    'NVIDIA RTX A6000',            # 48GB headroom; pricier but always available
  ],
  'gpuCount': 1,
  'workersMin': 0,
  'workersMax': 3,
  'workersStandby': 0,             # cold start OK; bump to 1 after we see real traffic
  'idleTimeout': 5,
  'flashboot': True,
  'scalerType': 'QUEUE_DELAY',
  'scalerValue': 1,
  'executionTimeoutMs': 900000,
  'networkVolumeIds': ['${VOL_RO}', '${VOL_IS}', '${VOL_NO}'],
  'dataCenterIds':    ['EU-RO-1', 'EUR-IS-1', 'EUR-NO-1'],
}))
")
echo "[+] creating endpoint..."
EP_RESP=$(curl -sS -X POST "${API}/endpoints" -H "${H_AUTH}" -H "${H_JSON}" -d "${ENDPOINT_PAYLOAD}")
ENDPOINT_ID=$(echo "${EP_RESP}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('id') or '')")
[[ -n "${ENDPOINT_ID}" ]] || { echo "endpoint create failed: ${EP_RESP}"; exit 1; }
echo "  → ACE_ENDPOINT_ID=${ENDPOINT_ID}"

# ---------- 3. Persist IDs to .env (idempotent — append/update) ----------
ENV_FILE="${ROOT}/.env"
python3 - "${ENV_FILE}" "${TEMPLATE_ID}" "${ENDPOINT_ID}" <<'PY'
import sys, re, pathlib
env_path, tpl, ep = sys.argv[1], sys.argv[2], sys.argv[3]
text = pathlib.Path(env_path).read_text()
def upsert(content, key, val):
    pat = re.compile(rf'^{key}=.*$', re.MULTILINE)
    line = f'{key}={val}'
    if pat.search(content): return pat.sub(line, content)
    if not content.endswith('\n'): content += '\n'
    return content + line + '\n'
text = upsert(text, 'ACE_TEMPLATE_ID', tpl)
text = upsert(text, 'ACE_ENDPOINT_ID', ep)
pathlib.Path(env_path).write_text(text)
print(f'wrote ACE_TEMPLATE_ID={tpl} and ACE_ENDPOINT_ID={ep} to {env_path}')
PY

echo ""
echo "=== deployed ==="
echo "Template: ${TEMPLATE_ID}"
echo "Endpoint: ${ENDPOINT_ID}"
echo ""
echo "Smoke test (after first cold-boot ~3-5 min while flashboot warms):"
echo "  curl -X POST https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync \\"
echo "    -H \"Authorization: Bearer \$RUNPOD_API_KEY\" -H \"Content-Type: application/json\" \\"
echo "    -d '{\"input\":{\"prompt\":\"russian horror punk, korol i shut style\",\"lyrics\":\"[Verse]\\\\ntest line\\\\n\\\\n[Chorus]\\\\ntest hook\",\"mode\":\"turbo\",\"duration_sec\":30}}'"
