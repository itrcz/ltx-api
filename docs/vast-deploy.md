# Vast.ai deployment

LTX-2.3 worker running on vast.ai serverless. Side-by-side with the RunPod
deploy (`worker/Dockerfile` + `worker/src/handler.py`) — neither replaces
the other.

## Why Vast

- 10–20-minute generations need an **async API with webhook + S3 sidecar**;
  vast doesn't supply this natively but its direct-connect routing has no
  proxy timeout, so we build it ourselves in the PyWorker.
- **No network volumes on vast** — weights are baked into the Docker image.
- Endpoint configuration stays editable in the **vast UI**: we use the
  Template path (NOT the Deployment SDK), so `min_load` / `cold_workers` /
  `max_workers` are knobs anyone can twist without code changes.

## What's in the image

`ghcr.io/<GHCR_USER>/gr-tv-vst:<tag>` is ~63 GB (public). Layers:

| layer | source | size |
|---|---|---|
| `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel` | base | ~10 GB |
| ComfyUI + ComfyUI-LTXVideo + ComfyMath + transformers 4.57.6 + lt.py patch | `worker/Dockerfile.vast` | ~4 GB |
| LTX-2.3 22B dev fp8 | from volume `c25vvptq5f` | 28 GB |
| LTX-2.3 spatial upscaler x2 + distilled LoRA | same | 8 GB |
| Gemma 12B fp8 _e4m3fn (locally quantized) | same | 13 GB |
| handler.py + pyworker.py + start-vast.sh | source | <1 MB |

CLAUDE.md gotcha #2 still applies — the gemma file MUST be the
`_e4m3fn` variant from the production volume. The publicly downloadable
`Comfy-Org/...gemma_3_12B_it_fp8_scaled.safetensors` crashes SigLIP.

## API contract (vast worker)

Single route: `POST <worker_url>/run`. Two modes triggered by the request body.

```jsonc
// Sync — no webhook_url. Holds the connection for the full pipeline.
POST /run
{ "input": { "prompt": "...", "duration_sec": 5, "quality": "hd", ... } }
→ 200 { "video_url": "...", "thumbnail_url": "...",
        "elapsed_sec": 42.1, "width": 1344, "height": 768,
        "num_frames": 121, "duration_sec": 5.04, "fps": 24, "mode": "t2v",
        "quality": "hd" }

// Async — pass webhook_url. Returns immediately, background thread does the work.
POST /run
{ "input": { ... },
  "webhook_url":    "https://api.example.com/ltx-callback",
  "webhook_secret": "shared-secret-≥16-bytes" }    // optional, enables HMAC
→ 200 { "async": true,
        "job_id": "abc123def456ab78",
        "estimated_seconds": 480,
        "result_url": "https://<bucket>/run/result/abc123.../result.json?<presign>",
        "webhook_will_fire": true }

// Webhook delivery (if webhook_url given)
POST <webhook_url>
Headers:
  Content-Type:     application/json
  X-LTX-Signature:  sha256=<hex(hmac_sha256(secret, raw_body))>     # if webhook_secret given
  X-LTX-Job-Id:     abc123def456ab78
  X-LTX-Delivery:   <uuid>
Body: { "job_id": "...", "status": "done"|"failed",
        "result": { "video_url": ..., ... } | null,
        "error":  "..." | null,
        "elapsed_sec": 482.3 }
```

**Request routing**: clients first POST `/route/?endpoint_id=<id>` to
`https://run.vast.ai` with Bearer `VAST_API_KEY` to get a worker URL + JWT,
then POST `/run` to that worker URL with Bearer `<JWT>`. See vast docs.

**Async polling fallback**: when the webhook can't be received (firewall,
serverless client, etc.) the client polls `result_url` directly. 404 means
"still running", 200 + JSON means done. Long presign TTL (24 h default,
`ASYNC_PRESIGN_TTL` env on the template) makes this safe across vast
worker reassignments.

**Input schema** = exactly what `worker/src/handler.py:_validate` accepts:
`prompt`, `negative_prompt`, `quality` (sd/hd/fullhd), `aspect_ratio`
(9:16/16:9), `duration_sec` (1..20), `seed`, `steps` (5..30), frames /
`first_frame_url` / `last_frame_url`, `lora_strength`, `no_tile_vae`.

## Build + deploy (operator)

### 0. One-time setup

- GHCR package `ghcr.io/<GHCR_USER>/gr-tv-vst` auto-creates on first push.
  Make it **public** afterward: github.com/<GHCR_USER>?tab=packages → package
  → Package settings → Change visibility → Public. Vast hosts pull anonymously,
  so public visibility avoids per-host auth setup.
- Set `VAST_API_KEY` in `.env`
- `GHCR_USER` / `GHCR_PAT` already in `.env` (same creds used for the RunPod
  image at `ghcr.io/itrcz/ltx-worker-comfy`). PAT needs `write:packages`.

GHCR public packages have unlimited anonymous pulls — no rate-limit
mitigation needed.

### 1. Stage weights onto the build host

RunPod **unprivileged** pods can't run `docker buildx` (no `CAP_SYS_ADMIN`
→ buildkit's bind-mount syscalls fail with `operation not permitted`).
So the build itself happens on your **local Mac/Linux** with Docker
Desktop, not on the pod. The pod's only job is to ship the 4 weight files.

Provision a temp pod just to access the volume:

| field | value |
|---|---|
| Pod type | GPU (cheap one, e.g. RTX 4090) in **EU-RO-1** |
| Image | `runpod/base:0.6.2-cuda12.4.1` (or any with sshd) |
| Volume | `ry2gwb83q9` mounted at `/runpod-volume` (the production EU-RO-1 volume; has all 4 weights) |
| Disk | 200 GB (one of the files needs to be copied locally inside the pod) |

Stage the files to the pod's local disk + rsync to your laptop:

```bash
# On the pod (one-off setup, then leave it running while rsync grabs):
mkdir -p /workspace/ltx-api/worker/build-artifacts
for spec in \
    "/runpod-volume/models/ltx23/ltx-2.3-22b-dev-fp8.safetensors|ltx-2.3-22b-dev-fp8.safetensors" \
    "/runpod-volume/models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors|ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    "/runpod-volume/models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors|ltx-2.3-22b-distilled-lora-384-1.1.safetensors" \
    "/runpod-volume/models/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors|gemma_3_12B_it_fp8_e4m3fn.safetensors"; do
    src="${spec%|*}"; dst="${spec#*|}"
    cp "$(readlink -f "$src")" "/workspace/ltx-api/worker/build-artifacts/$dst"
done

# On your Mac (4 parallel rsync streams; ~50 GB total):
DST=/Users/<you>/ltx-api/worker/build-artifacts
for f in *.safetensors; do
    rsync -avh --partial --inplace -e "ssh -i .runpod-ssh/key -p <port>" \
        "root@<pod-ip>:/workspace/ltx-api/worker/build-artifacts/$f" "$DST/$f" &
done; wait
```

### 2. Build locally + push

```bash
GHCR_USER=itrcz GHCR_PAT=ghp_... IMAGE_TAG=v0.3.0 \
    ./scripts/build-ltx-vast-image.sh
```

Script flow:
1. Detects weights already in `worker/build-artifacts/` (skips stage)
2. `docker login ghcr.io`
3. `docker buildx build --platform linux/amd64 --push` — streams each layer
   to GHCR as it's built (peak local disk ~80 GB, with `--push` no need to
   keep final 63 GB image around)

Expected wall: 30–60 min on Mac M-series (base pull 10 GB + 50 GB COPY layer
push to GHCR depending on uplink).

After first push, mark the GHCR package **Public**:
github.com/<GHCR_USER>?tab=packages → `gr-tv-vst` → Package settings
→ Change visibility → Public. Vast hosts pull anonymously.

The script verifies the gemma file size to catch the `_scaled` substitution
mistake before push.

### 3. Register the vast template

Locally (or on the same pod):

```bash
DH_USER=jleed IMAGE_TAG=v0.3.0 \
S3_ENDPOINT_URL=... S3_BUCKET=... \
S3_ACCESS_KEY_ID=... S3_SECRET_ACCESS_KEY=... \
VAST_API_KEY=... \
    ./scripts/create-vast-template.sh
```

Output prints `template hash_id=...`. Save that — UI shows it on the
endpoint creation screen.

### 4. Create the Endpoint (vast UI)

Go to <https://cloud.vast.ai/serverless/>. Click **+ New Endpoint**.

- **Name**: `ltx-2.3`
- **Workergroup → Template**: pick the just-registered template (filter by
  the hash)
- **GPU filter (search params)**: `gpu_name=RTX_5090 num_gpus=1 reliability>0.98`
- **gpu_ram**: 24 GB (LTX-2.3 22B fp8 + Gemma fp8 fit comfortably)
- **Autoscaler defaults** (tune later):
  - `min_load`: 0 (scale-to-zero when idle)
  - `cold_workers`: 0 (no idle disk preallocation)
  - `max_workers`: 4
  - `target_util`: 0.85
  - `target_queue_time`: 300 s (tolerate up to 5 min queue before scaling
    out — generations are long, queueing 5 min is cheap vs spinning up a
    new $0.50/h worker that takes 10 min to cold-pull the image)
  - `max_queue_time`: 1800 s (hard reject after 30 min queue)

Click create. Endpoint ID will be visible at the URL (e.g. `/serverless/<id>`)
and via `vastai show endpoints --raw | jq`.

### 5. Smoke test

```bash
# Wait for one worker to spawn (vast UI shows "running" + benchmark passes)
# this takes ~10-15 min on a fresh host (image pull dominates).

VAST_ENDPOINT_ID=<id> VAST_API_KEY=... python3 - <<'PY'
import os, time, requests

ep = os.environ["VAST_ENDPOINT_ID"]
key = os.environ["VAST_API_KEY"]

# 1. /route/ to pick a worker
r = requests.post(
    f"https://run.vast.ai/route/?endpoint_id={ep}",
    headers={"Authorization": f"Bearer {key}"},
    timeout=30,
)
r.raise_for_status()
route = r.json()
print("route:", route)
worker_url = route["url"]
jwt = route["worker_jwt"]   # or whatever key the route resp uses

# 2. POST /run (sync — short clip, ~30 s wall)
t0 = time.time()
r = requests.post(
    f"{worker_url}/run",
    headers={"Authorization": f"Bearer {jwt}"},
    json={"input": {"prompt": "a short clip", "quality": "sd",
                    "duration_sec": 3}},
    timeout=600,
)
print("sync result:", r.status_code, r.json())
print(f"wall = {time.time() - t0:.1f}s")
PY
```

## Updating an existing deploy

For code-only changes (handler.py, pyworker.py, system prompts):

```bash
# In the build pod:
git pull
GHCR_USER=... GHCR_PAT=... IMAGE_TAG=v0.3.1 ./scripts/build-ltx-vast-image.sh
# Then in vast UI: edit the endpoint's workergroup → change image tag →
# rolling worker update kicks in.
```

Or programmatically: `vastai update workergroup <wg_id> --image <new_image>`.

Vast caches images by digest, not tag — pushing the same tag again doesn't
update running workers. **Always bump the tag.**

## Operating gotchas

- **GHCR public packages** have no anonymous pull-rate limit. If you ever
  switch the image to a private GHCR package, add `docker_login_*` fields to
  the template payload (the create-vast-template.sh script doesn't pass them
  by default — uncomment the relevant lines).
- **`vast UI shows endpoint as deployment-managed`** — this is the *opposite*
  problem from grom-art's. Our Template flow → workergroup is **NOT**
  deployment-managed, so the UI lets you edit autoscaler params without
  re-running any script.
- **Cold start ≈ 10–15 min** on a host that doesn't have the image cached
  (63 GB pull). After warm, cold-restart of a stopped worker is ≪1 min.
  Set `cold_workers=1` if you can afford ~$8/day idle (5090 spot rate) to
  guarantee fast first-request.
- **GPU lock serialization**: pyworker.py holds a process-wide threading
  lock for the full pipeline duration so the autoscaler sees real
  back-pressure on the next /run. If you change to true parallelism (e.g.
  multi-GPU node) drop the lock.
- **Workload accounting**: `_workload()` in pyworker.py returns
  `duration × quality_factor × steps/8`. Benchmark = sd × 1 s × 5 steps ≈
  0.625 unit. Real hd × 5 s × 8 steps = 10 units. Vast sees the 16× heavier
  request and scales accordingly. Verify numbers against actual gen-times
  after a week of traffic; recalibrate `QUALITY_W` if scaling feels off.
- **Webhook delivery is best-effort** with 3 retries (`jobs.py:WEBHOOK_RETRIES`).
  The S3 sidecar at `run/result/<job_id>/result.json` is the source of truth.
- **license check**: LTX-2.3 community license + Gemma terms both permit
  redistribution with attribution. Confirm before making the image public
  (current default in `create-vast-template.sh` is `"private": false`).
