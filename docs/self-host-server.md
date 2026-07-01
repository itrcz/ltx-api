# Self-hosted LTX-2.3 server (bare-metal RTX 5090)

A fourth deployment mode alongside RunPod, vast.ai and Yotta: a **persistent GPU
container with its own HTTP API**, running on a box you own (target: a bare-metal
RTX 5090 VDS). It reuses the exact same `run_pipeline()` as every other mode, so
the pipeline behaves identically — it only swaps the transport layer for a
long-lived FastAPI server with an in-memory queue.

What you get over the serverless modes:

- **No per-second billing / cold-start churn** — one always-warm worker.
- **In-process queue with a hard capacity gate** (`QUEUE_MAX`). The intended
  setting is `QUEUE_MAX=2` (one render on the GPU + one waiting); a 3rd
  concurrent submit is rejected with `429 Retry-After` so a client can fail
  over to another box.
- **Sync and async** submission, **HMAC-signed webhooks**, **Prometheus
  metrics**, result TTL — same machinery as grom-art's `server.py`.

```
client → POST :8000/generate (Bearer API_KEY)
              ├── mode:"sync"  → holds the connection, 200 with result (or 202 on timeout)
              └── mode:"async" → 202 {task_id}; poll GET /result/{id}; webhook on done
                       ↓
            server.py (FastAPI, --workers 1)
            ├── in-memory queue + 1 GPU worker thread (serial)
            ├── QUEUE_MAX capacity gate (429 on overflow)
            └── ComfyUI 127.0.0.1:8188  (run_pipeline → render → S3 → presigned URL)
```

## Files

```
worker/Dockerfile.server   — engine-only image (~10 GB, no baked weights) + FastAPI deps
worker/start-server.sh      — boot: GPU check → R2 weight fetch → ComfyUI → server.py
worker/src/server.py        — the FastAPI server (queue, sync/async, webhooks, metrics, QUEUE_MAX)
scripts/build-server-image.sh — buildx + push to ghcr.io/<user>/ltx-server
```

## 1. Build the image

Engine-only — no weight staging, no RunPod volume. Two options:

**a) Build + push to a registry** (build anywhere with Docker):

```bash
GHCR_USER=itrcz GHCR_PAT=<pat> IMAGE_TAG=v0.1.0 ./scripts/build-server-image.sh
# → ghcr.io/itrcz/ltx-server:v0.1.0
```

**b) Build directly on the GPU host** (no registry needed — what we did on the
5090 box). Copy the `worker/` tree to the host and build there:

```bash
# from your machine (repo root):
tar czf - --exclude='__pycache__' worker | ssh root@<host> 'mkdir -p /root/ltx-api && tar xzf - -C /root/ltx-api'
# on the host:
cd /root/ltx-api && docker build -f worker/Dockerfile.server -t ltx-server:local .
```

The build is ~30 GB (base CUDA image + ComfyUI + pinned LTX/WhatDreamsCost/
ComfyMath nodes + transformers 4.57.6 + kornia 0.8.2). ~10–15 min on a fast host.

Weights (~50 GB) are NOT baked; they download from R2 (`s3.unne.ai`, public) on
first boot into `/opt/models/` and are cached there. Mount a host volume at
`/opt/models` so they survive container restarts (else ~50 GB re-download).

> **Slow first boot?** `start-server.sh` fetches weights with a single `curl`
> stream. On a host whose pipe to R2 is bandwidth-capped (~25–30 MB/s seen on
> the test VDS), 50 GB still takes ~30 min, and a single stream can be slower.
> To speed it up, pre-fetch in parallel **before** first start, then mount the
> same dir — the container verifies sizes and skips the download:
> ```bash
> apt-get install -y aria2
> A="aria2c -x16 -s16 -k1M -c"; B=https://s3.unne.ai
> mkdir -p /opt/ltx-models/ltx23/loras /opt/ltx-models/gemma-fp8
> $A -d /opt/ltx-models/ltx23       -o ltx-2.3-22b-dev-fp8.safetensors            $B/ltx-2.3-22b-dev-fp8.safetensors
> $A -d /opt/ltx-models/ltx23       -o ltx-2.3-spatial-upscaler-x2-1.1.safetensors $B/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
> $A -d /opt/ltx-models/ltx23/loras -o ltx-2.3-22b-distilled-lora-384-1.1.safetensors $B/ltx-2.3-22b-distilled-lora-384-1.1.safetensors
> $A -d /opt/ltx-models/gemma-fp8   -o gemma_3_12B_it_fp8_e4m3fn.safetensors       $B/gemma_3_12B_it_fp8_e4m3fn.safetensors
> ```

## 2. Provision the GPU host

Bare-metal Debian 11 (bullseye) + RTX 5090. **Verified end-to-end on
158.255.7.131** (Debian 11, kernel 5.10, 32 vCPU / 125 GB RAM / 1.8 TB disk,
Secure Boot disabled).

> **Blackwell driver gotcha.** The RTX 5090 (GB202, sm_120) needs an NVIDIA
> driver **≥570**. Debian 11's CUDA-repo `cuda-drivers` metapackage tops out at
> **560** — too old; it will not drive the card. Use NVIDIA's official `.run`
> installer with a 570+ driver and the **open** kernel modules (Blackwell
> requires the open GPU kernel module). Verified with **595.71.05**.

### 2a. Driver (NVIDIA `.run`, open modules, DKMS)

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y linux-headers-$(uname -r) build-essential dkms curl ca-certificates gnupg pciutils

# Blacklist nouveau so it releases the GPU, then reboot.
printf 'blacklist nouveau\noptions nouveau modeset=0\n' > /etc/modprobe.d/blacklist-nouveau.conf
update-initramfs -u
reboot

# After reboot — install the driver (open modules + DKMS so it survives kernel bumps):
cd /root
curl -fSL -o NVIDIA-595.71.05.run \
  https://us.download.nvidia.com/XFree86/Linux-x86_64/595.71.05/NVIDIA-Linux-x86_64-595.71.05.run
chmod +x NVIDIA-595.71.05.run
./NVIDIA-595.71.05.run --silent --dkms --kernel-module-type=open

nvidia-smi   # must show "RTX 5090", Driver Version 595.71.05
```

(595.71.05 was the current long-lived production driver; any 570+ works. Get the
latest with `curl -fsSL https://download.nvidia.com/XFree86/Linux-x86_64/latest.txt`.)

### 2b. Docker + NVIDIA Container Toolkit

```bash
curl -fsSL https://get.docker.com | sh

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker

# GPU must be visible inside a container:
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 3. Run the server

Put the runtime config in an **env-file** (cleaner than many `-e` flags, and
keeps secrets out of `ps`/shell history):

```bash
# /root/ltx-server.env  — NOTE: values must be UNQUOTED. docker --env-file does
# NOT strip quotes the way a shell does, so S3_REGION="ru-central" would arrive
# as the literal  "ru-central"  (with quotes) and boto3 rejects it. No quotes:
cat > /root/ltx-server.env <<'EOF'
QUEUE_MAX=2
API_KEY=<choose-a-strong-token>
S3_ENDPOINT_URL=<your-s3-endpoint>
S3_BUCKET=<bucket>
S3_ACCESS_KEY_ID=<key>
S3_SECRET_ACCESS_KEY=<secret>
S3_REGION=ru-central
EOF
chmod 600 /root/ltx-server.env

docker run -d --name ltx-server --restart unless-stopped \
  --gpus all -p 8000:8000 \
  -v /opt/ltx-models:/opt/models \      # persist weights across restarts
  --env-file /root/ltx-server.env \
  ltx-server:local                      # or ghcr.io/itrcz/ltx-server:<tag>

docker logs -f ltx-server     # watch [weights] (cached/download), then "server ready"
```

> `--env-file` is read at container **create** time — to change any value you
> must `docker rm -f ltx-server` and re-run, not just `docker restart`.

First boot: ~50 GB R2 download (or instant if you pre-fetched per §1) + ComfyUI
startup. `/health` returns `503 {stage}` until ComfyUI answers, then `200`.
Verified on the 5090: a warm `sd`/3 s render is ~43 s end-to-end (incl. S3
upload); the first render after a fresh container adds the one-time model load.

### Required vs optional env

| env | meaning | default |
|---|---|---|
| `QUEUE_MAX` | max held tasks (queued + on GPU); `429` past it. `0` = unlimited | `2` (set in start-server.sh) |
| `API_KEY` | Bearer token guarding `/generate` + `/result`. Unset = open | unset |
| `S3_ENDPOINT_URL` / `S3_BUCKET` / `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` / `S3_REGION` | output store for rendered mp4 (presigned URL returned) | — (required for real renders) |
| `PORT` | listen port | `8000` |
| `SYNC_TIMEOUT_S` | sync wait before falling back to `202` | `600` |
| `QUEUE_TTL_S` / `RESULT_TTL_S` | queued-task + result TTL (sec) | `3600` |
| `JOB_TIMEOUT_S` | per-render watchdog (handler.py) | `1500` |
| `R2_PUBLIC_URL` | weights mirror base | `https://s3.unne.ai` |
| `WITH_TALKVID` | fetch the TalkVid lip-sync LoRA (required for `audio_url`; R2→HF fallback) | `1` |
| `FAKE_PIPE` | stub the pipeline (no GPU/ComfyUI) for local API smoke tests | off |

## 4. API

All paths require `Authorization: Bearer $API_KEY` (except `/health`,
`/healthz`, `/metrics`).

| method | path | notes |
|---|---|---|
| POST | `/generate` | body `{input:{<LTX schema>}, mode:"async"\|"sync", webhook?, webhook_secret?, timeout?}`. async→`202 {task_id, estimated_seconds}`; sync→`200 {result}` or `202` on timeout (keeps running, poll `/result`). `429`+`Retry-After` if `QUEUE_MAX` reached. |
| GET | `/result/{task_id}` | status body; `queued` includes `position`; `404` if unknown/TTL-evicted. |
| GET | `/health` | `503` until ComfyUI warm, then `200` + queue stats. |
| GET | `/metrics` | Prometheus: `ltx_queue_depth`, `ltx_in_flight`, `ltx_ready`, `ltx_tasks_total{status}`, `ltx_gen_seconds{mode}`, `ltx_queue_wait_seconds`, `ltx_webhook_total{outcome}`, `ltx_sync_timeouts_total`, `ltx_queue_full_total`. |

The `input` object is the standard LTX schema (`handler.py:_validate`): `prompt`
or `first_frame_url` (i2v), `quality` (`sd`/`hd`/`fullhd`), `aspect_ratio`
(`9:16`/`16:9`), `duration_sec` (1–20), `steps` (5–30), `seed`, `audio_url`, etc.
The result is `{video_url, thumbnail_url, elapsed_sec, width, height, num_frames,
duration_sec, fps, mode, quality}` — the mp4 is uploaded to your S3 and
`video_url` is a presigned link.

### Smoke test

```bash
# async
curl -s -X POST http://<host>:8000/generate \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"prompt":"a calico cat stretching","quality":"sd","duration_sec":3},"mode":"async"}'
# → {"task_id":"...","status":"queued","estimated_seconds":...}
curl -s http://<host>:8000/result/<task_id> -H "Authorization: Bearer $API_KEY"

# sync (holds the connection; set a generous client timeout)
curl -s --max-time 1800 -X POST http://<host>:8000/generate \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"prompt":"a short clip","quality":"sd","duration_sec":3},"mode":"sync"}'

# QUEUE_MAX=2 in action: fire 3 async submits fast — the 3rd returns 429.
```

### Local API smoke test (no GPU)

```bash
cd worker/src
FAKE_PIPE=1 API_KEY=testkey PORT=8077 python server.py
# stubs run_pipeline; exercises queue/sync/async/webhook/TTL/QUEUE_MAX without a GPU.
```

## Monitoring

**GPU (live, interactive):**

```bash
nvtop                       # live GPU/mem/power TUI
watch -n1 nvidia-smi        # or the plain query
```

> On Debian 11 with a `.run`-installed driver, `apt install nvtop` tries to pull
> the `nvidia-installer-cleanup` + `nvidia-alternative`/`glx-alternative` stack,
> whose postinst fails in noninteractive mode (it wants to prompt about removing
> your `.run` driver) and which would also divert GLX libs. Avoid it: purge the
> blocker and satisfy nvtop's NVML dep with an `equivs` stub (the `.run` driver
> already ships `libnvidia-ml.so.1`):
> ```bash
> dpkg --purge --force-all nvidia-installer-cleanup; dpkg --configure -a
> apt-get install -y equivs
> printf 'Package: libnvidia-ml1\nVersion: 595.71.05\nProvides: libnvidia-ml1\nDescription: NVML from the .run driver\n' > /tmp/ml.ctl
> (cd /tmp && equivs-build ml.ctl && dpkg -i libnvidia-ml1_*.deb)
> apt-get install -y nvtop
> ```

**Server metrics (Prometheus):** `GET /metrics` (no auth) exposes the `ltx_*`
series — `ltx_ready`, `ltx_queue_depth`, `ltx_in_flight`, `ltx_tasks_total{status}`,
`ltx_gen_seconds{mode}`, `ltx_queue_wait_seconds`, `ltx_queue_full_total`,
`ltx_webhook_total{outcome}`, `ltx_sync_timeouts_total`. Point a Prometheus at it:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: ltx-server
    static_configs:
      - targets: ['<host>:8000']
```

Useful starting panels: `ltx_in_flight` + `ltx_queue_depth` (load),
`rate(ltx_tasks_total{status="done"}[5m])` (throughput),
`histogram_quantile(0.9, rate(ltx_gen_seconds_bucket[15m]))` (p90 render time),
`rate(ltx_queue_full_total[5m])` (how often you're shedding with 429 → add a box).

## Operator notes

- **`--workers 1` is mandatory.** The queue + result store are in-process Python
  objects; >1 uvicorn worker = multiple disjoint queues + multiple ComfyUI users.
- **Serial GPU.** One render at a time; `QUEUE_MAX` is the only back-pressure
  knob. Don't raise it expecting parallelism — raise it to allow deeper queuing.
- **Webhooks** are HMAC-signed with `webhook_secret` (header `X-LTX-Signature:
  sha256=...`, same scheme as the vast PyWorker via `jobs.deliver_webhook`) and
  retried; the S3 result is still the durable record.
- **Restart picks up cached weights** from `/opt/models` instantly; only a fresh
  host re-downloads from R2.
- **Changing env** needs `docker rm -f` + re-run (env-file is read at create),
  not `docker restart`.
