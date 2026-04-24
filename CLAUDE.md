# LTX-API — project context

## What this is
Self-hosted API for **LTX-2.3** video generation (Lightricks, 22B params, image-to-video with synchronized audio). RunPod Serverless worker + local test UI. Eventually a gateway in user's k8s cluster.

## Architecture (current)

```
[browser] → http://localhost:8080 (testui/app.py, FastAPI on Mac)
           → uploads image to Yandex S3 (timenote bucket, ru-central)
           → POSTs to https://api.runpod.ai/v2/<ENDPOINT_ID>/run
           → polls /status/<job_id>
             ↓
[RunPod Serverless] endpoint d7kud62ob6wwtp  (5090 × EU-RO-1)
           → pulls ghcr.io/itrcz/ltx-worker:<tag>  (private, auth via registry creds)
           → mounts network volume c25vvptq5f at /runpod-volume/
           → runs worker/handler.py
             ↓
[handler.py] loads DistilledPipeline or TI2VidTwoStagesHQPipeline
           → generates mp4, uploads to Yandex S3
           → returns presigned URL
```

## Key RunPod infra IDs (already created, don't re-create)

- **Endpoint**: `d7kud62ob6wwtp` (name `ltx`, 5090 EU-RO-1, workersMin=0, workersMax=3 ish, flashboot)
- **Template**: `ybom2lfy44` (`ltx-worker-v0.1.0`, links to image + env vars)
- **Network Volume**: `c25vvptq5f` (`ltx-weights`, 130GB, EU-RO-1) — **MUST stay EU-RO-1** (region-locked)
- **Registry auth**: `cmochn28x00chl10759279la8` (`ghcr-itrcz`) — GHCR pull creds, image is private

**Mount path on Serverless: `/runpod-volume/`** — NOT `/workspace/` (that's for pod usage, different). Paths in handler must use `/runpod-volume/models/...`.

## Models on the volume (don't re-download, they're huge)

- `/runpod-volume/models/ltx23/ltx-2.3-22b-distilled-1.1.safetensors` (43GB) — primary fast model
- `/runpod-volume/models/ltx23/ltx-2.3-22b-dev.safetensors` (43GB) — HQ base
- `/runpod-volume/models/ltx23/ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (7GB) — LoRA for dev path
- `/runpod-volume/models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (1GB) — stage 2
- `/runpod-volume/models/ltx23/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` (1GB) — alt upscaler (not used yet)
- `/runpod-volume/models/gemma/` (23GB) — `unsloth/gemma-3-12b-it` text encoder

## VRAM constraints (learned empirically, 5090 = 32GB)

LTX-2.3 is a 22B model. In fp8_cast quantization the weights alone are ~22GB on GPU. Everything else (activations, latents, VAE intermediate conv3d buffers) fights for the remaining ~10GB.

**Measured peaks at 121 frames (5 sec):**
| Resolution | Peak VRAM | Verdict |
|---|---|---|
| 960×1728 (1080p-ish) | 28.8GB (vast) / ~31GB (RunPod) | RunPod OOMs at the edge |
| 832×1472 (900p)     | ~29-30GB est. | tight, current top preset |
| 768×1344 (720p)     | ~26GB | safe |
| 576×1024 (576p)     | ~22GB | fastest |

**10-second clips (241 frames)** OOM at 960×1728 even on vast. Duration > 5s is risky at high res.

Docker/RunPod overhead ≈ 2GB more than bare vast. That's why 960×1728 worked on vast but OOMs on Serverless.

## Critical pipeline gotchas (all hit once, documented to avoid re-hitting)

1. **`torchaudio` must be installed with `--index-url .../cu128`**, otherwise pip picks up `torchaudio==2.11+cu130` which needs `libcudart.so.13` → `OSError: libcudart.so.13 cannot open`. Always install torch + torchvision + torchaudio together from same cu128 index.

2. **`transformers` must be pinned to 4.57.6** (or close to that in 4.x). transformers 5.x removed `SiglipVisionModel.vision_model` attribute and LTX-2 code accesses it → `AttributeError`.

3. **`torch.inference_mode()` breaks LTX-2 stage 2 upsampler** on longer clips → `RuntimeError: Inference tensors cannot be saved for backward`. Use `torch.no_grad()` instead.

4. **HQ pipeline (`TI2VidTwoStagesHQPipeline`) has `@torch.inference_mode()` decorator** on `__call__` — same bug. Bypass via `cls.__call__ = cls.__call__.__wrapped__` at import time.

5. **CPU offload is MUTUALLY EXCLUSIVE with `QuantizationPolicy.fp8_cast()`** → `ValueError: quantization is not supported with layer streaming`. Pick one.

6. **All `height` and `width` must be multiples of 64** for the two-stage pipeline. Values not divisible by 64 → `ValueError: Resolution (WxH) is not divisible by 64`.

7. **Distilled pipeline's step count is controlled via `stage_1_sigmas` tensor**, NOT `num_inference_steps`. Default = `DISTILLED_SIGMAS` = 8 steps. For other counts use `torch.linspace(1.0, 0.0, n+1)` (quality may be degraded — model was trained for the shipped schedule).

8. **Dev-LoRA pipeline uses `num_inference_steps`** (normal scheduler), NOT `stage_1_sigmas`. Also takes `negative_prompt`, `video_guider_params`, `audio_guider_params`. Get `detect_params(DEV_CKPT)` to obtain sensible guider params.

9. **`first_frame_url` is REQUIRED** — LTX-2 is fundamentally image-to-video; there is no text-to-video path via these pipelines.

10. **RunPod Serverless sets env var `RUNPOD_WEBHOOK_GET_JOB`** automatically on real workers. Local testing without this var → SDK runs in "local mode" looking for `test_input.json` and exits.

11. **`runpod.serverless.progress_update`** may or may not exist in SDK 1.9 — wrap in try/except.

12. **RunPod caches workers by digest**. Pushing a new image with same tag does NOT guarantee existing throttled workers are replaced. **Always bump tag** (`v0.1.0` → `v0.1.1`) AND PATCH template's `imageName`. Throttle state persists ~5-10 min even after.

## How to iterate (build + deploy cycle)

```bash
# 1. Edit worker/handler.py or Dockerfile.
# 2. Build + push (buildx cache makes tiny changes fast):
IMAGE_TAG=v0.1.X docker buildx build \
  --platform linux/amd64 \
  --build-arg LTX2_REF=main \
  --tag ghcr.io/itrcz/ltx-worker:v0.1.X \
  --push /Users/macbook/ltx-api/worker

# 3. Update template to new tag:
curl -X PATCH https://rest.runpod.io/v1/templates/ybom2lfy44 \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"imageName": "ghcr.io/itrcz/ltx-worker:v0.1.X"}'

# 4. Purge stale queue:
curl -X POST https://api.runpod.ai/v2/d7kud62ob6wwtp/purge-queue \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

If workers are all `throttled`: either wait ~5-10 min or bump `workersMin: 1` temporarily to force fresh spawn, then restore `workersMin: 0`.

## Files that matter

- `worker/Dockerfile` — nvidia/cuda:12.8.1-cudnn-devel base, python3.11, torch 2.9.1+cu128, LTX-2 cloned from github, env `MODELS_DIR=/runpod-volume/models`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `worker/handler.py` — RunPod Serverless entrypoint. Supports `model: distilled | dev-lora`, 3 quality presets, validates params, loads pipeline lazily, calls with torch.no_grad().
- `worker/requirements.txt` — runpod, boto3, requests, pillow, pydantic, **transformers==4.57.6**
- `worker/s3_upload.py` — uploads mp4 to S3 and returns presigned URL
- `testui/app.py` — single-file FastAPI on Mac, simple form + localStorage job history + live health banner
- `testui/run.sh` — starts uvicorn on :8080
- `.env` — secrets (HF_TOKEN, GHCR_PAT, RUNPOD_API_KEY, S3_*). GITIGNORED. Also loaded by `scripts/build-worker.sh`.
- `scripts/build-worker.sh` — wrapper that sources .env and runs buildx

## What's NOT yet built (pending tasks)

- Gateway FastAPI + k8s manifests (will live in user's cluster, proxies to RunPod endpoint, bearer auth, rate limiting).

## Operating rules (PLEASE FOLLOW)

- **Don't "fix" by iterating small patches**. Before pushing a rebuild, re-read this file + the relevant pipeline source (LTX-2 has it at `packages/ltx-core/src/...`). Catch the whole class of errors together.
- **Always state the tag bump** (`v0.1.X → v0.1.X+1`) when pushing, and PATCH template in same step.
- **Don't silently change behavior under a same image tag** — RunPod caches, debugging becomes impossible.
- **When OOM happens**, first check if `/runpod-volume/_worker_log_*.txt` exists (startup diag) — read it via SSH on a pod that has the volume attached.
- **When mutating infra** (endpoint, template, volume, pods), confirm with user first unless the action is clearly reversible and low-cost.
- **Secrets previously leaked in chat** (`HF_TOKEN`, `GHCR_PAT`, `RUNPOD_API_KEY`, Yandex S3 keys) — user was warned; assume they'll rotate them post-project.
- **User's endpoint is running, workers cost $ when spawned**. Be thoughtful about leaving pods/workers alive.
