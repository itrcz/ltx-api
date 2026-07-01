# LTX-API — project context

## What this is
Self-hosted Serverless API for **LTX-2.3** (Lightricks, 22B params,
image-to-video with synchronized audio). One worker image runs ComfyUI +
ComfyUI-LTXVideo on a RunPod Serverless endpoint, sharing weights from a
per-region network volume. Eventually a thin FastAPI gateway will sit in front
in the user's k8s cluster.

## Architecture

```
[client] → POST https://api.runpod.ai/v2/d7kud62ob6wwtp/run
                      │
                      ▼
[Serverless endpoint d7kud62ob6wwtp]  workersMin=0, workersMax=3, flashboot
                      │  pulls ghcr.io/itrcz/ltx-worker-comfy:<tag>
                      │  spawns one worker in EU-RO-1 / EUR-IS-1 / EUR-NO-1
                      │  mounts the matching network volume at /runpod-volume/
                      ▼
[worker = ComfyUI + handler.py]
   • workflow_builder.build(...) → ComfyUI prompt JSON
   • POST /prompt → poll /history
   • encode mp4 + audio
   • upload to Yandex S3, return presigned URL
```

## RunPod infra (already provisioned, do NOT re-create)

| Resource         | ID                      | Notes                                                         |
|------------------|-------------------------|---------------------------------------------------------------|
| Endpoint         | `d7kud62ob6wwtp`        | name `Serverless Europe (RO + IS + NO)`, 5090, flashboot      |
| Template         | `ybom2lfy44`            | links to current `imageName`. PATCH it on every release.      |
| Registry auth    | `cmochn28x00chl10759279la8` | `ghcr-itrcz`, GHCR pull creds, image is private          |
| Volume EU-RO-1   | `ry2gwb83q9` (60 GB)    | `ltx-weights-eu-ro-1`, populated for prod                     |
| Volume EUR-IS-1  | `v53ngmp8uf` (60 GB)    | `ltx-weights-eur-is-1`, populated for prod                    |
| Volume EUR-NO-1  | `azs9bp5b96` (60 GB)    | `ltx-weights-eur-no-1`, populated for prod                    |
| Volume EU-RO-1†  | `c25vvptq5f` (130 GB)   | original prod weights — source of truth for `_e4m3fn` gemma   |

† Not mounted in the live endpoint anymore. Kept around as the source for
`scripts/migrate-gemma-from-prod.sh` and any future per-region rebuilds.
Region locked — RunPod volumes are not portable across data centers.

**Mount path on Serverless: `/runpod-volume/`** — NOT `/workspace/` (that's
the pod-mode default). Handler paths are `/runpod-volume/models/...`.

## Volume layout (per region, ~49 GB used of 60 GB)

```
/runpod-volume/models/
  ltx23/
    ltx-2.3-22b-dev-fp8.safetensors                       28 GB  primary
    ltx-2.3-spatial-upscaler-x2-1.1.safetensors            1 GB  stage-2
    loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors   7 GB  distill-style
  gemma-fp8/
    gemma_3_12B_it_fp8_e4m3fn.safetensors                 13 GB  text encoder
  checkpoints/, text_encoders/, latent_upscale_models/, loras/ltxv/ltx2/
                                                                  symlinks into ltx23/ + gemma-fp8/
```

`extra_model_paths.yaml` only maps top-level dirs (`checkpoints/`,
`text_encoders/`, `latent_upscale_models/`, `loras/`), so `start.sh` also
re-links `gemma_*.safetensors` into `/comfyui/models/text_encoders/` at boot.

## Live pipeline gotchas

1. **`transformers` pinned to `4.57.6`**. transformers 5.x removes
   `SiglipVisionModel.vision_model` and `Gemma3ForConditionalGeneration.multi_modal_projector`,
   both of which ComfyUI-LTXVideo accesses directly.

2. **Gemma file naming matters.** The workflow references
   `gemma_3_12B_it_fp8_e4m3fn.safetensors`, which is a **locally quantized**
   file living only on `c25vvptq5f`. The publicly available
   `Comfy-Org/ltx-2/...gemma_3_12B_it_fp8_scaled.safetensors` has different
   per-layer dtypes — substituting it (even via symlink rename) crashes the
   SigLIP vision tower with `Promotion for Float8 Types is not supported,
   attempted to promote BFloat16 and Float8_e4m3fn`. Always run
   `scripts/migrate-gemma-from-prod.sh` after `setup-volume.sh` for any
   newly provisioned volume.

3. **Frame count must be `8k + 1`** (one reference + k temporal latent
   blocks). `_num_frames(duration_sec)` rounds to the nearest valid value.
   Old behaviour was floor → user-visible "video is 1 second short" bug.

4. **Distilled LoRA was trained for the canonical 8-step `DISTILLED_SIGMAS`
   schedule.** Non-8 step counts use a linear `1.0→0.0` schedule, which
   works (5..30 supported) but quality is off-distribution.

5. **`runpod.serverless.progress_update`** may be missing in SDK 1.9 — wrap
   in try/except.

5b. **Multiple-Subject-Reference (`reference_image_urls`) is a wholly separate
   graph** (`_build_msr` in `workflow_builder.py`), like lipsync — not a
   template patch. `LiconMSR` (ComfyUI-Licon-MSR custom node) only exposes 4
   numbered subject slots + 1 required background slot; `reference_image_urls`
   is capped at 1-4 and the background defaults to `first_frame_url`, else the
   first reference image. The MSR IC-LoRA (`MSR_LORA`) stacks on the distilled
   LoRA the same way `TALKVID_LORA` does, and was validated against the same
   canonical 8-step `DISTILLED_SIGMAS_8` schedule. The two other nodes it needs
   (`LTXICLoRALoaderModelOnly`, `LTXAddVideoICLoRAGuide`) already ship in the
   pinned `ComfyUI-LTXVideo` — no `LTX_NODES_REF` bump was needed to add this.

6. **flashboot caches the worker process state**, including model weights in
   memory. Editing files on the volume does NOT take effect for warm
   workers — bump the image tag (or set `workersStandby=0`, then back to N)
   to force fresh spawns.

7. **RunPod caches workers by image digest.** Pushing a new image with the
   same tag does NOT replace existing workers. **Always bump the tag** AND
   PATCH the template's `imageName`.

## Weights mirror on Cloudflare R2 (`s3.unne.ai`)

All six weight files are mirrored to Cloudflare R2 bucket `unne`, served
**publicly** via custom domain `s3.unne.ai`. This is the source of truth for
serverless workers that don't have access to the RunPod network volume
(Vast.ai, Yotta, GitHub Actions builders, self-host server, etc.).

```
https://s3.unne.ai/ltx-2.3-22b-dev-fp8.safetensors                      29,145,431,166 B (27 GB)
https://s3.unne.ai/ltx-2.3-spatial-upscaler-x2-1.1.safetensors             995,743,560 B (1 GB)
https://s3.unne.ai/ltx-2.3-22b-distilled-lora-384-1.1.safetensors        7,605,507,256 B (7 GB)
https://s3.unne.ai/gemma_3_12B_it_fp8_e4m3fn.safetensors                13,210,008,986 B (13 GB)
https://s3.unne.ai/ltx-2.3-id-lora-talkvid-3k.safetensors                1,157,884,304 B (1 GB)  TalkVid lip-sync LoRA (audio_url path)
https://s3.unne.ai/ltx-2.3-licon-msr-v1.safetensors                        654,443,424 B (624 MiB)  Multiple-Subject-Reference IC-LoRA (reference_image_urls path). Source: LiconStudio/LTX-2.3-Multiple-Subject-Reference (Apache-2.0) — keep attribution alongside the mirrored file.
```

R2 advantages:
- **Free egress** — workers download without bandwidth charges (vs Yandex S3
  which has egress fees).
- Public bucket → no creds needed in worker env, simple `curl https://...`.
- Fast: Cloudflare's network typically delivers 30-100 MB/s to most cloud
  regions.

The R2 API creds (S3-compatible, for writes/re-uploads) live in `.env` as
`R2_ENDPOINT_URL` / `R2_BUCKET=unne` / `R2_ACCESS_KEY_ID` /
`R2_SECRET_ACCESS_KEY`. Don't commit them.

**Worker boot pattern** (e.g. vast `start-vast.sh`, Yotta `start-yotta.sh`):
download each file into `/opt/models/...` with `curl -fL --retry 10 --continue-at -`,
verifying byte size against the expected value above. Cached after first boot
on the instance's persistent disk.

**To refresh R2** (after retraining or weight bumps): re-upload via boto3
from any host that has the source files (the prod RunPod volume, or one of
the regional volumes after `migrate-gemma-from-prod.sh`). Public URLs do not
change.

## Volume provisioning (per region)

Two-step, both required:

```bash
# 1. Public files (~50 GB) + ComfyUI symlink farm.
REGION=EUR-IS-1 ./scripts/setup-volume.sh
# 2. Replace the public _scaled gemma stub with the byte-exact _e4m3fn from c25vvptq5f.
REGION=EUR-IS-1 VOLUME_ID=<id-from-step-1> ./scripts/migrate-gemma-from-prod.sh
```

`setup-volume.sh` defaults to `SIZE_GB=60`. 50 GB is too tight (HuggingFace
double-buffers a 13 GB partial during the gemma download → ~59 GB peak).
RunPod can grow volumes (`PATCH /networkvolumes/{id} {"size": N}`) but
cannot shrink them.

EUR-IS-2 is **not supported** for network volumes — only EUR-IS-1 / EUR-IS-3
in Iceland.

## Build + deploy cycle

```bash
# 1. Edit worker/* — handler.py, workflow_builder.py, system_prompts/, etc.

# 2. Build + push (buildx layer cache makes incremental changes fast):
IMAGE_TAG=v0.2.X ./scripts/build-worker.sh

# 3. Point the template at the new tag:
curl -X PATCH https://rest.runpod.io/v1/templates/ybom2lfy44 \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"imageName": "ghcr.io/itrcz/ltx-worker-comfy:v0.2.X"}'

# 4. Force fresh worker spawn (kill cached weights / flashboot snapshot):
curl -X PATCH https://rest.runpod.io/v1/endpoints/d7kud62ob6wwtp \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"workersStandby": 0}'

# 5. Smoke test:
curl -X POST https://api.runpod.ai/v2/d7kud62ob6wwtp/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"prompt":"a short clip","quality":"sd","duration_sec":3}}'
```

## Files that matter

```
worker/
  Dockerfile                — RunPod: ComfyUI + custom nodes, pin transformers 4.57.6
  Dockerfile.vast           — Vast: same recipe + baked weights + PyWorker entry
  Dockerfile.server         — Self-host: engine-only (no baked weights) + FastAPI deps, server.py entry
  start.sh                  — RunPod: sshd (optional), gemma symlink fix, launch comfy + handler
  start-vast.sh             — Vast: GPU check, launch comfy in bg, exec pyworker
  start-server.sh           — Self-host: GPU check, R2 weight fetch, launch comfy in bg, exec server.py (QUEUE_MAX=2)
  extra_model_paths.yaml    — RunPod: ComfyUI → /runpod-volume/models/*
  extra_model_paths.vast.yaml — Vast: ComfyUI → /opt/models/*
  system_prompts/           — gemma_i2v / gemma_t2v (used by LTXVGemmaEnhancePrompt)
  src/
    handler.py              — RunPod entrypoint + shared `run_pipeline()` (all other modes call it)
    pyworker.py             — Vast: vastai.Worker + WorkerConfig + /run handler with async webhook
    yotta_worker.py         — Yotta: FastAPI /run, GPU lock, sync only
    server.py               — Self-host: persistent FastAPI server — in-memory queue, sync/async,
                              HMAC webhooks, Prometheus /metrics, QUEUE_MAX gate. Reuses run_pipeline.
    jobs.py                 — JobState + HMAC-signed webhook delivery + in-memory LRU store
    workflow_builder.py     — builds the ComfyUI prompt JSON from a typed input
    workflow_template_api.json — hand-authored ComfyUI workflow (the live one)
    s3_upload.py            — Yandex S3 upload + presign
scripts/
  build-worker.sh           — RunPod: buildx + push to ghcr.io/itrcz/ltx-worker-comfy
  build-ltx-vast-image.sh   — Vast: stage weights from c25vvptq5f, buildx + push to docker.io
  build-server-image.sh     — Self-host: engine-only buildx + push to ghcr.io/itrcz/ltx-server
  create-vast-template.sh   — Vast: register template via REST; endpoint is created in UI
  setup-volume.sh           — provision a fresh region volume from public sources
  migrate-gemma-from-prod.sh — copy locally-quantized gemma _e4m3fn from c25vvptq5f
docs/
  api.md                    — public-facing API reference for the RunPod endpoint
  server-api.md             — public-facing API reference for the self-host server (hand to integrators)
  self-host-server.md       — Self-host operator guide (build image → provision 5090 → docker run)
  network-volume-setup.md   — operator guide for the two volume scripts
  vast-deploy.md            — Vast operator guide (build pod → push → template → endpoint)
.env                        — secrets (HF_TOKEN, GHCR_PAT, RUNPOD_API_KEY, VAST_API_KEY,
                              DH_USER, DH_TOKEN, S3_*); gitignored
```

## Vast.ai deployment (parallel to RunPod)

Same code, packaged for vast.ai serverless via a Template + Workergroup +
Endpoint (NOT the Deployment SDK — that pattern locks the endpoint in the
UI; see grom-art's CLAUDE.md for the lessons-learned). Endpoint stays
editable in the vast UI; `min_load` / `cold_workers` / `max_workers` are
knobs anyone can turn.

```
client → POST https://run.vast.ai/route/?endpoint_id=<id> (Bearer VAST_API_KEY)
         ← { worker_url, worker_jwt }
       → POST <worker_url>/run (Bearer <worker_jwt>)
                                ↓
                  PyWorker on :3000 (pyworker.py)
                  ├── sync mode (no webhook_url in body) — holds connection
                  ├── async mode (webhook_url present) — 200 immediately,
                  │   bg thread runs run_pipeline → result.json sidecar to S3
                  │   → HMAC-SHA256 POST to webhook_url
                  └── ComfyUI 127.0.0.1:8188 (started by start-vast.sh)
```

**Image** `ghcr.io/<GHCR_USER>/gr-tv-vst:<tag>` ≈ 63 GB (public).
Weights baked from `c25vvptq5f`; built on a RunPod CPU pod (volume not
portable). Same gemma `_e4m3fn` quirk as RunPod (gotcha #2 still bites).

**API**: single route `POST /run`. Body either
`{input: {LTX schema}}` for sync (held connection 10–20 min, no proxy
timeouts on the direct connection) OR
`{input, webhook_url, [webhook_secret]}` for async (202 + presigned
`result_url` of the S3 sidecar; webhook fires on completion).
See `docs/vast-deploy.md` for the full operator flow + smoke test.

**Operator gotchas:**
- **Always bump IMAGE_TAG** — vast caches by digest, same-tag re-push is
  invisible (same rule as RunPod).
- **63 GB image cold-pull ≈ 10–15 min** on a fresh host. Use `cold_workers=1`
  (≈$8/day idle on 5090) if first-request latency matters.
- **Workload calculator** in `pyworker.py:_workload` is calibrated against
  the sd × 1 s × 5-step benchmark. Real hd×5s×8 = ~10 units, fullhd×20s = ~80.
  Recalibrate `QUALITY_W` after live data.
- **GPU lock**: pyworker.py serializes via `threading.Lock` to make autoscaler
  see real back-pressure. Don't remove unless you switch to multi-GPU.
- **GHCR public packages** — no anonymous pull-rate limit (unlike Docker Hub).
- **License**: LTX-2.3 community + Gemma terms allow redistribution with
  attribution. `create-vast-template.sh` defaults `"private": false`.

## Self-hosted server (own GPU, non-serverless)

Fourth deployment mode, for a box you own (target: a bare-metal RTX 5090 VDS).
A **persistent FastAPI server** (`worker/src/server.py`) instead of a serverless
worker — modeled on grom-art's `server.py`. Reuses the same `run_pipeline()` as
every other mode, so the pipeline is identical; only the transport differs.

```
client → POST https://<host>:8000/generate (Bearer API_KEY)
         ├── mode:"sync"  — holds the connection, 200 with result (202 on timeout)
         └── mode:"async" — 202 {task_id}; poll GET /result/{id}; HMAC webhook on done
                  ↓
         server.py (FastAPI, --workers 1)
         ├── in-memory queue + 1 GPU worker thread (serial)
         ├── QUEUE_MAX gate — 429 + Retry-After on overflow (default 2: 1 running + 1 queued)
         └── ComfyUI 127.0.0.1:8188 (run_pipeline → render → S3 → presigned URL)
```

- **Image** `Dockerfile.server`: engine-only (~10 GB, NO baked weights). Weights
  fetch from R2 (`s3.unne.ai`) at boot into `/opt/models/` (mount a host volume
  there to cache across restarts). Build anywhere: `GHCR_USER=.. GHCR_PAT=..
  IMAGE_TAG=.. scripts/build-server-image.sh`.
- **Run**: `docker run -d --gpus all -p 8000:8000 -v /opt/ltx-models:/opt/models
  -e QUEUE_MAX=2 -e API_KEY=.. -e S3_ENDPOINT_URL/-BUCKET/-ACCESS_KEY_ID/-SECRET_ACCESS_KEY=..
  ghcr.io/itrcz/ltx-server:<tag>`. Full runbook in `docs/self-host-server.md`.
- **API**: `POST /generate` (async|sync, optional `webhook`+`webhook_secret`),
  `GET /result/{id}`, `GET /health`, `GET /metrics` (prom `ltx_*`). Auth = your
  own `API_KEY` bearer. Hand `docs/server-api.md` to integrators — it is the
  standalone, infra-free API reference for this mode.

**Operator gotchas:**
- **`--workers 1` is mandatory** — queue + result store are in-process Python
  objects; >1 uvicorn worker = multiple disjoint queues + multiple ComfyUI users.
- **Serial GPU.** One render at a time; `QUEUE_MAX` is the only back-pressure
  knob — raise it to allow deeper queuing, not parallelism. `0` = unlimited.
- **Bump IMAGE_TAG** on every push (same digest-cache rule as the other modes).
- **S3 env required** for real renders (output mp4 lands there, presigned URL
  returned). Without it the pipeline render step fails.
- **Blackwell (5090) driver:** Debian 11's CUDA-repo `cuda-drivers` tops out at
  560 — too old for sm_120. Use the NVIDIA `.run` installer with a 570+ driver
  and `--kernel-module-type=open` (verified 595.71.05). See `docs/self-host-server.md`.
- **`-v <host>:/opt/models` masks the baked symlink farm.** The Dockerfile bakes
  `checkpoints/`, `loras/ltxv/ltx2/`, `text_encoders/`, `latent_upscale_models/`
  symlinks under /opt/models, but a volume mount hides them → ComfyUI loaders see
  empty lists (`ckpt_name ... not in []`). `start-server.sh` rebuilds the farm at
  boot (after the mount + weights are present). Vast didn't hit this — no mount.
- **`docker --env-file` keeps quotes.** Unlike a shell, it does NOT strip quotes
  from `KEY="value"`. Use unquoted values (e.g. `S3_REGION=ru-central`, not
  `"ru-central"`) or boto3 rejects the region format. Verified end-to-end on a
  bare-metal 5090 (158.255.7.131): sd/3s ≈ 43 s warm, QUEUE_MAX=2 sheds the 3rd
  with 429.

## Operating rules

- **Don't iterate-and-pray.** Re-read this file + the relevant pipeline source
  before pushing. Catch the whole class of errors per release.
- **Always bump the image tag** on every push. RunPod caches by digest;
  same-tag pushes are invisible to running workers.
- **PATCH the template + bounce standby workers in the same release step.**
  Otherwise warm workers keep serving the old code or model.
- **When OOM happens**, check `/runpod-volume/_worker_log_*.txt` first (boot
  diag). Read it via SSH on a pod that has the volume mounted.
- **When mutating infra** (endpoint, template, volumes, pods), confirm with
  the user first unless the action is clearly reversible and low-cost.
- **Secrets previously leaked in chat** (`HF_TOKEN`, `GHCR_PAT`,
  `RUNPOD_API_KEY`, Yandex S3 keys) — assume they'll be rotated post-project.
- **Workers cost money when alive.** Flashboot keeps `workersStandby` warm
  — useful for latency, expensive when idle. Be deliberate.
