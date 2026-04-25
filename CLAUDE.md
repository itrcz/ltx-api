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

6. **flashboot caches the worker process state**, including model weights in
   memory. Editing files on the volume does NOT take effect for warm
   workers — bump the image tag (or set `workersStandby=0`, then back to N)
   to force fresh spawns.

7. **RunPod caches workers by image digest.** Pushing a new image with the
   same tag does NOT replace existing workers. **Always bump the tag** AND
   PATCH the template's `imageName`.

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
  Dockerfile                — ComfyUI + custom nodes, pin transformers 4.57.6
  start.sh                  — sshd (optional), gemma symlink fix, launch comfy + handler
  extra_model_paths.yaml    — point ComfyUI at /runpod-volume/models/*
  system_prompts/           — gemma_i2v / gemma_t2v (used by LTXVGemmaEnhancePrompt)
  src/
    handler.py              — RunPod entrypoint, validates, calls workflow_builder, polls comfy, uploads
    workflow_builder.py     — builds the ComfyUI prompt JSON from a typed input
    workflow_template_api.json — hand-authored ComfyUI workflow (the live one)
    s3_upload.py            — Yandex S3 upload + presign
scripts/
  build-worker.sh           — buildx + push to ghcr.io/itrcz/ltx-worker-comfy
  setup-volume.sh           — provision a fresh region volume from public sources
  migrate-gemma-from-prod.sh — copy locally-quantized gemma _e4m3fn from c25vvptq5f
docs/
  api.md                    — public-facing API reference (request/response/examples)
  network-volume-setup.md   — operator guide for the two volume scripts
.env                        — secrets (HF_TOKEN, GHCR_PAT, RUNPOD_API_KEY, S3_*); gitignored
```

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
