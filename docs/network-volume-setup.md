# RunPod network volume setup (LTX-2.3 prod weights)

How to provision a fresh ~50 GB network volume in any RunPod region with the
exact ComfyUI prod model set, ready to be attached to a Serverless endpoint.

## TL;DR

```bash
# from repo root, with .env populated (RUNPOD_API_KEY + HF_TOKEN)
# Step 1 — install publicly available files (~50 GB):
REGION=EUR-IS-1 ./scripts/setup-volume.sh
REGION=EUR-NO-1 ./scripts/setup-volume.sh
REGION=EU-RO-1  ./scripts/setup-volume.sh

# Step 2 — replace the gemma stub with the byte-exact prod file:
REGION=EUR-IS-1 VOLUME_ID=<id-from-step-1> ./scripts/migrate-gemma-from-prod.sh
REGION=EUR-NO-1 VOLUME_ID=<id-from-step-1> ./scripts/migrate-gemma-from-prod.sh
REGION=EU-RO-1  VOLUME_ID=<id-from-step-1> ./scripts/migrate-gemma-from-prod.sh
```

> **Step 2 is required.** `setup-volume.sh` installs the publicly available
> `gemma_3_12B_it_fp8_scaled.safetensors` from `Comfy-Org/ltx-2`, but the live
> ComfyUI workflow references `gemma_3_12B_it_fp8_e4m3fn.safetensors` — a
> locally-quantized variant that lives only on the original prod volume
> `c25vvptq5f`. The two files are NOT drop-in compatible: feeding the workflow
> the `_scaled` file aliased as `_e4m3fn` crashes inference with
> `RuntimeError: Promotion for Float8 Types is not supported, attempted to
> promote BFloat16 and Float8_e4m3fn` in the SigLIP vision tower.
>
> `migrate-gemma-from-prod.sh` spawns a temporary source pod with `c25vvptq5f`
> mounted, exposes the file over HTTP, then has a target pod fetch it directly.
> It cleans up both pods on exit.

> **Why 60 GB and not 50?** The four model files sum to ~47 GB, but HuggingFace
> double-buffers the largest in `.cache/huggingface/download/` while moving it
> into place. Peak disk during the gemma download is ~59 GB. 60 GB gives a
> small headroom; 50 GB triggers `Disk quota exceeded`. Existing volumes can
> be grown via `PATCH /networkvolumes/{id}` (`{"size": 60}`).

EUR-IS-2 is **not available** for network volumes — RunPod only supports
`EUR-IS-1` and `EUR-IS-3` in Iceland.

Each run prints the resulting `networkVolumeId`. Plug that ID into a per-region
endpoint/template (one endpoint per region — RunPod volumes are region-locked).

## Why per region

RunPod network volumes live in a single data center and **cannot be moved or
mounted across regions**. To run Serverless workers in EUR-IS-2 and EUR-NO-1
you need one populated 50 GB volume in each.

The existing 130 GB volume `c25vvptq5f` lives in EU-RO-1 and powers endpoint
`d7kud62ob6wwtp`. The script does **not** touch it.

## What gets installed (~50 GB)

| Path on volume                                                                | Source                                                | Size  | Purpose |
|--------------------------------------------------------------------------------|-------------------------------------------------------|-------|---------|
| `models/ltx23/ltx-2.3-22b-dev-fp8.safetensors`                                 | `Lightricks/LTX-2.3-fp8`                              | 27 GB | dev checkpoint (workflow primary) |
| `models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`                     | `Lightricks/LTX-2.3`                                  |  1 GB | stage-2 spatial upscaler |
| `models/ltx23/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors`            | `Lightricks/LTX-2.3`                                  |  7 GB | distill-style LoRA |
| `models/gemma-fp8/gemma_3_12B_it_fp8_scaled.safetensors`                       | `Comfy-Org/ltx-2/split_files/text_encoders/`          | 12 GB | text encoder (FP8 scaled) |

Plus a symlink farm that mirrors what `worker/extra_model_paths.yaml` expects:

```
models/
  checkpoints/                         (symlinks → ../ltx23/*.safetensors)
  text_encoders/                       (symlinks → ../gemma-fp8/*.safetensors)
  latent_upscale_models/               (symlinks → ../ltx23/*.safetensors)
  loras/ltxv/ltx2/                     (symlinks → ../../../ltx23/loras/*.safetensors)
```

### Note on the gemma filename

The live ComfyUI workflow (`worker/src/workflow_template_api.json`) references
`gemma_3_12B_it_fp8_e4m3fn.safetensors` by name, but the publicly available
Comfy-Org file is `gemma_3_12B_it_fp8_scaled.safetensors` — a different fp8
variant with different per-layer dtypes. They are **not** drop-in compatible:
loading `_scaled` under the `_e4m3fn` name crashes the SigLIP vision tower
during inference. `setup-volume.sh` therefore ships `_scaled` under its real
name only, and `migrate-gemma-from-prod.sh` swaps in the byte-exact `_e4m3fn`
afterwards.

## How the script works

1. **Find or create volume** — looks up an existing volume by name
   (`ltx-weights-<region-lc>`) in the target region; creates one if missing.
2. **Spawn a CPU pod** (`runpod/base:0.6.2-cpu`, ~$0.06/h) in the same region
   with the volume mounted at `/runpod-volume` and your `~/.runpod-ssh/key.pub`
   pre-installed.
3. **SSH in**, install `huggingface_hub`+`hf_transfer`, download each file with
   size verification (skips files already at expected size, redownloads on
   mismatch).
4. **Create the symlink farm** so ComfyUI's `extra_model_paths.yaml` resolution
   works without code changes.
5. **Terminate the pod** (volume persists). Cleanup runs even if the script
   fails partway.

The whole thing is idempotent — re-running on an already-populated volume
finishes in seconds.

## Using a populated volume from a Serverless endpoint

The volume ID printed by the script is what goes into the endpoint config.
RunPod Serverless endpoints are bound to a single data center, so create one
endpoint per region:

```bash
# Example: bind a new endpoint to the EUR-IS-2 volume
curl -X POST https://rest.runpod.io/v1/endpoints \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ltx-eur-is-2",
    "templateId": "ybom2lfy44",
    "networkVolumeId": "<paste-the-id-here>",
    "computeType": "GPU",
    "gpuTypeIds": ["NVIDIA GeForce RTX 5090"],
    "dataCenterIds": ["EUR-IS-2"],
    "workersMin": 0,
    "workersMax": 3,
    "flashboot": true
  }'
```

(Adjust `templateId` if you use a different worker image.)

## Manual fallback (if the script fails)

If you want to run the steps by hand:

```bash
# 1. Create volume
curl -X POST https://rest.runpod.io/v1/networkvolumes \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"ltx-weights-eur-is-2","size":50,"dataCenterId":"EUR-IS-2"}'

# 2. Spawn a CPU pod with the returned ID and your SSH pubkey via PUBLIC_KEY env.
#    Image: runpod/base:0.6.2-cpu, ports: ["22/tcp"], computeType: CPU.

# 3. SSH in once port mapping is up, then:
export HF_TOKEN=...
pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /runpod-volume/models/{ltx23/loras,gemma-fp8,checkpoints,text_encoders,latent_upscale_models,loras/ltxv/ltx2}

cd /tmp && hf download Lightricks/LTX-2.3-fp8 ltx-2.3-22b-dev-fp8.safetensors --local-dir .
mv ltx-2.3-22b-dev-fp8.safetensors /runpod-volume/models/ltx23/

# ... etc for the other 3 files. Then make the symlinks (see scripts/setup-volume.sh).

# 4. Delete the pod via REST API. Volume persists.
```

## Costs

- CPU pod during download: ~$0.06/h × ~30 min = **~$0.03 per provisioning run**.
- 50 GB network volume: **~$3.50/month** (RunPod is roughly $0.07/GB-month).
- HuggingFace egress: free.

## Troubleshooting

- **`SSH_KEY missing`** — generate once: `ssh-keygen -t ed25519 -N '' -f .runpod-ssh/key -C ltx-volume-setup`
- **`HTTP 400` on volume create** — region ID is case-sensitive (`EUR-IS-2`, not `eur-is-2`).
- **`hf download` 401** — `HF_TOKEN` doesn't have read access to `Lightricks/LTX-2.3-fp8` (gated repo). Visit the repo page on HF and accept terms once with your account.
- **Pod stuck in `EXITED` or never gets `portMappings`** — region might be out of CPU capacity. Try a different region or retry in a few minutes.
- **Disk full mid-download** — bump `SIZE_GB` and re-run; the volume is grown via `/networkvolumes/{id}` PATCH (not yet wired into this script).
