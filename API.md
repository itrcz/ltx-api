# LTX-2.3 Serverless API

RunPod Serverless endpoint that generates short video clips (up to 20 sec) with
synchronized audio from a prompt, one or more keyframes, or both. Runs on a
single RTX 5090 with ComfyUI + LTX-2.3 22B dev-fp8 + distilled LoRA + Gemma
fp8 text encoder.

## Endpoint

```
Base URL:  https://api.runpod.ai/v2/d7kud62ob6wwtp
Auth:      Authorization: Bearer <RUNPOD_API_KEY>
```

`RUNPOD_API_KEY` is your RunPod API key — keep it server-side, do not ship it
to clients. Build a thin gateway if you need client-facing access.

## Flow

Jobs are asynchronous. Three ways to get the result:

1. **Polling** — submit with `POST /run`, then hit `GET /status/<id>` until
   `status == COMPLETED`. Default choice for short synchronous-feeling flows
   in your own backend.
2. **Webhook** — submit with `POST /run` and include a `webhook` URL. RunPod
   calls it with the final payload when the job finishes (success *or*
   failure). No polling needed. Preferred for fan-out / queue workers.
3. **Synchronous** — `POST /runsync` blocks until the job finishes. Works for
   fast jobs only (≤300 s on the RunPod side); a fullhd 10-second clip can
   exceed that, so don't use runsync for anything other than `sd` / ≤5 sec
   smoke tests.

## Request schema

`POST /run` (or `/runsync`) with JSON body:

```jsonc
{
  "input": {
    "prompt":          "строка на любом языке; опционально",
    "negative_prompt": "optional, default: worst quality, static, blurry, ...",
    "quality":         "sd" | "hd" | "fullhd",      // default: "hd"
    "aspect_ratio":    "9:16" | "16:9",             // default: "9:16"
    "duration_sec":    1..20,                        // default: 5
    "steps":           5..30,                        // default: 8
    "seed":            42,                           // optional

    // Keyframes. Use either the `frames` array OR the shortcut fields below.
    // `frames` wins if both provided.
    "frames": [
      {"url": "https://...", "frame_idx":  0,  "strength": 1.0},
      {"url": "https://...", "frame_idx": 60,  "strength": 0.3},
      {"url": "https://...", "frame_idx": -1,  "strength": 0.5}
    ],

    // Shortcuts (translated into `frames` internally)
    "first_frame_url": "https://...",
    "last_frame_url":  "https://..."
  },

  "webhook": "https://your-api.example.com/ltx/callback"   // optional
}
```

### Parameter details

| Field | Type | Notes |
|---|---|---|
| `prompt` | string | Any language. Gemma auto-enhances into an English description with your dialogue wrapped in quotes. Required if no `frames`. |
| `negative_prompt` | string | Optional. Overrides the built-in default. |
| `quality` | enum | `sd` (1024×576 / 576×1024), `hd` (1344×768 / 768×1344), `fullhd` (1920×1088 / 1088×1920). |
| `aspect_ratio` | enum | `9:16` portrait or `16:9` landscape. |
| `duration_sec` | number | 1..20. Snapped to the closest `8n+1` frame count at 24 fps. |
| `steps` | integer | Stage-1 denoising steps. 5..30, default 8 (trained distilled schedule). More steps ≠ always better — the distilled LoRA is tuned for 8; values other than 8 use a linear schedule. |
| `seed` | integer | For reproducibility. Defaults to 42. |
| `frames[].url` | string | Public HTTPS URL — must be reachable from the worker (no auth). JPG/PNG/WebP, any size. |
| `frames[].frame_idx` | integer | Absolute frame index `0..N-1`. `-1` resolves to the last frame. |
| `frames[].strength` | number | `0..1` guide weight. Use `1.0` for the first frame (hard conditioning) and `0.2–0.5` for intermediate/last frames (soft keyframe). |
| `webhook` | string | Top-level, not inside `input`. Called once on job completion. |

### Validation errors

| Condition | Response |
|---|---|
| No `prompt` and no `frames`/`first_frame_url` | `{"error": "either 'prompt' or at least one image (frames / first_frame_url) must be provided"}` |
| `last_frame_url` without `first_frame_url` | `{"error": "'last_frame_url' requires 'first_frame_url'"}` |
| `frames[i].strength` out of `[0, 1]` | `{"error": "frames[i].strength must be in [0, 1]"}` |
| `frames[i].frame_idx` >= num_frames | Handler returns `{"error": "frame_idx ... out of range ..."}` |
| `steps` outside `5..30` | `{"error": "steps must be between 5 and 30"}` |
| `duration_sec` outside `1..20` | `{"error": "duration_sec must be between 1 and 20"}` |
| Unknown `quality` / `aspect_ratio` | `{"error": "quality must be one of [...]"}` |

## Response schema

### On submission (`/run`)

```json
{
  "id": "02709183-b86b-423a-885d-cd78dd672a75",
  "status": "IN_QUEUE"
}
```

### On completion

```json
{
  "id": "02709183-b86b-423a-885d-cd78dd672a75",
  "status": "COMPLETED",
  "output": {
    "video_url":     "https://timenote.s3.ru-central-1.yandexcloud.net/ltx/2026/04/24/<job_id>.mp4?X-Amz-Expires=3600&...",
    "width":          768,
    "height":         1344,
    "num_frames":     97,
    "fps":            24.0,
    "duration_sec":   4.0,
    "steps":          8,
    "quality":        "hd",
    "aspect_ratio":   "9:16",
    "mode":           "i2v",
    "keyframes":      [{"frame_idx": 0, "strength": 1.0}],
    "elapsed_sec":    79.4
  }
}
```

`video_url` is a presigned S3 URL (Yandex Cloud, EU-RO-1). Default TTL is
1 hour — cache the video bytes on your side if you need longer retention.

### On error

```json
{
  "id":     "...",
  "status": "FAILED",
  "error":  "ValueError: frame_idx 500 out of range 0..88"
}
```

## Example: polling

```bash
# 1. Submit
JOB=$(curl -s -X POST https://api.runpod.ai/v2/d7kud62ob6wwtp/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "мужчина говорит: привет ребята",
      "quality": "hd",
      "aspect_ratio": "9:16",
      "duration_sec": 4,
      "frames": [
        {"url": "https://your-cdn.example.com/first.jpg", "frame_idx": 0, "strength": 1.0}
      ]
    }
  }' | jq -r .id)
echo "job id: $JOB"

# 2. Poll every ~5 seconds
while true; do
  RESP=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
    https://api.runpod.ai/v2/d7kud62ob6wwtp/status/$JOB)
  STATUS=$(echo "$RESP" | jq -r .status)
  echo "status: $STATUS"
  case "$STATUS" in
    COMPLETED) echo "$RESP" | jq .output.video_url; break ;;
    FAILED|CANCELLED) echo "$RESP" | jq .error; exit 1 ;;
  esac
  sleep 5
done
```

Poll cadence: anything from 2 to 10 seconds is fine. RunPod does not rate-limit
/status. Typical end-to-end latency (cold start excluded): **25–30 s for sd/3s,
80–120 s for hd/5s, 250–350 s for fullhd/5s, up to ~600 s for fullhd/10s**.

## Example: webhook

Webhook avoids polling. RunPod does a single `POST <webhook>` with the exact
same JSON body you would have received from `/status` on completion:

```bash
curl -X POST https://api.runpod.ai/v2/d7kud62ob6wwtp/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a cat walking down the street",
      "quality": "fullhd",
      "duration_sec": 5
    },
    "webhook": "https://your-api.example.com/ltx/callback"
  }'
```

Your webhook endpoint receives:

```json
{
  "id": "...",
  "status": "COMPLETED",
  "output": { "video_url": "https://...", ... },
  "delayTime": 1200,     // ms in queue
  "executionTime": 95000 // ms of handler runtime
}
```

**Webhook requirements:**
- Must be publicly reachable HTTPS.
- Must return `2xx` within **30 s**. Slow or failing responses are retried
  with exponential backoff (up to ~4 attempts, then dropped).
- Not signed — validate by job id / hold a short-lived correlation table on
  your side tied to the id returned from `/run`.
- No webhook arrives for `CANCELLED` jobs cancelled via API.

On failure, the webhook still fires with `status: "FAILED"` and an `error`
field — so your handler must branch on status, not just assume success.

## Example: sync (sd / short only)

```bash
curl -X POST https://api.runpod.ai/v2/d7kud62ob6wwtp/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"prompt":"a short clip","quality":"sd","duration_sec":3}}'
# blocks until COMPLETED, FAILED, or 300s timeout
```

If the job exceeds 5 minutes on RunPod's side, `runsync` returns `IN_PROGRESS`
and you must fall back to `/status/<id>` polling.

## Operational notes

- **Cold start**: ~3–5 min on first call after idle (pull 20 GB image, warm
  weights). Subsequent calls while the worker is alive are warm (~same elapsed
  as shown above).
- **Autoscaling**: `workersMin=0`, `workersMax=2`. Set `workersMin=1` via the
  RunPod dashboard for latency-sensitive prod to eliminate cold starts (costs
  ~$0.69/hr per kept-warm 5090).
- **Timeouts**: the handler itself times out after 25 minutes (`JOB_TIMEOUT_S`).
  RunPod's endpoint-level timeout is `executionTimeoutMs=600000` (10 min) —
  bump it on the endpoint if you need longer fullhd/10s jobs.
- **Region lock**: endpoint + template + network volume are pinned to EU-RO-1.
  Request latency from outside EU will add ~50–150 ms on each HTTP call.
- **S3 storage**: output mp4s land in `s3://<bucket>/ltx/YYYY/MM/DD/<job_id>.mp4`.
  Presigned links live for `PRESIGN_TTL` seconds (default 3600). Bucket keeps
  them forever unless you add a lifecycle rule.

## Job statuses reference

| Status | Meaning |
|---|---|
| `IN_QUEUE` | Waiting for a worker. Cold-start counts here. |
| `IN_PROGRESS` | Worker accepted the job and is executing. |
| `COMPLETED` | Success. `output` has the video URL. |
| `FAILED` | Handler returned `{"error": ...}` or hit exception. See `error`. |
| `CANCELLED` | User called `/cancel/<id>`. |
| `TIMED_OUT` | Exceeded `executionTimeoutMs`. |

## Cancelling a job

```bash
curl -X POST https://api.runpod.ai/v2/d7kud62ob6wwtp/cancel/<id> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

Jobs already in `IN_PROGRESS` are killed; `IN_QUEUE` are removed from queue.
No webhook fires for cancelled jobs.

## Health check

```bash
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.ai/v2/d7kud62ob6wwtp/health
# → {"jobs": {...}, "workers": {"idle": 0, "ready": 1, "running": 0, ...}}
```

Use this to verify workers are warm before sending a latency-critical job.
