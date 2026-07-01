# LTX-2.3 Server API

HTTP API for the self-hosted LTX-2.3 video generation server. Generates short
video clips (up to 20 s) with synchronized audio from a text prompt, one or more
keyframe images, or both — optionally lip-synced to a supplied audio track.

This is a **persistent server** (not serverless): it holds an in-memory queue and
renders one clip at a time on a single GPU. It exposes its own simple REST API —
**not** the RunPod API. If you've integrated against RunPod before, note the
differences: endpoint is `/generate` (not `/run`), the auth token is the server's
own API key (not a RunPod key), webhooks are HMAC-signed, and the server returns
`429` when its queue is full.

---

## Base URL & auth

```
Base URL:  https://<your-host>            # provided to you by the operator
Auth:      Authorization: Bearer <API_KEY>
```

- `API_KEY` is a shared token issued by the operator. Send it on **every**
  `/generate` and `/result` call. Keep it server-side; don't ship it to browsers.
- `GET /health`, `GET /healthz`, `GET /metrics` need no auth.
- Missing/invalid key → `401 {"detail": "invalid or missing API key"}`.

---

## Flow

Submit a job to `POST /generate`. Pick how you get the result:

1. **Async + polling** — `mode: "async"` → immediate `202 {task_id}`, then poll
   `GET /result/{task_id}` until `status` is `done` or `error`. Default choice.
2. **Async + webhook** — `mode: "async"` + a `webhook` URL → `202` immediately,
   and the server `POST`s the final result to your URL when done. No polling.
3. **Sync** — `mode: "sync"` → the connection is **held open** until the render
   finishes, then `200` with the result inline. Simplest to code, but ties up a
   connection for the whole render (tens of seconds to minutes). Set a generous
   client read timeout (e.g. 1800 s). If the render outlives the server's
   `SYNC_TIMEOUT_S`, you get `202` instead and must fall back to polling.

> **Capacity.** The server renders one clip at a time and holds a small queue
> (operator-configured, typically `QUEUE_MAX=2` → one rendering + one waiting).
> When full, `POST /generate` returns **`429`** with a `Retry-After` header.
> Treat `429` as "retry shortly or fail over to another host" — see
> [Capacity & 429](#capacity--429).

---

## `POST /generate`

```jsonc
{
  "input": {
    "prompt":          "строка на любом языке; опционально если задан кадр",
    "negative_prompt": "optional, default: worst quality, static, blurry, ...",
    "quality":         "sd" | "hd" | "fullhd",      // default: "hd"
    "aspect_ratio":    "9:16" | "16:9",             // default: "9:16"
    "duration_sec":    1..20,                        // default: 5
    "steps":           5..30,                        // default: 8
    "seed":            42,                           // optional

    // Keyframes. Use either the `frames` array OR the shortcut fields below.
    // `frames` wins if both are provided.
    "frames": [
      {"url": "https://...", "frame_idx":  0,  "strength": 1.0},
      {"url": "https://...", "frame_idx": 60,  "strength": 0.3},
      {"url": "https://...", "frame_idx": -1,  "strength": 0.5}
    ],

    // Shortcuts (translated into `frames` internally)
    "first_frame_url": "https://...",
    "last_frame_url":  "https://...",

    // Input audio (mp3/wav/m4a/… by URL). When set → custom-audio lip-sync:
    // the speech drives the subject's lips and is the output soundtrack.
    "audio_url": "https://..."
  },

  "mode":           "async",                       // "async" (default) | "sync"
  "webhook":        "https://your-api.example.com/ltx/callback",  // optional (async)
  "webhook_secret": "shared-hmac-secret",          // optional; signs the webhook
  "timeout":        600                            // optional; sync-mode wait (sec)
}
```

Everything inside `input` is the generation contract; the sibling fields
(`mode`, `webhook`, `webhook_secret`, `timeout`) control delivery.

### `input` parameter details

| Field | Type | Notes |
|---|---|---|
| `prompt` | string | Any language. Auto-enhanced into an English description with dialogue wrapped in quotes. Required if no `frames`/`first_frame_url`. |
| `negative_prompt` | string | Optional. Overrides the built-in default. |
| `quality` | enum | `sd` (1024×576 / 576×1024), `hd` (1344×768 / 768×1344), `fullhd` (1920×1088 / 1088×1920). |
| `aspect_ratio` | enum | `9:16` portrait or `16:9` landscape. |
| `duration_sec` | number | 1..20. Snapped to the closest `8n+1` frame count at 24 fps. |
| `steps` | integer | Denoising steps. 5..30, default 8. The distilled LoRA is tuned for 8; other values use a linear schedule (works, slightly off-distribution). |
| `seed` | integer | Reproducibility. Random per-request if omitted. |
| `frames[].url` | string | Public HTTPS URL reachable from the server (no auth). JPG/PNG/WebP, any size. |
| `frames[].frame_idx` | integer | Absolute frame index `0..N-1`. `-1` = last frame. |
| `frames[].strength` | number | `0..1` guide weight. `1.0` for the first frame (hard conditioning), `0.2–0.5` for intermediate/last (soft keyframe). |
| `first_frame_url` / `last_frame_url` | string | Shortcuts for `frames`. `last_frame_url` requires `first_frame_url`. |
| `audio_url` | string | Public HTTPS URL to audio (mp3/wav/m4a/…). Enables **lip-sync**: the speech is fixed as the soundtrack and the subject's lips follow it. Trimmed to the start of the clip length. `first_frame_url` supplies the face (i2v); without it the speaker is generated from the prompt (t2v). |

### Delivery fields

| Field | Type | Notes |
|---|---|---|
| `mode` | enum | `"async"` (default) or `"sync"`. |
| `webhook` | string | Async only. HTTPS URL the server POSTs the final result to. |
| `webhook_secret` | string | If set, the webhook body is HMAC-SHA256 signed (`X-LTX-Signature`). |
| `timeout` | number | Sync only. Seconds to hold the connection before falling back to `202`. Defaults to the server's `SYNC_TIMEOUT_S`. |

### Responses

**Async submit → `202`:**

```json
{ "task_id": "fc3cdc3005d94e76", "status": "queued", "estimated_seconds": 108 }
```

**Sync, finished in time → `200`:**

```json
{
  "task_id": "f203a4866a0846f0",
  "status": "done",
  "result": { "video_url": "https://...", "width": 768, "height": 1344, ... },
  "elapsed_sec": 82.1
}
```

**Sync, render outlived `timeout` → `202`** (job keeps running; poll `/result`):

```json
{ "task_id": "f203a4866a0846f0", "status": "running" }
```
(also sets a `Location: /result/{task_id}` header.)

**Validation error → `400`:**

```json
{ "detail": "either 'prompt' or at least one image (frames / first_frame_url) must be provided" }
```

**Queue full → `429`** (+ `Retry-After: 5` header):

```json
{ "detail": "queue full (2/2)" }
```

**Server still warming up → `503`:**

```json
{ "detail": "not ready (waiting_comfy)" }
```

### Validation rules

| Condition | `400 detail` |
|---|---|
| No `prompt` and no `frames`/`first_frame_url` | `either 'prompt' or at least one image ... must be provided` |
| `last_frame_url` without `first_frame_url` | `'last_frame_url' requires 'first_frame_url'` |
| `frames[i].strength` out of `[0,1]` | `frames[i].strength must be in [0, 1]` |
| `steps` outside `5..30` | `steps must be between 5 and 30` |
| `duration_sec` outside `1..20` | `duration_sec must be between 1 and 20` |
| Unknown `quality` / `aspect_ratio` | `quality must be one of [...]` |

---

## `GET /result/{task_id}`

Auth required. Returns the current state of a task.

| `status` | Body |
|---|---|
| `queued` | `{task_id, status, created_at, position}` — `position` = place in line (1 = next). |
| `running` | `{task_id, status, started_at, elapsed_sec}` |
| `done` | `{task_id, status, result, elapsed_sec}` |
| `error` | `{task_id, status, error}` |

Unknown or TTL-evicted id → `404 {"detail": "unknown or expired task_id"}`.
Results are retained for a limited window (operator-configured, default 1 h)
then evicted — fetch them promptly or rely on the webhook.

> **Error messages are intentionally generic.** When a render fails, `error` is a
> short, safe string (e.g. `"internal error while rendering — please retry; if it
> persists, contact support with the task_id"` or `"render timed out — ..."`). It
> never exposes internal detail (model filenames, ComfyUI node errors,
> tracebacks) — that is logged **server-side only**. Use the `task_id` (present in
> every response) as the support correlation key; the operator can grep the full
> detail from the server logs by that id. So branch on `status`, not on the
> wording of `error`.

**`done` example:**

```json
{
  "task_id": "fc3cdc3005d94e76",
  "status": "done",
  "elapsed_sec": 82.1,
  "result": {
    "video_url":     "https://<bucket>.s3.../<job_id>.mp4?X-Amz-Expires=3600&...",
    "thumbnail_url": "https://<bucket>.s3.../<job_id>.jpg?...",
    "width":          768,
    "height":         1344,
    "num_frames":     97,
    "fps":            24.0,
    "duration_sec":   4.0,
    "steps":          8,
    "seed":           42,
    "quality":        "hd",
    "aspect_ratio":   "9:16",
    "mode":           "i2v",
    "keyframes":      [{"frame_idx": 0, "strength": 1.0}],
    "elapsed_sec":    82.1
  }
}
```

`video_url` / `thumbnail_url` are **presigned S3 URLs** with a default 1-hour
TTL. Download and store the bytes if you need them longer.

---

## Webhooks

When you submit with `mode: "async"` + a `webhook` URL, the server `POST`s the
final state to that URL on completion (success **or** failure). The body:

```json
{
  "job_id":     "fc3cdc3005d94e76",
  "status":     "done",            // "done" | "failed"
  "result":     { "video_url": "https://...", ... },   // null on failure
  "error":      null,              // string on failure
  "elapsed_sec": 82.1
}
```

Headers:

| Header | Meaning |
|---|---|
| `X-LTX-Job-Id` | The `task_id` / `job_id`. |
| `X-LTX-Delivery` | Unique per delivery attempt (for dedup/logging). |
| `X-LTX-Signature` | `sha256=<hex>` HMAC of the raw body with `webhook_secret`. Present only if you supplied a secret. |

**Requirements:**
- Publicly reachable HTTPS, returns `2xx` quickly. Non-2xx / errors are retried
  (3 attempts: immediate, +2 s, +8 s) then dropped.
- **Branch on `status`** — a `failed` webhook still fires; don't assume success.
- The webhook is best-effort. If you never receive it, the job result is still
  retrievable via `GET /result/{task_id}` (within its TTL).

**Verify the signature** (Python):

```python
import hmac, hashlib

def verify(raw_body: bytes, header_sig: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header_sig or "")
```

Compute over the **raw request bytes**, not a re-serialized JSON object.

---

## Capacity & 429

The server processes one render at a time and holds a bounded queue (operator
sets `QUEUE_MAX`, typically 2). Once `queued + in-progress` reaches the cap,
`POST /generate` returns:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 5

{ "detail": "queue full (2/2)" }
```

Client strategy:
- Honor `Retry-After` and retry, **or**
- Fail over to another server instance (run several behind a load balancer; each
  is independent), **or**
- Queue on your side and back-pressure your own producers.

Check current load any time via `GET /health` (`queue_depth`, `in_flight`).

---

## `GET /health`

No auth. `503` while the server is still starting (model server warming up),
`200` once ready:

```json
{
  "status": "ok",
  "ready": true,
  "stage": "ready",
  "queue_depth": 1,
  "in_flight": 1,
  "queue_max": 2
}
```

During boot: `503 {"status":"loading","ready":false,"stage":"waiting_comfy"}`.
Use this before sending latency-critical jobs.

`GET /healthz` is a lightweight liveness alias → `{"ok": true}`.

---

## `GET /metrics`

No auth. Prometheus exposition. Key series:

| Metric | Meaning |
|---|---|
| `ltx_ready` | `1` once accepting work. |
| `ltx_queue_depth` | Tasks currently queued. |
| `ltx_in_flight` | `0` or `1` — render on the GPU. |
| `ltx_tasks_total{status}` | Lifecycle counts (`submitted`/`done`/`error`/`expired`). |
| `ltx_gen_seconds{mode}` | Render time histogram (`t2v`/`i2v`). |
| `ltx_queue_wait_seconds` | Submit-to-pickup latency. |
| `ltx_queue_full_total` | Submissions rejected with `429`. |
| `ltx_webhook_total{outcome}` | Webhook deliveries (`ok`/`failed`). |
| `ltx_sync_timeouts_total` | Sync requests that fell back to `202`. |

---

## Examples

### Async + polling

```bash
BASE=https://<your-host>; KEY=<API_KEY>

JOB=$(curl -s -X POST $BASE/generate \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "мужчина говорит: привет ребята",
      "quality": "hd", "aspect_ratio": "9:16", "duration_sec": 4,
      "frames": [{"url": "https://your-cdn.example.com/first.jpg", "frame_idx": 0, "strength": 1.0}]
    },
    "mode": "async"
  }' | jq -r .task_id)
echo "task: $JOB"

while true; do
  R=$(curl -s -H "Authorization: Bearer $KEY" $BASE/result/$JOB)
  S=$(echo "$R" | jq -r .status)
  echo "status: $S"
  case "$S" in
    done)  echo "$R" | jq -r .result.video_url; break ;;
    error) echo "$R" | jq -r .error; exit 1 ;;
  esac
  sleep 3
done
```

### Async + webhook

```bash
curl -s -X POST $BASE/generate \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
    "input": { "prompt": "a cat walking down the street", "quality": "fullhd", "duration_sec": 5 },
    "mode": "async",
    "webhook": "https://your-api.example.com/ltx/callback",
    "webhook_secret": "shared-hmac-secret"
  }'
# → 202 {"task_id": "...", "status": "queued", "estimated_seconds": ...}
# Your endpoint later receives a POST with the result + X-LTX-Signature.
```

### Sync (short clips; holds the connection)

```bash
curl -s --max-time 1800 -X POST $BASE/generate \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"input":{"prompt":"a short clip","quality":"sd","duration_sec":3},"mode":"sync"}'
# → 200 with {"status":"done","result":{...}} once the render finishes,
#   or 202 {"status":"running"} if it outlives the server timeout (then poll /result).
```

---

## Latency reference

Typical end-to-end render time (warm server, excludes the one-time boot):

| quality | 3 s | 5 s | 10 s |
|---|---|---|---|
| `sd` | ~25–30 s | ~40–55 s | ~80–110 s |
| `hd` | ~50–65 s | ~80–120 s | ~160–240 s |
| `fullhd` | ~120–160 s | ~250–350 s | ~500–700 s |

`estimated_seconds` in the `202` response is a rough hint (includes a fixed
cold-kernel allowance) — use `/result` polling or the webhook for the real
finish, not a fixed sleep.
