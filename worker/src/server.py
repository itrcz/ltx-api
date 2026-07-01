"""Universal (non-serverless) HTTP server for the LTX-2.3 ComfyUI worker.

Runs as a persistent GPU container on a box you own (first target: a bare-metal
RTX 5090 VDS). It is the fourth deployment mode alongside RunPod (`handler.py`),
vast (`pyworker.py`) and Yotta (`yotta_worker.py`) — and like all of them it
reuses the shared `run_pipeline()` from handler.py verbatim, so every pipeline
constraint (8k+1 frames, gemma _e4m3fn, distilled-LoRA schedule, S3 upload) is
preserved automatically. This module is only the transport layer:

  - FastAPI + uvicorn, --workers 1 (the queue + result store are in-process
    Python objects; >1 worker = multiple disjoint queues + multiple GPU users).
  - One background worker thread consumes a thread-safe queue.Queue of task_ids
    and runs run_pipeline() serially (one render at a time on the GPU).
  - SYNC requests await an asyncio.Future the worker resolves cross-thread.
  - Webhooks fire on a detached daemon thread (jobs.deliver_webhook, HMAC-signed,
    with retries) so neither the worker nor the event loop waits on them.
  - A janitor evicts tasks queued > QUEUE_TTL_S and results > RESULT_TTL_S.
  - QUEUE_MAX caps held work (queued + on-GPU); overflow gets 429 so a client
    can fail over to another box. The intended self-host setting is QUEUE_MAX=2
    (one rendering, one waiting).
  - Prometheus metrics at /metrics; readiness at /health.

"Boot" here is NOT an in-process model load (the model lives in the separate
ComfyUI process). It is just: wait for ComfyUI on :8188 to answer /system_stats.

Endpoints: POST /generate (async|sync), GET /result/{id}, GET /health, GET /metrics.

Request body for POST /generate:
    {
      "input":   { ... LTX schema, see handler._validate ... },   required
      "mode":    "async" | "sync",        default "async"
      "webhook": "https://...",           optional (async); HMAC-signed if secret
      "webhook_secret": "...",            optional; X-LTX-Signature = sha256=...
      "timeout": 120                      optional; sync wait before 202 fallback
    }
"""
from __future__ import annotations

import asyncio
import hmac
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from handler import (  # noqa: E402
    _eta_seconds,
    _hms,
    _validate,
    _wait_comfy_ready,
    run_pipeline,
)
from jobs import JobState, deliver_webhook  # noqa: E402


def log(scope: str, msg: str) -> None:
    """One-line stdout format, matching handler's [HH:MM:SS] scope | message."""
    print(f"[{time.strftime('%H:%M:%S')}] {scope:<10} | {msg}", flush=True)


def _public_error(exc: Exception) -> str:
    """Map an internal pipeline exception to a SAFE client-facing message.

    The raw exception text leaks implementation detail — ComfyUI node names,
    model filenames, tracebacks ("lora_name '...talkvid...' not in [...]"). None
    of that should reach an API caller. The full detail is logged server-side and
    kept in Task.error_detail; the client only ever sees one of these generic
    strings. The task_id is already in the response body for support correlation.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if "timed out" in text or "timeout" in text or "watchdog" in text:
        return "render timed out — please retry, optionally with a shorter clip or lower quality"
    # Everything else (ComfyUI validation, missing model, OOM, decode failure,
    # S3 upload error, ...) is a server-side problem the caller can't fix and
    # must not see the internals of.
    return "internal error while rendering — please retry; if it persists, contact support with the task_id"


# ---------------------------------------------------------------------------
# Config (all env-tunable)
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("PORT", "8000"))
API_KEY = os.environ.get("API_KEY") or None  # if unset, submission endpoints are open
SYNC_TIMEOUT_S = float(os.environ.get("SYNC_TIMEOUT_S", "600"))
# Capacity gate: max tasks the server will hold (queued + the one on the GPU).
# 0 (or unset) = unlimited. The self-host target runs QUEUE_MAX=2 (one
# rendering + one waiting); a 3rd concurrent submit gets 429 + Retry-After so
# the client can fail over to another box. Accepts MAX_QUEUE as an alias for
# parity with grom-art's server.py.
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", os.environ.get("MAX_QUEUE", "0")))
# Opt-in only: exposes the full internal error (ComfyUI node errors, model
# filenames, traceback) via GET /result instead of the sanitized message.
# Never enable on a box handling real client traffic.
DEBUG_ERRORS = os.environ.get("DEBUG_ERRORS") == "1"
QUEUE_TTL_S = float(os.environ.get("QUEUE_TTL_S", "3600"))
RESULT_TTL_S = float(os.environ.get("RESULT_TTL_S", "3600"))
JANITOR_INTERVAL_S = float(os.environ.get("JANITOR_INTERVAL_S", "60"))
# How long boot waits for ComfyUI to come up before the server gives up.
COMFY_BOOT_TIMEOUT_S = int(os.environ.get("COMFY_BOOT_TIMEOUT_S", "1200"))

# FAKE_PIPE=1 stubs run_pipeline (no GPU, no ComfyUI) for local smoke tests of
# the queue / worker / sync / async / webhook / TTL machinery.
FAKE_PIPE = os.environ.get("FAKE_PIPE") == "1"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
import queue as _queue  # noqa: E402

WORK_QUEUE: "_queue.Queue[str]" = _queue.Queue()
TASKS: "dict[str, Task]" = {}
STORE_LOCK = threading.Lock()
READY = threading.Event()
BOOT_STAGE = "starting"           # starting|waiting_comfy|ready|boot_failed: ...
EVENT_LOOP: "asyncio.AbstractEventLoop | None" = None
IN_FLIGHT = 0                     # 0 or 1; guarded by STORE_LOCK

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
M_QUEUE_DEPTH = Gauge("ltx_queue_depth", "Tasks currently queued")
M_IN_FLIGHT = Gauge("ltx_in_flight", "Render currently on the GPU (0 or 1)")
M_READY = Gauge("ltx_ready", "1 once ComfyUI is up and the server accepts work")
M_TASKS = Counter("ltx_tasks_total", "Task lifecycle counts", ["status"])
M_GEN = Histogram(
    "ltx_gen_seconds", "End-to-end render time", ["mode"],
    buckets=(5, 10, 30, 60, 120, 240, 480, 720, 1200, 1800),
)
M_QUEUE_WAIT = Histogram(
    "ltx_queue_wait_seconds", "Time from submit to pickup",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 300, 1200),
)
M_WEBHOOK = Counter("ltx_webhook_total", "Webhook deliveries", ["outcome"])
M_SYNC_TIMEOUTS = Counter("ltx_sync_timeouts_total", "Sync requests that fell back to 202")
M_QUEUE_FULL = Counter("ltx_queue_full_total", "Submissions rejected because QUEUE_MAX was reached")


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------
@dataclass
class Task:
    id: str
    params: dict                   # validated LTX input (output of _validate)
    raw_input: dict                # original input echoed in webhook job state
    webhook: "str | None" = None
    webhook_secret: "str | None" = None
    mode: str = "async"
    state: str = "queued"          # queued|running|done|error|expired
    created_at: float = field(default_factory=time.time)
    started_at: "float | None" = None
    finished_at: "float | None" = None
    result: "dict | None" = None
    error: "str | None" = None          # sanitized, client-facing
    error_detail: "str | None" = None   # full internal detail — logs only, never returned
    _future: "asyncio.Future | None" = None

    def public(self) -> dict:
        body: dict = {"task_id": self.id, "status": self.state}
        if self.state == "queued":
            body["created_at"] = self.created_at
        elif self.state == "running":
            body["started_at"] = self.started_at
            if self.started_at:
                body["elapsed_sec"] = round(time.time() - self.started_at, 2)
        elif self.state == "done":
            body["result"] = self.result
            if self.started_at and self.finished_at:
                body["elapsed_sec"] = round(self.finished_at - self.started_at, 2)
        elif self.state == "error":
            body["error"] = self.error
            if DEBUG_ERRORS and self.error_detail:
                body["error_detail"] = self.error_detail
        return body


# ---------------------------------------------------------------------------
# Boot: wait for ComfyUI -> ready
# ---------------------------------------------------------------------------
def _fake_run_pipeline(p: dict, job_id: str, *, progress_cb=None, timeout_s=None) -> dict:
    """GPU-less stub matching run_pipeline's signature + output shape, for smoke tests."""
    time.sleep(float(os.environ.get("FAKE_GEN_S", "2")))
    return {
        "video_url": f"https://example.invalid/{job_id}.mp4",
        "thumbnail_url": f"https://example.invalid/{job_id}.jpg",
        "elapsed_sec": 2.0,
        "width": 768, "height": 1344, "num_frames": 121,
        "duration_sec": p["duration_sec"], "fps": 24,
        "mode": "i2v" if p["frames"] else "t2v", "quality": p["quality"],
    }


# Bound at boot to run_pipeline (real) or the stub (FAKE_PIPE).
RUN = run_pipeline


def _boot() -> None:
    """Blocking boot — run via asyncio.to_thread so /health can serve 503
    meanwhile. For this worker, boot is just waiting for ComfyUI to be up."""
    global BOOT_STAGE, RUN
    try:
        if FAKE_PIPE:
            log("init", "FAKE_PIPE=1 — stubbing run_pipeline (no GPU, no ComfyUI)")
            RUN = _fake_run_pipeline
        else:
            BOOT_STAGE = "waiting_comfy"
            log("init", "waiting for ComfyUI to come up...")
            _wait_comfy_ready(timeout_s=COMFY_BOOT_TIMEOUT_S)
        M_READY.set(1)
        BOOT_STAGE = "ready"
        READY.set()
        log("init", f"server ready — accepting work (QUEUE_MAX={QUEUE_MAX or 'unlimited'})")
    except Exception as e:  # boot failure must be visible, not a silent hang
        BOOT_STAGE = f"boot_failed: {e}"
        log("init", f"BOOT FAILED: {e!r}")
        raise


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
def _resolve_future(task: Task) -> None:
    """Resolve a SYNC waiter's future on the event loop (cross-thread-safe)."""
    fut = task._future
    if fut is None or EVENT_LOOP is None:
        return

    def _set() -> None:
        if not fut.done():
            fut.set_result(task)

    EVENT_LOOP.call_soon_threadsafe(_set)


def _fire_webhook(task: Task) -> None:
    """Deliver the webhook on a detached daemon thread. jobs.deliver_webhook is
    blocking (HMAC sign + retries with sleeps) — keeping it off the worker
    thread lets the next render start immediately."""
    if not task.webhook:
        return
    js = JobState(
        job_id=task.id,
        status="done" if task.state == "done" else "failed",
        input=task.raw_input,
        webhook_url=task.webhook,
        webhook_secret=task.webhook_secret,
        started_at=task.started_at,
        finished_at=task.finished_at,
        result=task.result,
        error=task.error,
    )

    def _deliver() -> None:
        ok = deliver_webhook(js)
        M_WEBHOOK.labels(outcome="ok" if ok else "failed").inc()

    threading.Thread(target=_deliver, name=f"webhook-{task.id[:6]}", daemon=True).start()


def _worker_loop() -> None:
    global IN_FLIGHT
    while True:
        task_id = WORK_QUEUE.get()
        with STORE_LOCK:
            task = TASKS.get(task_id)
            if task is None or task.state != "queued":
                # Evicted by TTL or already handled — skip.
                M_QUEUE_DEPTH.set(WORK_QUEUE.qsize())
                continue
            task.state = "running"
            task.started_at = time.time()
            IN_FLIGHT = 1
            M_IN_FLIGHT.set(1)
            M_QUEUE_DEPTH.set(WORK_QUEUE.qsize())
        M_QUEUE_WAIT.observe(task.started_at - task.created_at)

        mode = "i2v" if task.params.get("frames") else "t2v"
        log(f"job {task_id[:6]}", f"render start mode={mode} quality={task.params.get('quality')}")
        try:
            out = RUN(task.params, task_id, progress_cb=None)
            with STORE_LOCK:
                task.state = "done"
                task.result = out
                task.finished_at = time.time()
            M_TASKS.labels(status="done").inc()
            if isinstance(out, dict) and out.get("elapsed_sec"):
                M_GEN.labels(mode=mode).observe(float(out["elapsed_sec"]))
            log(f"job {task_id[:6]}", f"done in {_hms(task.finished_at - task.started_at)}")
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            with STORE_LOCK:
                task.state = "error"
                task.error = _public_error(e)   # sanitized — what the client sees
                task.error_detail = detail       # full — logs only, never returned
                task.finished_at = time.time()
            M_TASKS.labels(status="error").inc()
            # Full detail (ComfyUI node errors, model names, traceback) stays
            # server-side in the logs, keyed by the full task_id for correlation.
            log(f"job {task_id[:6]}", f"ERROR job={task_id}: {detail}")
            log(f"job {task_id[:6]}", "traceback:\n" + tb)
        finally:
            with STORE_LOCK:
                IN_FLIGHT = 0
                M_IN_FLIGHT.set(0)
            _resolve_future(task)
            _fire_webhook(task)


# ---------------------------------------------------------------------------
# Janitor: evict expired queued tasks & old results
# ---------------------------------------------------------------------------
async def _janitor_loop() -> None:
    while True:
        await asyncio.sleep(JANITOR_INTERVAL_S)
        now = time.time()
        with STORE_LOCK:
            for tid, t in list(TASKS.items()):
                if t.state == "queued" and now - t.created_at > QUEUE_TTL_S:
                    t.state = "expired"
                    t.finished_at = now
                    M_TASKS.labels(status="expired").inc()
                elif (
                    t.state in ("done", "error", "expired")
                    and t.finished_at is not None
                    and now - t.finished_at > RESULT_TTL_S
                ):
                    del TASKS[tid]
            M_QUEUE_DEPTH.set(WORK_QUEUE.qsize())


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global EVENT_LOOP
    EVENT_LOOP = asyncio.get_running_loop()
    threading.Thread(target=_worker_loop, name="gpu-worker", daemon=True).start()
    asyncio.create_task(_janitor_loop())
    # Boot (wait-for-ComfyUI) runs off the loop so /health serves 503 meanwhile.
    asyncio.create_task(asyncio.to_thread(_boot))
    yield


app = FastAPI(title="ltx-2.3 universal worker", lifespan=lifespan)


def require_auth(request: Request) -> None:
    if API_KEY is None:
        return
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    if not hmac.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class GenerateRequest(BaseModel):
    input: dict = Field(default_factory=dict)
    mode: str = "async"
    webhook: "str | None" = None
    webhook_secret: "str | None" = None
    timeout: "float | None" = None


@app.post("/generate", dependencies=[Depends(require_auth)])
async def generate(req: GenerateRequest):
    if not READY.is_set():
        raise HTTPException(status_code=503, detail=f"not ready ({BOOT_STAGE})")

    # _validate enforces the full LTX contract (prompt-or-image, quality, AR,
    # duration, steps, seed, audio_url, ...) and normalises the dict.
    try:
        params = _validate(req.input or {})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Capacity gate. Reject (429) when queued + in-flight would exceed QUEUE_MAX,
    # so a client can fail over to another box. 0 = unlimited.
    if QUEUE_MAX > 0:
        with STORE_LOCK:
            active = WORK_QUEUE.qsize() + IN_FLIGHT
        if active >= QUEUE_MAX:
            M_QUEUE_FULL.inc()
            raise HTTPException(
                status_code=429,
                detail=f"queue full ({active}/{QUEUE_MAX})",
                headers={"Retry-After": "5"},
            )

    tid = uuid.uuid4().hex[:16]
    task = Task(
        id=tid,
        params=params,
        raw_input=req.input or {},
        webhook=(req.webhook or "").strip() or None,
        webhook_secret=(req.webhook_secret or "").strip() or None,
        mode=req.mode,
    )
    if req.mode == "sync":
        task._future = EVENT_LOOP.create_future()  # type: ignore[union-attr]
    with STORE_LOCK:
        TASKS[tid] = task
    WORK_QUEUE.put(tid)
    M_TASKS.labels(status="submitted").inc()
    M_QUEUE_DEPTH.set(WORK_QUEUE.qsize())
    log(f"job {tid[:6]}", f"queued mode={req.mode} "
                          f"quality={params.get('quality')} dur={params.get('duration_sec')}s")

    if req.mode != "sync":
        return JSONResponse(
            status_code=202,
            content={
                "task_id": tid,
                "status": "queued",
                "estimated_seconds": _eta_seconds(params["quality"],
                                                  _approx_num_frames(params["duration_sec"])),
            },
        )

    timeout = req.timeout if req.timeout is not None else SYNC_TIMEOUT_S
    try:
        await asyncio.wait_for(asyncio.shield(task._future), timeout=timeout)
    except asyncio.TimeoutError:
        # GPU still busy — fall back to polling; the job keeps running.
        M_SYNC_TIMEOUTS.inc()
        return JSONResponse(
            status_code=202,
            content={"task_id": tid, "status": task.state},
            headers={"Location": f"/result/{tid}"},
        )
    status = 200 if task.state == "done" else 500
    return JSONResponse(status_code=status, content=task.public())


@app.get("/result/{task_id}", dependencies=[Depends(require_auth)])
async def get_result(task_id: str):
    with STORE_LOCK:
        task = TASKS.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="unknown or expired task_id")
        body = task.public()
        if task.state == "queued":
            body["position"] = sum(
                1 for t in TASKS.values()
                if t.state == "queued" and t.created_at <= task.created_at
            )
    return JSONResponse(status_code=200, content=body)


@app.get("/health")
async def health():
    if not READY.is_set():
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "ready": False, "stage": BOOT_STAGE},
        )
    with STORE_LOCK:
        in_flight = IN_FLIGHT
    return {
        "status": "ok",
        "ready": True,
        "stage": BOOT_STAGE,
        "queue_depth": WORK_QUEUE.qsize(),
        "in_flight": in_flight,
        "queue_max": QUEUE_MAX or "unlimited",
    }


# Liveness alias (handler/yotta use /healthz; keep both).
@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/debug/comfylog")
async def debug_comfylog(lines: int = 400):
    """DEBUG_ERRORS-gated: tail ComfyUI's own stdout/stderr log. Only path to
    ComfyUI-side node exceptions when the history record shows a misleadingly
    clean 'success' with empty outputs (silently-skipped/pruned execution)."""
    if not DEBUG_ERRORS:
        raise HTTPException(status_code=404, detail="not found")
    log_path = os.environ.get("COMFY_LOG_FILE", "/tmp/comfyui.log")
    try:
        with open(log_path, "r", errors="replace") as f:
            content = f.readlines()
    except FileNotFoundError:
        return PlainTextResponse(f"no such file: {log_path}")
    return PlainTextResponse("".join(content[-lines:]))


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ETA helper needs num_frames before build_workflow runs (8k+1 rule, CLAUDE.md
# gotcha #3). Mirrors pyworker._approx_num_frames.
_FPS_DEFAULT = 24


def _approx_num_frames(duration_sec: float) -> int:
    k = max(1, int(round(duration_sec * _FPS_DEFAULT / 8)))
    return 8 * k + 1


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1, timeout_keep_alive=130)
