"""Vast.ai PyWorker entry for the LTX-2.3 ComfyUI worker.

Wraps the same `run_pipeline()` that handler.py exposes to RunPod, but
adapts it to the vastai serverless framework (vastai.serverless.server.worker
== Worker + WorkerConfig + HandlerConfig + BenchmarkConfig).

Single POST route — `/run`. Sync vs async is chosen by the presence of
`webhook_url` in the request body, NOT by URL:

  Request body:
    {
      "input": { ... LTX request schema, see handler.py:_validate ... },
      "webhook_url":    str,     # optional, presence ⇒ async
      "webhook_secret": str,     # optional, HMAC key
    }

  Sync mode (no webhook_url):
    Response 200 body = run_pipeline()'s result dict ({video_url, ...}).
    The HTTP connection is held for the full pipeline duration (10–20 min
    for fullhd × 20s); vast's autoscaler routes directly to the worker
    so there's no proxy timeout to dodge. Client must set a generous
    read timeout (`requests.post(..., timeout=1800)`).

  Async mode (webhook_url set):
    Response 200 body =
      { "async": true, "job_id": "abc123def456",
        "estimated_seconds": 480,
        "result_url": "<presigned S3 URL of run/result/<job_id>/result.json>",
        "webhook_will_fire": true }
    Returned immediately (≪1 s) after spawning a background thread. The
    bg thread runs run_pipeline → writes result.json + video.mp4 into S3
    (run_pipeline already does this) → POSTs the result to webhook_url
    with HMAC-SHA256 signature.

S3 sidecar (`run/result/<job_id>/result.json`) is the canonical store of
record. If webhook delivery fails, the client can still recover via a
GET on `result_url`. Cross-worker requests are safe — vast may route the
next request to a different worker, but artifacts live in shared S3.

GPU serialization is done via a process-wide `threading.Lock`. ComfyUI
itself queues internally, but the autoscaler can't see that — by holding
this lock for the full pipeline we make the next request's HTTP wait
time reflect actual GPU contention, which is what vast uses for scaling
decisions.
"""
from __future__ import annotations

import asyncio
import os
import secrets  # noqa: F401  (kept for future job_id hardening if needed)
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

from vastai.serverless.server.worker import (
    BenchmarkConfig,
    HandlerConfig,
    Worker,
    WorkerConfig,
)

sys.path.insert(0, str(Path(__file__).parent))
from handler import _eta_seconds, _hms, _validate, run_pipeline  # noqa: E402
from jobs import JobState, deliver_webhook  # noqa: E402
from s3_upload import _client as _s3_client, upload_bytes  # noqa: E402


# Process-wide GPU lock. See module docstring.
_gpu_lock = threading.Lock()

# ETA helper mirrors handler.py:_eta_seconds but we need num_frames before
# build_workflow runs. 8k+1 rule from CLAUDE.md gotcha #3.
_FPS_DEFAULT = 24


def _approx_num_frames(duration_sec: float) -> int:
    k = max(1, int(round(duration_sec * _FPS_DEFAULT / 8)))
    return 8 * k + 1


def _gen_job_id() -> str:
    # 16 hex chars = 64 bits of entropy. Plenty for collision-safety inside
    # a single worker's lifetime (workers churn well before 2^32 jobs).
    return uuid.uuid4().hex[:16]


def _presign_result_json(job_id: str, ttl_sec: int) -> str:
    bucket = os.environ["S3_BUCKET"]
    s3 = _s3_client()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": f"run/result/{job_id}/result.json"},
        ExpiresIn=ttl_sec,
    )


def _run_sync_blocking(p: dict, job_id: str) -> dict:
    """Sync helper — hold the GPU lock for the full pipeline duration so a
    second concurrent /run waits, and the autoscaler sees the wait."""
    with _gpu_lock:
        return run_pipeline(p, job_id, progress_cb=None)


def _upload_error_sidecar(job_id: str, js) -> None:
    """Stand-in for the `result.json` that run_pipeline normally writes on
    success — gives async clients polling the presigned `result_url` a 200
    with an `error` field instead of polling 404 forever."""
    import json
    started = js.started_at or time.time()
    payload = {
        "job_id": job_id,
        "status": js.status,
        "error": js.error,
        "elapsed_sec": round(time.time() - started, 2),
    }
    try:
        upload_bytes(
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            key=f"run/result/{job_id}/result.json",
            content_type="application/json",
        )
        print(f"[async] {job_id} error sidecar uploaded", flush=True)
    except Exception as e:
        print(f"[async] {job_id} error-sidecar upload failed: "
              f"{type(e).__name__}: {str(e)[:200]}", flush=True)


def _run_async_in_thread(p: dict, job_id: str,
                         webhook_url: str,
                         webhook_secret: Optional[str]) -> None:
    """Background worker — runs the pipeline, then fires the webhook.

    Detached from the asyncio event loop so the client-facing 202 response
    isn't blocked. Holds `_gpu_lock` while the pipeline is running so the
    next /run (sync or async) serializes against it."""
    js = JobState(
        job_id=job_id,
        input=p,
        webhook_url=webhook_url,
        webhook_secret=webhook_secret,
        status="running",
        started_at=time.time(),
    )
    print(f"[async] {job_id} acquiring GPU lock", flush=True)
    with _gpu_lock:
        print(f"[async] {job_id} pipeline starting", flush=True)
        try:
            js.result = run_pipeline(p, job_id, progress_cb=None)
            js.status = "done"
        except Exception as e:
            js.error = f"{type(e).__name__}: {e}"
            js.status = "failed"
            print(f"[async] {job_id} pipeline failed: {js.error}\n"
                  f"{traceback.format_exc()[-2000:]}", flush=True)
            # On success run_pipeline writes the result.json sidecar inside S3
            # itself; on failure we still need to drop one so the client's
            # presigned `result_url` eventually GETs 200 with the error
            # instead of polling 404 forever.
            _upload_error_sidecar(job_id, js)
        finally:
            js.finished_at = time.time()
    elapsed = (js.finished_at or 0) - (js.started_at or 0)
    print(f"[async] {job_id} pipeline done status={js.status} elapsed={_hms(elapsed)}",
          flush=True)
    # Webhook fires OUTSIDE the lock — webhook latency must not stall the
    # next job. Result sidecar in S3 has already been written by run_pipeline.
    deliver_webhook(js)


async def run_handler(**payload) -> dict:
    """Vast remote_function for POST /run. See module docstring for shape."""
    inp = payload.get("input") or {}
    webhook_url = (payload.get("webhook_url") or "").strip() or None
    webhook_secret_raw = payload.get("webhook_secret")
    webhook_secret = (webhook_secret_raw.strip() or None) \
        if isinstance(webhook_secret_raw, str) else None

    try:
        p = _validate(inp)
    except ValueError as e:
        return {"error": str(e)}

    job_id = _gen_job_id()

    if webhook_url is None:
        # Sync mode — block until done.
        try:
            return await asyncio.to_thread(_run_sync_blocking, p, job_id)
        except Exception as e:
            return {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-3000:],
            }

    # Async mode — fire bg thread, return immediately.
    threading.Thread(
        target=_run_async_in_thread,
        args=(p, job_id, webhook_url, webhook_secret),
        daemon=True,
        name=f"ltx-job-{job_id}",
    ).start()

    ttl = int(os.environ.get("ASYNC_PRESIGN_TTL", "86400"))   # 24h default
    try:
        result_url = _presign_result_json(job_id, ttl)
    except Exception as e:
        # S3 mis-config — bail the async response with a clear error. The
        # bg thread will still run but the client has no way to recover
        # without webhook → tell them now.
        return {
            "error": f"async-mode requires reachable S3 for result_url: "
                     f"{type(e).__name__}: {e}",
        }

    return {
        "async": True,
        "job_id": job_id,
        "estimated_seconds": _eta_seconds(p["quality"],
                                          _approx_num_frames(p["duration_sec"])),
        "result_url": result_url,
        "webhook_will_fire": True,
    }


# ----- vastai SDK config -----

# Benchmark — sd × 5s × 8 steps. ~45-60s per run on 5090, 2 runs ≈ 90-120s
# at worker boot. Chosen to be in the same unit-of-measurement as a typical
# real request (hd × 5s × 8 = 10 units; this benchmark is 5 units / ~50s →
# max_perf ≈ 0.1 unit/s). With do_warmup=True the SDK does a free warmup
# run first to load models into VRAM; the measured runs are then warm and
# stable (low variance → vast doesn't flag as degraded).
#
# Grom-art's "too-cheap benchmark inflates max_perf, scaler under-provisions"
# gotcha (CLAUDE.md grom-art:108): tiny benchmark (0.5K + 2 steps) measured
# 213 units/s, real workload was 20 units/s — 10× off. We avoid that by
# picking a shape close to median real request.
BENCHMARK = BenchmarkConfig(
    generator=lambda: {
        "input": {
            "prompt": "benchmark",
            "quality": "sd",
            "aspect_ratio": "16:9",
            "duration_sec": 5.0,
            "seed": 42,
            "steps": 8,
        }
    },
    runs=2,
    concurrency=1,
    do_warmup=True,
)


# Workload per real request. Same units as benchmark (sd × 1s × 5steps ≈
# 1.0 unit). hd × 5s × 8steps ≈ 10. fullhd × 20s × 8 ≈ 80.
QUALITY_W = {"sd": 1.0, "hd": 2.0, "fullhd": 4.0}


def _workload(payload: dict) -> float:
    inp = payload.get("input") or {}
    q = inp.get("quality", "hd")
    dur = max(float(inp.get("duration_sec", 5)), 1.0)
    steps = int(inp.get("steps", 8))
    return dur * QUALITY_W.get(q, 2.0) * (steps / 8.0)


HANDLERS = [
    HandlerConfig(
        route="/run",
        allow_parallel_requests=True,   # SDK won't queue; our _gpu_lock does.
        max_queue_time=None,
        benchmark_config=BENCHMARK,
        workload_calculator=_workload,
        remote_function=run_handler,
    ),
]


def _build_worker() -> Worker:
    return Worker(WorkerConfig(
        model_server_url="http://127.0.0.1",
        model_server_port=int(os.environ.get("COMFY_PORT", "8188")),
        model_log_file=os.environ.get("COMFY_LOG_FILE", "/tmp/comfyui.log"),
        handlers=HANDLERS,
    ))


if __name__ == "__main__":
    _build_worker().run()
