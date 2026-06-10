"""FastAPI entry for Yotta Labs Serverless (QUEUE mode).

Yotta dispatches tasks via HTTP POST to `http://<worker>:<workerPort><processUri>`
(default 8000 + /run) with the body being the `taskData` from the original
`POST /v2/serverless/<id>/tasks` request. We extract `input` and call
`run_pipeline()` — same code path as the RunPod handler.

Request shape (from Yotta's docs):
    POST /run
    Body: { "input": { ... LTX schema ... }, ...other taskData fields... }

Response shape (returned to Yotta):
    On success: 200 with result JSON. Stored in task's `output` field.
                If task had `webhook` set, Yotta POSTs this body to it.
    On error:   200 with `{"error": "..."}` (Yotta records as SUCCESS with
                that body) OR 5xx (recorded as FAILED with no output).

We use a process-wide threading.Lock to serialize GPU work — if Yotta sends
two concurrent requests to one worker (unlikely with proper worker scaling
but possible), they queue instead of contending for the GPU.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent))
from handler import _validate, run_pipeline  # noqa: E402


_gpu_lock = threading.Lock()
app = FastAPI(title="ltx-2.3-yotta-worker")


def _run_sync(p: dict, job_id: str) -> dict:
    """Hold the GPU lock while run_pipeline is executing."""
    with _gpu_lock:
        return run_pipeline(p, job_id, progress_cb=None)


@app.get("/healthz")
async def healthz():
    """Simple liveness probe — returns 200 once FastAPI is up. Yotta may
    use it to know when the worker is ready to accept tasks (TBD; not in
    docs but standard pattern)."""
    return {"ok": True}


@app.post("/run")
async def run(body: dict):
    """Main worker endpoint. Receives Yotta's taskData JSON, extracts
    `input`, runs the LTX pipeline, returns the result dict.

    Same input contract as RunPod's handler.py (see _validate)."""
    inp = body.get("input") or {}
    try:
        p = _validate(inp)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Use Yotta's taskId if provided; fall back to UUID.
    import uuid
    job_id = body.get("taskId") or uuid.uuid4().hex[:16]

    try:
        # asyncio.to_thread so the FastAPI event loop stays unblocked
        # — important if Yotta health-checks or sends heartbeat probes
        # during the (potentially 10-20 min) pipeline.
        result = await asyncio.to_thread(_run_sync, p, job_id)
        return result
    except Exception as e:
        return JSONResponse(
            {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-3000:],
                "job_id": job_id,
            },
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("WORKER_PORT", "8000"))
    # workers=1 — we hold a per-process GPU lock; multi-process would
    # contend without seeing each other's lock. Yotta scales via more
    # worker instances, not more processes per instance.
    uvicorn.run(
        "yotta_worker:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
        access_log=True,
    )
