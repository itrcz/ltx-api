"""In-memory job store + HMAC-signed webhook delivery for the vast PyWorker.

Workers are stateless across reboots (vast may evict an instance any time),
so we treat the in-memory store as a best-effort scratch pad — the canonical
result always lives in S3 at `run/result/{job_id}/result.json` (the sidecar
that run_pipeline writes). A client that misses /status polling or webhook
delivery can recover by GET-ing the sidecar directly.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import requests

# Cap. ~100 jobs × ~3 KB JobState = ~300 KB, negligible. The cap exists so a
# misbehaving client that spams /submit doesn't grow memory unbounded; LRU
# eviction drops the oldest entries first.
MAX_JOBS = int(os.environ.get("PYWORKER_MAX_JOBS", "100"))

# Webhook retry schedule. Each (delay_sec, timeout_sec) pair = one attempt.
# 3 attempts total; if all fail we just log and move on — sidecar in S3 is
# the source of truth, no further escalation.
WEBHOOK_RETRIES = [(0.0, 10), (2.0, 10), (8.0, 15)]


@dataclass
class JobState:
    job_id: str
    status: str = "queued"               # queued | running | done | failed
    input: dict = field(default_factory=dict)
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    progress: float = 0.0
    current_stage: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None

    def to_status_dict(self) -> dict:
        elapsed = None
        if self.started_at is not None:
            end = self.finished_at if self.finished_at is not None else time.time()
            elapsed = round(end - self.started_at, 2)
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": round(self.progress, 3),
            "current_stage": self.current_stage,
            "elapsed_sec": elapsed,
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    """Thread-safe in-memory dict with LRU eviction. One per process."""

    def __init__(self, max_jobs: int = MAX_JOBS):
        self._lock = threading.RLock()
        self._jobs: "OrderedDict[str, JobState]" = OrderedDict()
        self._max = max_jobs

    def new(self, input_dict: dict,
            webhook_url: Optional[str] = None,
            webhook_secret: Optional[str] = None) -> JobState:
        job_id = uuid.uuid4().hex[:16]
        js = JobState(
            job_id=job_id,
            status="queued",
            input=input_dict,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
        )
        with self._lock:
            self._jobs[job_id] = js
            self._jobs.move_to_end(job_id)
            while len(self._jobs) > self._max:
                self._jobs.popitem(last=False)
        return js

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            js = self._jobs.get(job_id)
            if js is not None:
                self._jobs.move_to_end(job_id)
            return js

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            js = self._jobs.get(job_id)
            if js is None:
                return
            for k, v in fields.items():
                setattr(js, k, v)
            self._jobs.move_to_end(job_id)


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256,
    ).hexdigest()


def deliver_webhook(js: JobState) -> bool:
    """POST the job's final state to webhook_url. Returns True on first 2xx.
    Retries per WEBHOOK_RETRIES. All exceptions swallowed — sidecar in S3 is
    the durable source of truth, webhook is best-effort."""
    if not js.webhook_url:
        return False
    payload = {
        "job_id": js.job_id,
        "status": js.status,
        "result": js.result,
        "error": js.error,
        "elapsed_sec": (js.finished_at - js.started_at)
                       if (js.started_at and js.finished_at) else None,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "gr-tv-vst/0.1",
        "X-LTX-Job-Id": js.job_id,
        "X-LTX-Delivery": uuid.uuid4().hex,
    }
    if js.webhook_secret:
        headers["X-LTX-Signature"] = _sign(js.webhook_secret, body)

    for attempt_idx, (delay, timeout) in enumerate(WEBHOOK_RETRIES):
        if delay > 0:
            time.sleep(delay)
        try:
            r = requests.post(js.webhook_url, data=body, headers=headers, timeout=timeout)
            if 200 <= r.status_code < 300:
                print(f"[webhook] {js.job_id} delivered "
                      f"attempt={attempt_idx + 1} status={r.status_code}",
                      flush=True)
                return True
            print(f"[webhook] {js.job_id} non-2xx "
                  f"attempt={attempt_idx + 1} status={r.status_code} "
                  f"body={r.text[:200]!r}",
                  flush=True)
        except Exception as e:
            print(f"[webhook] {js.job_id} err "
                  f"attempt={attempt_idx + 1} {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
    print(f"[webhook] {js.job_id} EXHAUSTED — sidecar in S3 still authoritative",
          flush=True)
    return False
