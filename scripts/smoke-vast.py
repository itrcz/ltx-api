#!/usr/bin/env python3
"""End-to-end smoke test for the LTX-2.3 vast.ai endpoint.

Flow:
  1. POST /route/?endpoint_id=<id> → vast picks a worker → returns worker URL + JWT
  2. POST <worker_url>/run with a tiny payload → block until done (sync) OR
     return 202 + result_url + webhook (async)

Usage:
    VAST_ENDPOINT_ID=12345 VAST_API_KEY=... python3 scripts/smoke-vast.py [--async]

Env:
    VAST_ENDPOINT_ID   Required. Endpoint ID from cloud.vast.ai/serverless/
    VAST_API_KEY       Required. API key from cloud.vast.ai → Account → API Keys
    WEBHOOK_URL        Optional. If set, runs in async mode and exits — the
                       webhook receiver should print the result there.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests


def route(endpoint_id: str, api_key: str) -> dict:
    """Ask the autoscaler for a worker URL. Returns {url, worker_jwt, ...}."""
    print(f"[{time.strftime('%H:%M:%S')}] POST /route/ endpoint={endpoint_id}", flush=True)
    r = requests.post(
        "https://run.vast.ai/route/",
        params={"endpoint_id": endpoint_id},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    if not r.ok:
        sys.exit(f"/route/ failed: {r.status_code} {r.text[:500]}")
    d = r.json()
    print(f"[{time.strftime('%H:%M:%S')}] got worker: {d.get('url') or d}", flush=True)
    return d


def run_sync(worker_url: str, jwt: str, payload: dict) -> dict:
    print(f"[{time.strftime('%H:%M:%S')}] POST {worker_url}/run (sync)", flush=True)
    t0 = time.time()
    # Generous timeout — sd × 3s ≈ 30s on warm; cold first-request ≈ 90s.
    # If you test fullhd × 20s, bump this to 1800 s.
    r = requests.post(
        f"{worker_url}/run",
        headers={"Authorization": f"Bearer {jwt}"},
        json={"input": payload},
        timeout=600,
    )
    wall = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] done in {wall:.1f}s: HTTP {r.status_code}", flush=True)
    try:
        return r.json()
    except Exception:
        return {"raw": r.text[:1000]}


def run_async(worker_url: str, jwt: str, payload: dict, webhook_url: str, webhook_secret: str | None) -> dict:
    print(f"[{time.strftime('%H:%M:%S')}] POST {worker_url}/run (async, webhook={webhook_url[:50]}...)", flush=True)
    body = {"input": payload, "webhook_url": webhook_url}
    if webhook_secret:
        body["webhook_secret"] = webhook_secret
    r = requests.post(
        f"{worker_url}/run",
        headers={"Authorization": f"Bearer {jwt}"},
        json=body,
        timeout=30,
    )
    print(f"[{time.strftime('%H:%M:%S')}] {r.status_code}: {r.json()}", flush=True)
    return r.json()


def poll_result_url(result_url: str, max_wait: int = 1800) -> dict | None:
    """Async polling fallback — GETs the presigned S3 result.json URL until
    200 (done) or max_wait elapsed. 404 means still running."""
    print(f"[{time.strftime('%H:%M:%S')}] polling {result_url[:80]}...", flush=True)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        r = requests.get(result_url, timeout=30)
        if r.status_code == 200:
            print(f"[{time.strftime('%H:%M:%S')}] result.json found after {time.time()-t0:.0f}s")
            return r.json()
        if r.status_code != 404:
            print(f"  unexpected: {r.status_code} {r.text[:200]}")
        time.sleep(15)
    print(f"[{time.strftime('%H:%M:%S')}] timeout after {max_wait}s")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sync", "async"], default="sync")
    ap.add_argument("--quality", default="sd", choices=["sd", "hd", "fullhd"])
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt", default="a peaceful lake at sunset, cinematic")
    args = ap.parse_args()

    ep = os.environ.get("VAST_ENDPOINT_ID") or sys.exit("VAST_ENDPOINT_ID required")
    key = os.environ.get("VAST_API_KEY") or sys.exit("VAST_API_KEY required")

    r = route(ep, key)
    worker_url = r.get("url") or sys.exit("no `url` in /route/ response")
    jwt = r.get("worker_jwt") or r.get("jwt") or r.get("token") or sys.exit("no JWT in /route/ response")

    payload = {
        "prompt": args.prompt,
        "quality": args.quality,
        "duration_sec": args.duration,
        "steps": args.steps,
        "seed": 42,
    }

    if args.mode == "sync":
        result = run_sync(worker_url, jwt, payload)
        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
        return

    # async
    webhook_url = os.environ.get("WEBHOOK_URL")
    webhook_secret = os.environ.get("WEBHOOK_SECRET")
    if not webhook_url:
        sys.exit("--mode=async requires WEBHOOK_URL env (use https://webhook.site for a quick listener)")

    resp = run_async(worker_url, jwt, payload, webhook_url, webhook_secret)
    if not resp.get("async"):
        print("worker did not enter async mode — response:", resp)
        sys.exit(1)
    result_url = resp["result_url"]
    print(f"job_id={resp['job_id']}  ETA={resp['estimated_seconds']}s")
    print("waiting for completion via S3 polling (webhook should fire in parallel)...")
    out = poll_result_url(result_url, max_wait=1800)
    if out:
        print(json.dumps(out, indent=2, ensure_ascii=False)[:2000])
    else:
        print("(timeout)")


if __name__ == "__main__":
    main()
