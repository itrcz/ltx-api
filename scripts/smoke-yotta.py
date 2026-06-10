#!/usr/bin/env python3
"""Smoke test for the LTX-2.3 Yotta Labs Serverless endpoint.

Submits a task via Yotta's queue, polls until SUCCESS/FAILED, prints result.

Usage:
    YOTTA_API_KEY=... YOTTA_ENDPOINT_ID=12345 python3 scripts/smoke-yotta.py [--async]

`--async` adds a webhook URL (set WEBHOOK_URL env) so Yotta posts the
result to a listener instead of (or in addition to) polling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

BASE = "https://api.yottalabs.ai"


def submit_task(endpoint_id: int, api_key: str, payload: dict,
                webhook_url: str | None = None) -> str:
    body = {
        "input": payload,
        "processUri": "/run",
        "workerPort": 8000,
    }
    if webhook_url:
        body["webhook"] = webhook_url
    print(f"[{time.strftime('%H:%M:%S')}] POST /v2/serverless/{endpoint_id}/tasks")
    print(f"  input: {json.dumps(payload)[:200]}")
    r = requests.post(
        f"{BASE}/v2/serverless/{endpoint_id}/tasks",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    d = r.json()
    if d.get("code") != 10000:
        sys.exit(f"submit failed: {d}")
    task_id = d["data"]["taskId"]
    print(f"[{time.strftime('%H:%M:%S')}] task_id={task_id}")
    return task_id


def poll_task(endpoint_id: int, task_id: str, api_key: str,
              max_wait: int = 1800) -> dict:
    t0 = time.time()
    last_status = None
    while time.time() - t0 < max_wait:
        r = requests.get(
            f"{BASE}/v2/serverless/{endpoint_id}/tasks/{task_id}",
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()
        if d.get("code") != 10000:
            print(f"  poll err: {d}")
            time.sleep(10)
            continue
        task = d["data"]
        st = task.get("status")
        if st != last_status:
            elapsed = int(time.time() - t0)
            print(f"[{time.strftime('%H:%M:%S')}] +{elapsed}s status={st}")
            last_status = st
        if st in ("SUCCESS", "FAILED"):
            return task
        time.sleep(10)
    sys.exit(f"timeout after {max_wait}s, last status={last_status}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="sd", choices=["sd", "hd", "fullhd"])
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt", default="a peaceful lake at sunset, cinematic")
    ap.add_argument("--async-webhook", action="store_true",
                    help="Set webhook from WEBHOOK_URL env so Yotta callbacks the result")
    args = ap.parse_args()

    api_key = os.environ.get("YOTTA_API_KEY") or sys.exit("YOTTA_API_KEY required")
    endpoint_id = os.environ.get("YOTTA_ENDPOINT_ID")
    if not endpoint_id:
        sys.exit("YOTTA_ENDPOINT_ID required")
    endpoint_id = int(endpoint_id)

    payload = {
        "prompt": args.prompt,
        "quality": args.quality,
        "duration_sec": args.duration,
        "steps": args.steps,
        "seed": 42,
    }
    webhook = os.environ.get("WEBHOOK_URL") if args.async_webhook else None

    task_id = submit_task(endpoint_id, api_key, payload, webhook)
    task = poll_task(endpoint_id, task_id, api_key)

    print("\n=== final task ===")
    print(json.dumps({k: v for k, v in task.items() if k != "output"}, indent=2))
    print("\n=== output ===")
    output = task.get("output")
    if isinstance(output, dict):
        print(json.dumps(output, indent=2, ensure_ascii=False)[:2000])
    else:
        print(output)


if __name__ == "__main__":
    main()
