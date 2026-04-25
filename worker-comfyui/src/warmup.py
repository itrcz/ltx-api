"""Pre-load LTX-2.3 weights into VRAM so RunPod FlashBoot snapshots a hot state.

Run after ComfyUI boots, before handler.py starts serving. Non-fatal: any
failure here is logged and the handler continues normally (cold-path still
works, just slow).

Env:
  WARMUP            — "1"/"0" (default "1"). Disable by setting to "0".
  WARMUP_TIMEOUT_S  — hard cap, default 900.
  COMFY_HOST        — default "127.0.0.1:8188".
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from workflow_builder import build as build_workflow

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_URL = f"http://{COMFY_HOST}"


def _wait_comfy(timeout: int = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
            if r.ok:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"comfy not ready in {timeout}s")


def _upload_dummy() -> str:
    img = Image.new("RGB", (256, 256), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    name = f"warmup_{uuid.uuid4().hex}.png"
    r = requests.post(
        f"{COMFY_URL}/upload/image",
        files={"image": (name, buf, "image/png")},
        data={"type": "input"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["name"]


def run() -> None:
    timeout = int(os.environ.get("WARMUP_TIMEOUT_S", "900"))
    t0 = time.time()

    print("[warmup] waiting for ComfyUI...", flush=True)
    _wait_comfy()

    print("[warmup] building minimal workflow (sd/1s/5 steps t2v, stage-1 480x272)", flush=True)
    dummy = _upload_dummy()
    wf, _meta = build_workflow(
        prompt="warmup",
        negative_prompt="",
        quality="sd",
        aspect_ratio="16:9",
        duration_sec=1,
        seed=42,
        frames=[],
        is_i2v=False,
        t2v_dummy_name=dummy,
        steps=5,
    )
    # Shrink stage-1 latent to the template-native dims so warmup is as light
    # as possible — final output ends up ~960x544 after the x2 upscaler, which
    # is the configuration shipped as LTX-ComfyUI's reference workflow.
    wf["3059"]["inputs"]["width"] = 480
    wf["3059"]["inputs"]["height"] = 272

    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": wf, "client_id": uuid.uuid4().hex},
        timeout=30,
    )
    r.raise_for_status()
    pid = r.json()["prompt_id"]
    print(f"[warmup] queued {pid}, polling (timeout={timeout}s)", flush=True)

    last_log = 0.0
    while time.time() - t0 < timeout:
        time.sleep(5)
        try:
            r = requests.get(f"{COMFY_URL}/history/{pid}", timeout=15)
            r.raise_for_status()
            h = r.json()
        except Exception as e:
            print(f"[warmup] poll error (will retry): {e}", flush=True)
            continue
        if pid in h:
            st = h[pid].get("status", {}) or {}
            if st.get("completed"):
                print(f"[warmup] done in {int(time.time() - t0)}s", flush=True)
                return
            if st.get("status_str") == "error":
                print(f"[warmup] comfy error (non-fatal): {json.dumps(st)[:500]}", flush=True)
                return
        if time.time() - last_log > 30:
            print(f"[warmup] still running, elapsed {int(time.time() - t0)}s", flush=True)
            last_log = time.time()

    print(f"[warmup] timeout after {timeout}s — continuing anyway", flush=True)


if __name__ == "__main__":
    if os.environ.get("WARMUP", "1") != "1":
        print("[warmup] disabled via WARMUP=0, skipping", flush=True)
        sys.exit(0)
    try:
        run()
    except Exception as e:
        print(f"[warmup] failed (non-fatal): {type(e).__name__}: {e}", flush=True)
        sys.exit(0)
