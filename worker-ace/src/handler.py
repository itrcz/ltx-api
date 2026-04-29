"""RunPod Serverless handler — ACE-Step 1.5 XL-SFT ComfyUI worker with typed API.

Input schema:
    {
      "prompt":         str   (required)        — style/genre tags
      "lyrics":         str   (required)        — song text with [Verse]/[Chorus] tags
      "bpm":            int   60..220           — default 135
      "key":            str                     — e.g. "C minor", default "C minor"
      "language":       str                     — "ru"|"en"|"auto", default "ru"
      "time_signature": str   "3"|"4"|...       — default "4"
      "duration_sec":   int   30..240           — default 120
      "seed":           int                     — default random

      "mode": str  "turbo"|"fast"|"quality"|"max"  — default "quality"
        turbo:   1 pass,  12 steps, CFG 2, ode  — ~10s,  preview-grade
        fast:    1 pass,  25 steps, CFG 4, sde  — ~30s,  decent
        quality: 4 passes, 50 steps, CFG 6, sde — ~90s, production default
        max:     4 passes, 65 steps, CFG 7, sde — ~150s, max effort

      # advanced — only set when you want to OVERRIDE the mode preset for that field:
      "steps":          int   8..80     (overrides mode)
      "cfg":            float 1..12     (overrides mode)
      "sampler":        str             (overrides mode; "euler"/"er_sde"/"res_2s"/...)
      "scheduler":      str             (overrides mode; "linear_quadratic"/"sgm_uniform"/...)
      "inference":      str   "ode"|"sde" (overrides mode)
    }

Output:
    { audio_url, format, elapsed_sec, ...meta (mode, seed, active_passes, ...) }
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import time
import traceback
import uuid
from pathlib import Path

import requests
import runpod

sys.path.insert(0, str(Path(__file__).parent))
from workflow_builder import (
    build as build_workflow,
    BPM_MIN, BPM_MAX,
    DURATION_MIN, DURATION_MAX, DURATION_DEFAULT,
    STEPS_MIN, STEPS_MAX,
    CFG_MIN, CFG_MAX,
    DEFAULT_KEY, DEFAULT_LANGUAGE, DEFAULT_TIME_SIG,
    NEG_SAMPLERS, MODES, DEFAULT_MODE,
)
from s3_upload import upload_and_presign

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_URL = f"http://{COMFY_HOST}"
UA = "ace-worker/0.1.0"

ALLOWED_LANG = {"ru", "en", "auto", "es", "fr", "de", "it", "ja", "ko", "zh"}
ALLOWED_INFERENCE = {"ode", "sde"}


def _wait_comfy_ready(timeout_s: int = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
            if r.ok:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"ComfyUI not ready at {COMFY_URL} in {timeout_s}s")


def _validate(i: dict) -> dict:
    prompt = (i.get("prompt") or "").strip()
    lyrics = (i.get("lyrics") or "").strip()
    if not prompt:
        raise ValueError("'prompt' (style/tags) is required")
    if not lyrics:
        raise ValueError("'lyrics' is required (use [Verse]/[Chorus] tags + blank lines between sections)")

    try:
        bpm = int(i.get("bpm", 135))
    except Exception:
        raise ValueError("bpm must be an integer")
    if not BPM_MIN <= bpm <= BPM_MAX:
        raise ValueError(f"bpm must be in [{BPM_MIN}, {BPM_MAX}]")

    try:
        duration = int(i.get("duration_sec", DURATION_DEFAULT))
    except Exception:
        raise ValueError("duration_sec must be an integer")
    if not DURATION_MIN <= duration <= DURATION_MAX:
        raise ValueError(f"duration_sec must be in [{DURATION_MIN}, {DURATION_MAX}]")

    key = (i.get("key") or DEFAULT_KEY).strip()
    language = (i.get("language") or DEFAULT_LANGUAGE).strip().lower()
    if language not in ALLOWED_LANG:
        raise ValueError(f"language must be one of {sorted(ALLOWED_LANG)}")
    time_sig = str(i.get("time_signature") or DEFAULT_TIME_SIG).strip()

    raw_seed = i.get("seed")
    if raw_seed is None or raw_seed == "":
        seed = secrets.randbelow(2**32)
    else:
        try:
            seed = int(raw_seed)
        except Exception:
            raise ValueError("seed must be an integer")

    mode = (i.get("mode") or DEFAULT_MODE).strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")

    # Advanced overrides — None means "use mode preset". Pass through as-is.
    def _opt_int(name, lo, hi):
        v = i.get(name)
        if v is None or v == "": return None
        try: vi = int(v)
        except Exception: raise ValueError(f"{name} must be an integer")
        if not lo <= vi <= hi: raise ValueError(f"{name} must be in [{lo}, {hi}]")
        return vi
    def _opt_float(name, lo, hi):
        v = i.get(name)
        if v is None or v == "": return None
        try: vf = float(v)
        except Exception: raise ValueError(f"{name} must be a number")
        if not lo <= vf <= hi: raise ValueError(f"{name} must be in [{lo}, {hi}]")
        return vf

    steps = _opt_int("steps", STEPS_MIN, STEPS_MAX)
    cfg = _opt_float("cfg", CFG_MIN, CFG_MAX)

    sampler_raw = i.get("sampler")
    sampler = sampler_raw.strip() if isinstance(sampler_raw, str) and sampler_raw.strip() else None
    if sampler and sampler in NEG_SAMPLERS:
        raise ValueError(f"sampler {sampler!r} is known-bad on ACE-Step; pick another")
    scheduler_raw = i.get("scheduler")
    scheduler = scheduler_raw.strip() if isinstance(scheduler_raw, str) and scheduler_raw.strip() else None
    inference_raw = i.get("inference")
    inference = inference_raw.strip().lower() if isinstance(inference_raw, str) and inference_raw.strip() else None
    if inference is not None and inference not in ALLOWED_INFERENCE:
        raise ValueError(f"inference must be one of {sorted(ALLOWED_INFERENCE)}")

    return {
        "prompt": prompt,
        "lyrics": lyrics,
        "bpm": bpm,
        "key": key,
        "language": language,
        "time_signature": time_sig,
        "duration_sec": duration,
        "seed": seed,
        "mode": mode,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": scheduler,
        "inference": inference,
    }


def _queue(wf: dict) -> str:
    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": wf, "client_id": uuid.uuid4().hex},
        headers={"User-Agent": UA},
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"queue failed: {r.status_code} {r.text[:2000]}")
    return r.json()["prompt_id"]


def _poll(prompt_id: str, timeout_s: int) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(3)
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=15)
        r.raise_for_status()
        h = r.json()
        if prompt_id in h:
            rec = h[prompt_id]
            st = rec.get("status", {})
            if st.get("completed"):
                return rec
            if st.get("status_str") == "error":
                raise RuntimeError(f"comfy error: {json.dumps(st)[:3000]}")
    raise TimeoutError(f"comfy did not finish within {timeout_s}s")


def _download_audio(rec: dict) -> tuple[bytes, str]:
    """Find the audio file in ComfyUI's outputs and return (bytes, ext)."""
    outputs = rec.get("outputs", {})
    for _, out in outputs.items():
        for f in (out.get("audio") or []) + (out.get("files") or []):
            name = f.get("filename", "")
            if any(name.lower().endswith(ext) for ext in (".mp3", ".flac", ".wav", ".ogg")):
                r = requests.get(
                    f"{COMFY_URL}/view",
                    params={
                        "filename": name,
                        "type": f.get("type", "output"),
                        "subfolder": f.get("subfolder", ""),
                    },
                    timeout=300,
                )
                r.raise_for_status()
                ext = name.rsplit(".", 1)[-1].lower()
                return r.content, ext
    raise RuntimeError(f"no audio in outputs: {json.dumps(outputs)[:1000]}")


def _progress(event, frac: float) -> None:
    try:
        runpod.serverless.progress_update(event, {"progress": round(frac, 3)})
    except Exception:
        pass


def handler(event):
    t0 = time.time()
    try:
        p = _validate(event.get("input") or {})
    except ValueError as e:
        return {"error": str(e)}

    job_id = event.get("id") or uuid.uuid4().hex
    try:
        _wait_comfy_ready()
        _progress(event, 0.05)

        wf, meta = build_workflow(
            prompt=p["prompt"],
            lyrics=p["lyrics"],
            bpm=p["bpm"],
            key=p["key"],
            language=p["language"],
            time_signature=p["time_signature"],
            duration_sec=p["duration_sec"],
            seed=p["seed"],
            mode=p["mode"],
            steps=p["steps"],
            cfg=p["cfg"],
            sampler=p["sampler"],
            scheduler=p["scheduler"],
            inference=p["inference"],
            job_id=job_id,
        )
        _progress(event, 0.10)

        prompt_id = _queue(wf)
        rec = _poll(prompt_id, timeout_s=int(os.environ.get("JOB_TIMEOUT_S", "1800")))
        _progress(event, 0.92)

        audio_bytes, ext = _download_audio(rec)
        date_prefix = time.strftime("%Y/%m/%d")
        key = f"ace/{date_prefix}/{job_id}.{ext}"
        tmp = Path("/tmp") / f"{job_id}.{ext}"
        tmp.write_bytes(audio_bytes)

        ttl = int(os.environ.get("PRESIGN_TTL", "3600"))
        ctype = {"mp3": "audio/mpeg", "flac": "audio/flac",
                 "wav": "audio/wav", "ogg": "audio/ogg"}.get(ext, "application/octet-stream")
        url = upload_and_presign(tmp, key, expires_sec=ttl, content_type=ctype)
        _progress(event, 1.0)

        return {
            "audio_url": url,
            "format": ext,
            "elapsed_sec": round(time.time() - t0, 2),
            **meta,
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-3000:],
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
