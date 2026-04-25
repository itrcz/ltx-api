"""RunPod Serverless handler — LTX-2.3 ComfyUI worker with typed API.

Input schema (one of prompt/first_frame_url is required):
    {
      "prompt": str,              optional; required if no first_frame_url
      "negative_prompt": str,     optional
      "quality": "sd"|"hd"|"fullhd",   default "hd"
      "aspect_ratio": "9:16"|"16:9",   default "9:16"
      "duration_sec": 1..20,           default 5
      "first_frame_url": str,     optional; if set → i2v mode
      "last_frame_url": str,      optional; requires first_frame_url
      "seed": int                 optional; default 42
    }
"""
from __future__ import annotations

import base64
import io
import json
import os
import secrets
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import requests
import runpod
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from workflow_builder import build as build_workflow
from s3_upload import upload_and_presign

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_URL = f"http://{COMFY_HOST}"
UA = "ltx-worker/0.2.8"

ALLOWED_QUALITY = {"sd", "hd", "fullhd"}
ALLOWED_AR = {"9:16", "16:9"}


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

    # Canonical frames list. Either `frames` directly OR the sugar
    # first_frame_url / last_frame_url. `frames` wins if given.
    frames = i.get("frames")
    if frames is None:
        frames = []
        if i.get("first_frame_url"):
            frames.append({"url": i["first_frame_url"], "frame_idx": 0, "strength": 1.0})
        if i.get("last_frame_url"):
            frames.append({"url": i["last_frame_url"], "frame_idx": -1, "strength": 0.5})

    if not isinstance(frames, list):
        raise ValueError("'frames' must be a list of {url, frame_idx, strength}")
    for idx, f in enumerate(frames):
        if not isinstance(f, dict) or not f.get("url"):
            raise ValueError(f"frames[{idx}] must be an object with 'url'")
        f.setdefault("frame_idx", 0 if idx == 0 else -1)
        f.setdefault("strength", 1.0 if f["frame_idx"] == 0 else 0.5)
        try:
            f["strength"] = float(f["strength"])
        except Exception:
            raise ValueError(f"frames[{idx}].strength must be a number")
        if not 0.0 <= f["strength"] <= 1.0:
            raise ValueError(f"frames[{idx}].strength must be in [0, 1]")
        try:
            f["frame_idx"] = int(f["frame_idx"])
        except Exception:
            raise ValueError(f"frames[{idx}].frame_idx must be an integer")

    if not prompt and not frames:
        raise ValueError("either 'prompt' or at least one image (frames / first_frame_url) must be provided")

    quality = i.get("quality", "hd")
    if quality not in ALLOWED_QUALITY:
        raise ValueError(f"quality must be one of {sorted(ALLOWED_QUALITY)}")

    ar = i.get("aspect_ratio", "9:16")
    if ar not in ALLOWED_AR:
        raise ValueError(f"aspect_ratio must be one of {sorted(ALLOWED_AR)}")

    try:
        duration = float(i.get("duration_sec", 5))
    except Exception:
        raise ValueError("duration_sec must be a number")
    if not 1 <= duration <= 20:
        raise ValueError("duration_sec must be between 1 and 20")

    raw_seed = i.get("seed")
    if raw_seed is None or raw_seed == "":
        seed = secrets.randbelow(2**32)
    else:
        try:
            seed = int(raw_seed)
        except Exception:
            raise ValueError("seed must be an integer")

    try:
        steps = int(i.get("steps", 8))
    except Exception:
        raise ValueError("steps must be an integer")
    if not 5 <= steps <= 30:
        raise ValueError("steps must be between 5 and 30")

    lora_strength_raw = i.get("lora_strength")
    if lora_strength_raw is None or lora_strength_raw == "":
        lora_strength = None
    else:
        try:
            lora_strength = float(lora_strength_raw)
        except Exception:
            raise ValueError("lora_strength must be a number")
        if not 0.0 <= lora_strength <= 1.0:
            raise ValueError("lora_strength must be in [0, 1]")

    no_tile_vae_raw = i.get("no_tile_vae")
    if no_tile_vae_raw is None:
        no_tile_vae = None
    else:
        no_tile_vae = bool(no_tile_vae_raw)

    return {
        "prompt": prompt,
        "negative_prompt": (i.get("negative_prompt") or "").strip(),
        "quality": quality,
        "aspect_ratio": ar,
        "duration_sec": duration,
        "frames": frames,
        "seed": seed,
        "steps": steps,
        "lora_strength": lora_strength,
        "no_tile_vae": no_tile_vae,
    }


def _upload_dummy_png() -> str:
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    name = f"ltx_dummy_{uuid.uuid4().hex}.png"
    up = requests.post(f"{COMFY_URL}/upload/image",
                       files={"image": (name, buf, "image/png")},
                       data={"type": "input"}, timeout=30)
    up.raise_for_status()
    return up.json()["name"]


def _upload_png_bytes(data: bytes) -> str:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    name = f"ltx_{uuid.uuid4().hex}.png"
    files = {"image": (name, buf, "image/png")}
    up = requests.post(f"{COMFY_URL}/upload/image",
                       files=files, data={"type": "input"}, timeout=60)
    up.raise_for_status()
    return up.json()["name"]


def _fetch_and_upload_image(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return _upload_png_bytes(r.content)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-2.5-flash-image"


def _generate_first_frame(prompt: str, aspect_ratio: str) -> bytes:
    """Generate a seed image via OpenRouter (Gemini 2.5 Flash Image).

    Returns raw PNG/JPEG bytes. Raises on failure.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not configured")

    orient = "landscape 16:9" if aspect_ratio == "16:9" else "portrait 9:16"
    user_msg = (
        f"Create a single cinematic {orient} photograph depicting this scene:\n"
        f"---\n{prompt}\n---\n\n"
        f"Strict requirements:\n"
        f"- Strictly {orient} aspect ratio.\n"
        f"- Photograph only. Absolutely NO rendered text, letters, captions, "
        f"subtitles, speech bubbles, signs with writing, logos, watermarks, or "
        f"handwriting anywhere in the image.\n"
        f"- If the scene description mentions dialogue, quoted speech, or words "
        f"spoken by characters (e.g. text inside quotation marks), depict ONLY "
        f"the speaker's facial expression, gesture, mouth movement, or action — "
        f"do NOT render the spoken words as visible text.\n"
        f"- Treat any quoted strings in the scene description as metadata about "
        f"what a character is saying, not as text to draw."
    )
    body = {
        "model": OPENROUTER_MODEL,
        "modalities": ["image", "text"],
        "messages": [{"role": "user", "content": user_msg}],
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/itrcz/ltx-api",
        "X-Title": "ltx-api worker",
    }
    r = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=120)
    if not r.ok:
        raise RuntimeError(f"openrouter {r.status_code}: {r.text[:500]}")
    data = r.json()
    try:
        msg = data["choices"][0]["message"]
    except Exception:
        raise RuntimeError(f"openrouter: no choices in response: {json.dumps(data)[:500]}")

    # Gemini returns image as message.images[0].image_url.url (data URL or https URL).
    images = msg.get("images") or []
    for item in images:
        iu = item.get("image_url") or {}
        url = iu.get("url") if isinstance(iu, dict) else None
        if not url:
            continue
        if url.startswith("data:"):
            b64 = url.split(",", 1)[1]
            return base64.b64decode(b64)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content

    raise RuntimeError(f"openrouter: no image in response: {json.dumps(msg)[:500]}")


def _queue(wf: dict) -> str:
    r = requests.post(f"{COMFY_URL}/prompt",
                      json={"prompt": wf, "client_id": uuid.uuid4().hex},
                      headers={"User-Agent": UA}, timeout=30)
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


def _download_video(rec: dict) -> bytes:
    outputs = rec.get("outputs", {})
    for _, out in outputs.items():
        files = (out.get("images") or []) + (out.get("videos") or []) + (out.get("gifs") or [])
        for f in files:
            if f.get("filename", "").lower().endswith(".mp4"):
                r = requests.get(f"{COMFY_URL}/view", params={
                    "filename": f["filename"],
                    "type": f.get("type", "output"),
                    "subfolder": f.get("subfolder", ""),
                }, timeout=300)
                r.raise_for_status()
                return r.content
    raise RuntimeError(f"no mp4 in outputs: {json.dumps(outputs)[:1000]}")


def _progress(event, frac: float):
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
        _progress(event, 0.02)

        # If no frames were supplied but we have a prompt, synthesize a first
        # frame with Gemini 2.5 Flash Image (OpenRouter) so we run i2v instead
        # of the fragile t2v dummy-image path.
        generated_first_frame = False
        if not p["frames"] and p["prompt"]:
            img_bytes = _generate_first_frame(p["prompt"], p["aspect_ratio"])
            name = _upload_png_bytes(img_bytes)
            p["frames"] = [{
                "url": "<generated>",
                "frame_idx": 0,
                "strength": 1.0,
                "name": name,
            }]
            generated_first_frame = True

        # Upload every (user-supplied) frame image → ComfyUI-side filename
        for f in p["frames"]:
            if "name" not in f:
                f["name"] = _fetch_and_upload_image(f["url"])
        t2v_dummy = None if p["frames"] else _upload_dummy_png()
        _progress(event, 0.05)

        wf, meta = build_workflow(
            prompt=p["prompt"],
            negative_prompt=p["negative_prompt"],
            quality=p["quality"],
            aspect_ratio=p["aspect_ratio"],
            duration_sec=p["duration_sec"],
            seed=p["seed"],
            frames=p["frames"],
            is_i2v=bool(p["frames"]),
            t2v_dummy_name=t2v_dummy,
            steps=p["steps"],
            lora_strength=p["lora_strength"],
            no_tile_vae=p["no_tile_vae"],
        )
        _progress(event, 0.08)

        prompt_id = _queue(wf)
        rec = _poll(prompt_id, timeout_s=int(os.environ.get("JOB_TIMEOUT_S", "1500")))
        _progress(event, 0.92)

        mp4 = _download_video(rec)
        date_prefix = time.strftime('%Y/%m/%d')
        key = f"ltx/{date_prefix}/{job_id}.mp4"
        tmp = Path("/tmp") / f"{job_id}.mp4"
        tmp.write_bytes(mp4)
        ttl = int(os.environ.get("PRESIGN_TTL", "3600"))
        url = upload_and_presign(tmp, key, expires_sec=ttl, content_type="video/mp4")

        thumb_url = None
        thumb_path = Path("/tmp") / f"{job_id}.jpg"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                 "-vframes", "1", "-q:v", "3", str(thumb_path)],
                check=True, timeout=30,
            )
            thumb_key = f"ltx/{date_prefix}/{job_id}.jpg"
            thumb_url = upload_and_presign(
                thumb_path, thumb_key, expires_sec=ttl, content_type="image/jpeg",
            )
        except Exception as te:
            print(f"thumbnail extraction failed: {type(te).__name__}: {te}", flush=True)

        _progress(event, 1.0)

        return {
            "video_url": url,
            "thumbnail_url": thumb_url,
            "generated_first_frame": generated_first_frame,
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
