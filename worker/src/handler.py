"""RunPod Serverless handler — LTX-2.3 ComfyUI worker with typed API.

Input schema (one of prompt/first_frame_url is required):
    {
      "prompt": str,              required if no first_frame_url; runs t2v
      "negative_prompt": str,     optional
      "quality": "sd"|"hd"|"fullhd",   default "hd"
      "aspect_ratio": "9:16"|"16:9",   default "9:16"
      "duration_sec": 1..20,           default 5
      "first_frame_url": str,     optional; if set → i2v mode
      "last_frame_url": str,      optional; requires first_frame_url
      "audio_url": str,           optional; mp3/wav/etc by URL. When set → custom-audio
                                  lip-sync: the speech drives the lips and is the output
                                  soundtrack (first_frame_url = the face, optional).
      "reference_image_urls": [str],  optional; 1-4 subject images by URL. When set →
                                  Multiple-Subject-Reference mode: identity-preserving
                                  generation via IC-LoRA guide conditioning. Mutually
                                  exclusive with audio_url.
      "background_image_url": str,    optional; MSR background plate. Defaults to
                                  first_frame_url, else the first reference image.
      "seed": int                 optional; default 42
    }
"""
from __future__ import annotations

import io
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

import requests
import runpod
import websocket
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from workflow_builder import build as build_workflow, _num_frames, FPS
from s3_upload import upload_and_presign, upload_bytes

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_URL = f"http://{COMFY_HOST}"
UA = "ltx-worker/0.3.0"

ALLOWED_QUALITY = {"sd", "hd", "fullhd"}
ALLOWED_AR = {"9:16", "16:9"}
# Audio VAE input rate for LTX-2.3; the encode node resamples internally, but
# normalising here keeps the uploaded file small and predictable.
AUDIO_SR = 48000


def _log(job_id: str, phase: str, **kw) -> None:
    extras = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[handler] {job_id} phase={phase} {extras}".rstrip(), flush=True)


# Node ID → (stage_idx, stage_name). Stages are the visible chunks of work; the
# log emits one banner per stage transition, so a glance at the log answers
# "where is the pipeline right now". Updated when workflow_template_api.json
# changes; sub-second nodes between samplers are silently grouped into setup.
_STAGES = [
    ("1/8 encoders",     ["4960", "4010", "5012"]),
    ("2/8 prompt",       ["5001", "2483", "2612", "1241"]),
    ("3/8 setup",        ["2004", "4981", "3336", "3059", "3980", "3159",
                          "4528", "3940", "4977", "4978", "4979", "4974",
                          "5030", "5027", "5031"]),
    ("4/8 sample s1a",   ["4968", "5020", "5029", "5021"]),
    ("5/8 sample s1b",   ["5026", "5033", "5032", "5028"]),
    ("6/8 upscale",      ["5025", "5046", "5044", "5043", "5040"]),
    ("7/8 sample s2",    ["5015", "5054", "5051", "5041", "5042"]),
    ("8/8 decode+save",  ["5045", "5050", "5039", "5038", "5055"]),
]
_STAGE_BY_NODE: dict[str, tuple[int, str]] = {}
for _i, (_name, _ids) in enumerate(_STAGES):
    for _nid in _ids:
        _STAGE_BY_NODE[_nid] = (_i, _name)
_TOTAL_STAGES = len(_STAGES)

# Wall-time estimates (sec/frame) measured on 5090 with v0.3.0 pipeline. Used
# only for the start banner's `eta=Ns` hint — it is not enforced anywhere.
_SEC_PER_FRAME = {"sd": 0.39, "hd": 0.64, "fullhd": 1.24}
# Cold-start (Gemma+text encoder + CLIP load + first sampler kernel compile)
# adds a fixed ~80s on a freshly spawned worker — flashboot warm runs skip it.
_COLD_START_SEC = 80


def _eta_seconds(quality: str, num_frames: int) -> int:
    return int(_SEC_PER_FRAME.get(quality, 1.0) * max(num_frames, 1) + _COLD_START_SEC)


def _hms(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m{s % 60:02d}s"


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


def _comfy_interrupt() -> None:
    """Best-effort cancel of the running ComfyUI prompt — fires on watchdog
    timeout so a hung sampler doesn't keep the worker tied up."""
    try:
        requests.post(f"{COMFY_URL}/interrupt", timeout=5)
    except Exception:
        pass


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

    # Input audio (by URL, like first_frame_url). When present, the request runs
    # the custom-audio lip-sync path: the speech drives the lips and is the output
    # soundtrack. A first frame (first_frame_url / frames[0]) acts as the face.
    audio_url = (i.get("audio_url") or "").strip()

    # Multiple-Subject-Reference: 1-4 subject images (LiconMSR only exposes 4
    # numbered slots) + a background plate. Falls back to first_frame_url, else
    # the first reference image, when background_image_url isn't given.
    reference_image_urls = i.get("reference_image_urls") or []
    if not isinstance(reference_image_urls, list):
        raise ValueError("'reference_image_urls' must be a list of URLs")
    reference_image_urls = [str(u).strip() for u in reference_image_urls if str(u).strip()]
    if reference_image_urls:
        if not 1 <= len(reference_image_urls) <= 4:
            raise ValueError("'reference_image_urls' must have between 1 and 4 images")
        if audio_url:
            raise ValueError("'reference_image_urls' and 'audio_url' are mutually exclusive")
        background_image_url = (i.get("background_image_url") or "").strip()
        if not background_image_url:
            background_image_url = (frames[0]["url"] if frames else reference_image_urls[0])
    else:
        background_image_url = ""

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
        "audio_url": audio_url,
        "reference_image_urls": reference_image_urls,
        "background_image_url": background_image_url,
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


def _video_duration_sec(duration_sec: float) -> float:
    """Exact playback length of the rendered clip — the builder rounds the
    requested duration to a valid 8k+1 frame count, so trim audio to match."""
    return round((_num_frames(duration_sec) - 1) / FPS, 3)


def _fetch_and_upload_audio(url: str, duration_sec: float, job_id: str) -> tuple[str, Path]:
    """Download the input audio, trim it to the start `duration_sec` (padding
    with silence if shorter — "take the beginning so the length fits"), transcode
    to a clean 48 kHz stereo wav, and upload to ComfyUI input/. Returns
    (comfy_name, local_wav) — the ComfyUI filename for LoadAudio + the local wav
    (run_pipeline mirrors it to the S3 exp48h/ prefix for auto-cleanup)."""
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    src = Path("/tmp") / f"{job_id}_audio_src"
    src.write_bytes(r.content)
    wav = Path("/tmp") / f"{job_id}_audio.wav"
    # apad + atrim guarantees exactly duration_sec of audio regardless of the
    # source being shorter or longer; ffmpeg autodetects the input container.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-af", f"apad,atrim=0:{duration_sec}",
         "-ac", "2", "-ar", str(AUDIO_SR), str(wav)],
        check=True, timeout=120,
    )
    name = f"ltx_audio_{uuid.uuid4().hex}.wav"
    with open(wav, "rb") as fh:
        up = requests.post(f"{COMFY_URL}/upload/image",
                           files={"image": (name, fh, "audio/wav")},
                           data={"type": "input"}, timeout=60)
    up.raise_for_status()
    return up.json()["name"], wav


def _queue(wf: dict, client_id: str) -> str:
    """Submit prompt to ComfyUI /prompt. Light retry on 5xx + connection
    errors — covers the brief window after ComfyUI's HTTP server binds but
    before custom-node imports finish, where requests can transiently get
    503/connection-refused. Fail-fast on 4xx (those are payload errors and
    won't get better with retries)."""
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(f"{COMFY_URL}/prompt",
                              json={"prompt": wf, "client_id": client_id},
                              headers={"User-Agent": UA}, timeout=30)
            if r.ok:
                return r.json()["prompt_id"]
            if 400 <= r.status_code < 500:
                # Validation error in the workflow — don't retry, the next
                # attempt will see the same broken JSON.
                raise RuntimeError(f"queue failed (4xx): {r.status_code} {r.text[:2000]}")
            last_err = f"HTTP {r.status_code}: {r.text[:500]}"
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {str(e)[:300]}"
        if attempt < 2:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"queue failed after 3 attempts: {last_err}")


def _ws_listen(prompt_id: str, client_id: str, wf: dict,
               t0: float, job_id: str, state: dict) -> None:
    """Stream ComfyUI execution events into `state`. Runs on a daemon thread
    while `_poll` keeps the runpod heartbeat alive on the main thread.

    Emits compact, glance-readable logs:
      - `phase=stage_start` once per pipeline stage transition (8 total)
      - `phase=node_long` only when an individual node took >3s
      - `phase=sampler` per ks-step with per-step time + node ETA
    Trivial sub-second nodes between samplers do NOT generate per-node lines."""
    try:
        ws = websocket.WebSocket()
        ws.settimeout(10)
        ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}", timeout=15)
    except Exception as e:
        _log(job_id, "ws_connect_failed", err=f"{type(e).__name__}: {str(e)[:200]}")
        return
    state["ws_connected"] = True

    cur_node = None
    cur_class = None
    cur_t0 = t0
    cur_stage_idx: Optional[int] = None
    stage_t0 = t0

    last_progress_log = 0.0
    last_step_seen = 0
    last_step_t = t0

    try:
        while not state.get("done"):
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                _log(job_id, "ws_recv_err", err=f"{type(e).__name__}: {str(e)[:200]}")
                break
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue
            typ = data.get("type")
            d = data.get("data") or {}
            # Filter to our prompt. status/queue events have prompt_id=None.
            if d.get("prompt_id") not in (prompt_id, None):
                continue

            if typ == "executing":
                nid = d.get("node")
                now = time.time()

                # Close-out for the previous node — only log if it was slow.
                if cur_node is not None:
                    took = now - cur_t0
                    if took > 3.0:
                        _log(job_id, "node_long",
                             node=cur_node, cls=cur_class,
                             took=f"{took:.1f}s",
                             elapsed=_hms(now - t0))

                if nid is None and d.get("prompt_id") == prompt_id:
                    if cur_stage_idx is not None:
                        _log(job_id, "stage_done",
                             stage=_STAGES[cur_stage_idx][0],
                             stage_took=f"{now - stage_t0:.1f}s",
                             elapsed=_hms(now - t0))
                    state["done"] = True
                    _log(job_id, "ws_finish", elapsed=_hms(now - t0))
                    break

                if nid is not None:
                    cur_node = nid
                    cur_t0 = now
                    cur_class = wf.get(nid, {}).get("class_type", "?")
                    state["current_node"] = nid
                    state["current_class"] = cur_class

                    new_stage = _STAGE_BY_NODE.get(nid, (-1, "?/8 unknown"))
                    if new_stage[0] != cur_stage_idx:
                        if cur_stage_idx is not None:
                            _log(job_id, "stage_done",
                                 stage=_STAGES[cur_stage_idx][0],
                                 stage_took=f"{now - stage_t0:.1f}s",
                                 elapsed=_hms(now - t0))
                        cur_stage_idx = new_stage[0]
                        stage_t0 = now
                        last_step_seen = 0
                        last_step_t = now
                        _log(job_id, "stage_start",
                             stage=new_stage[1],
                             cls=cur_class,
                             elapsed=_hms(now - t0))
            elif typ == "progress":
                now = time.time()
                val = d.get("value") or 0
                mx = d.get("max") or 1
                # Per-step timing — useful for long samplers. Compute step time
                # off the previous progress event (or stage start for step 1).
                if val != last_step_seen:
                    step_dt = now - last_step_t
                    last_step_t = now
                    last_step_seen = val
                    remaining = max(0, mx - val)
                    eta_node = step_dt * remaining if val > 0 else None
                    eta_str = f" eta_node={_hms(eta_node)}" if eta_node else ""
                    if now - last_progress_log > 5.0 or val == mx:
                        _log(job_id, "sampler",
                             node=cur_node, cls=cur_class,
                             step=f"{val}/{mx}",
                             step_dt=f"{step_dt:.1f}s",
                             elapsed=_hms(now - t0) + eta_str)
                        last_progress_log = now
            elif typ == "execution_error":
                state["error"] = (
                    f"{d.get('exception_type')}: {d.get('exception_message')}"
                )
                state["done"] = True
                _log(job_id, "ws_error",
                     err=state["error"][:400],
                     elapsed=_hms(time.time() - t0))
                break
    finally:
        try:
            ws.close()
        except Exception:
            pass


def _poll(prompt_id: str, client_id: str, wf: dict,
          timeout_s: int, job_id: str, *, progress_cb=None) -> dict:
    """Wait for ComfyUI to finish. WS thread streams per-node events; this
    main loop pumps an optional progress callback every 3s. On RunPod the
    callback wraps runpod.progress_update — without it RunPod re-queues the
    job as `-e2` after ~60s. On vast/PyWorker the callback updates the
    JobState so GET /status reflects progress.

    Returns when WS reports done (success or error), then fetches the final
    /history record. Falls back to /history-only polling if WS dies."""
    t0 = time.time()
    state = {
        "done": False, "error": None,
        "current_node": None, "current_class": None,
        "ws_connected": False,
    }
    ws_thread = threading.Thread(
        target=_ws_listen,
        args=(prompt_id, client_id, wf, t0, job_id, state),
        daemon=True,
    )
    ws_thread.start()

    # When WS is alive it produces all the human-readable progress; the main
    # loop's only job then is to pump runpod.progress_update silently so the
    # control plane doesn't kill the worker. The fallback `still_running` line
    # only fires when WS is dead AND no node event has been seen for 60s.
    last_status_log = 0.0
    while time.time() - t0 < timeout_s:
        time.sleep(3)
        elapsed = time.time() - t0
        if progress_cb is not None:
            frac = 0.10 + 0.80 * min(1.0, elapsed / max(timeout_s, 1))
            try:
                progress_cb(frac)
            except Exception:
                pass

        if state["done"]:
            if state["error"]:
                raise RuntimeError(f"comfy error: {state['error']}")
            try:
                r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=15)
                r.raise_for_status()
                h = r.json()
            except Exception as e:
                _log(job_id, "history_after_finish_err", err=str(e)[:120])
                continue
            if prompt_id in h:
                rec = h[prompt_id]
                st = rec.get("status", {})
                _log(job_id, "comfy_completed",
                     elapsed=_hms(elapsed),
                     status_str=st.get("status_str", "?"),
                     output_nodes=len((rec.get("outputs") or {})))
                return rec

        # Fallback path when WS never connected — poll /history directly.
        if not state["ws_connected"]:
            try:
                r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=15)
                r.raise_for_status()
                h = r.json()
                if prompt_id in h:
                    rec = h[prompt_id]
                    st = rec.get("status", {})
                    if st.get("completed"):
                        _log(job_id, "comfy_completed_fallback",
                             elapsed=_hms(elapsed),
                             status_str=st.get("status_str", "?"))
                        return rec
                    if st.get("status_str") == "error":
                        raise RuntimeError(f"comfy error: {json.dumps(st)[:3000]}")
            except RuntimeError:
                raise
            except Exception:
                pass

        # Heartbeat log only when WS is dead — otherwise it just duplicates the
        # per-stage banners. 60s cadence is enough that RunPod's log viewer
        # confirms liveness without burying the useful events.
        if not state["ws_connected"] and elapsed - last_status_log >= 60:
            _log(job_id, "still_running",
                 elapsed=_hms(elapsed),
                 ws="off",
                 cur_node=state["current_node"],
                 cur_class=state["current_class"])
            last_status_log = elapsed

    _log(job_id, "watchdog_timeout", elapsed=f"{timeout_s}s", action="interrupt")
    _comfy_interrupt()
    raise TimeoutError(f"comfy did not finish within {timeout_s}s for prompt {prompt_id}")


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


def run_pipeline(p: dict, job_id: str, *,
                 progress_cb=None,
                 timeout_s: Optional[int] = None) -> dict:
    """Pure pipeline — validated input dict + job_id → result dict. No
    framework coupling (no `event`, no runpod). The two callers are:
      - `handler(event)` for RunPod (wraps progress_cb around runpod.progress_update)
      - `pyworker` for vast (wraps progress_cb around the JobState store)

    Logs `start`, the per-stage banners (via _poll's WS listener), and either
    `done` on success or `failed` then re-raise on exception.

    Returns the same shape that handler() always returned:
        { video_url, thumbnail_url, elapsed_sec,
          width, height, num_frames, duration_sec, fps, mode, quality }
    """
    cb = progress_cb if progress_cb is not None else (lambda _: None)
    if timeout_s is None:
        timeout_s = int(os.environ.get("JOB_TIMEOUT_S", "1500"))
    t0 = time.time()
    _log(job_id, "start",
         duration=f"{p['duration_sec']}s",
         res=p["quality"],
         ar=p["aspect_ratio"],
         mode=("msr" if p["reference_image_urls"] else "i2v" if p["frames"] else "t2v"),
         keyframes=len(p["frames"]),
         audio=bool(p["audio_url"]),
         reference_images=len(p["reference_image_urls"]),
         steps=p["steps"],
         seed=p["seed"])
    try:
        _wait_comfy_ready()
        _log(job_id, "comfy_ready")
        cb(0.02)

        # Upload every (user-supplied) frame image → ComfyUI-side filename
        for f in p["frames"]:
            if "name" not in f:
                f["name"] = _fetch_and_upload_image(f["url"])
                _log(job_id, "frame_uploaded", url=f.get("url","")[:60], idx=f.get("frame_idx"))

        # Input audio → custom-audio lip-sync path (Director). Trim to the exact
        # clip length and upload for the timeline. The dummy png is only needed
        # by the non-audio t2v template.
        audio_name = None
        if p["audio_url"]:
            target = _video_duration_sec(p["duration_sec"])
            audio_name, audio_wav = _fetch_and_upload_audio(p["audio_url"], target, job_id)
            _log(job_id, "audio_prepared", secs=target,
                 url=p["audio_url"][:60], comfy_name=audio_name)
            # Mirror the processed wav to the bucket's exp48h/ prefix (lifecycle
            # rule auto-deletes after 48h) so the job's input audio is retrievable
            # without piling up. Non-fatal: never block the render on this.
            try:
                upload_and_presign(
                    audio_wav, f"exp48h/{job_id}/audio.wav",
                    expires_sec=int(os.environ.get("PRESIGN_TTL", "3600")),
                    content_type="audio/wav")
                _log(job_id, "audio_archived", key=f"exp48h/{job_id}/audio.wav")
            except Exception as ae:
                _log(job_id, "audio_archive_failed", err=f"{type(ae).__name__}: {str(ae)[:120]}")

        # Multiple-Subject-Reference: upload subject images + background plate.
        reference_names = None
        background_name = None
        if p["reference_image_urls"]:
            reference_names = [_fetch_and_upload_image(u) for u in p["reference_image_urls"]]
            background_name = _fetch_and_upload_image(p["background_image_url"])
            _log(job_id, "reference_images_uploaded", count=len(reference_names))

        t2v_dummy = None if (p["frames"] or audio_name or reference_names) else _upload_dummy_png()
        cb(0.05)

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
            audio_name=audio_name,
            reference_names=reference_names,
            background_name=background_name,
        )
        eta = _eta_seconds(meta["quality"], meta["num_frames"])
        _log(job_id, "gen",
             output=f"{meta['width']}x{meta['height']}",
             frames=meta["num_frames"],
             dur=f"{meta['duration_sec']}s",
             fps=meta["fps"],
             mode=meta["mode"],
             stages=_TOTAL_STAGES,
             nodes=len(wf),
             eta=_hms(eta))
        cb(0.08)

        client_id = uuid.uuid4().hex
        prompt_id = _queue(wf, client_id)
        _log(job_id, "queued", comfy_prompt_id=prompt_id, client_id=client_id[:8])
        rec = _poll(prompt_id, client_id, wf,
                    timeout_s=timeout_s,
                    job_id=job_id,
                    progress_cb=cb)
        cb(0.92)

        mp4 = _download_video(rec)
        _log(job_id, "video_downloaded", bytes=len(mp4))
        # Flat key layout: run/result/{job_id}/<file>. No date prefix —
        # client looks up artifacts deterministically from the job_id.
        prefix = f"run/result/{job_id}"
        video_key = f"{prefix}/video.mp4"
        tmp = Path("/tmp") / f"{job_id}.mp4"
        tmp.write_bytes(mp4)
        ttl = int(os.environ.get("PRESIGN_TTL", "3600"))
        url = upload_and_presign(tmp, video_key, expires_sec=ttl, content_type="video/mp4")
        _log(job_id, "video_uploaded", key=video_key, url_len=len(url))

        thumb_url = None
        thumb_path = Path("/tmp") / f"{job_id}.jpg"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                 "-vframes", "1", "-q:v", "3", str(thumb_path)],
                check=True, timeout=30,
            )
            thumb_key = f"{prefix}/thumb.jpg"
            thumb_url = upload_and_presign(
                thumb_path, thumb_key, expires_sec=ttl, content_type="image/jpeg",
            )
            _log(job_id, "thumb_uploaded", key=thumb_key)
        except Exception as te:
            _log(job_id, "thumb_failed", err=f"{type(te).__name__}: {str(te)[:120]}")

        cb(1.0)
        elapsed = round(time.time() - t0, 2)
        response = {
            "video_url": url,
            "thumbnail_url": thumb_url,
            "elapsed_sec": elapsed,
            **meta,
        }

        # Sidecar: write result.json next to the video. Survives RunPod's
        # `Failed to return job results | 400` bug — when control-plane
        # silently drops the worker→client handoff, the client can still
        # GET s3://{bucket}/run/result/{job_id}/result.json by jobId.
        # Vast doesn't have this bug but the sidecar is still useful as a
        # canonical "where did the video go" pointer.
        try:
            upload_bytes(
                data=json.dumps(response, ensure_ascii=False).encode("utf-8"),
                key=f"{prefix}/result.json",
                content_type="application/json",
            )
            _log(job_id, "result_sidecar_uploaded")
        except Exception as e:
            _log(job_id, "result_sidecar_failed", err=str(e)[:200])

        _log(job_id, "done",
             total=_hms(elapsed),
             output=f"{meta['width']}x{meta['height']}",
             frames=meta["num_frames"],
             video_mb=round(len(mp4) / 1024 / 1024, 1))
        return response
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        _log(job_id, "failed", err=f"{type(e).__name__}: {str(e)[:200]}", elapsed=f"{elapsed}s")
        raise


def handler(event):
    """RunPod entrypoint. Validates input, then delegates the heavy lifting
    to run_pipeline(). progress_cb is wired through to runpod.progress_update
    so the control plane sees this worker as alive across long generations."""
    try:
        p = _validate(event.get("input") or {})
    except ValueError as e:
        return {"error": str(e)}

    job_id = event.get("id") or uuid.uuid4().hex
    try:
        return run_pipeline(p, job_id, progress_cb=lambda f: _progress(event, f))
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-3000:],
        }


def _warmup() -> None:
    """Mandatory startup gate. Runs the full pipeline once at the lowest
    setting (t2v sd × 1s × 5 steps) AND uploads the result to S3 before the
    handler starts accepting jobs. Three classes of broken-worker land here:
      - MooseFS hang on first 25 GB read of LTXAVTEModel  → caught
      - GPU/driver-level kernel deadlock                   → caught
      - S3 unreachable from this pod (egress, creds, dns)  → caught

    Failure → sys.exit(1) so RunPod marks the worker unhealthy and recycles
    it (and likewise the vast benchmark fails so the autoscaler retries).
    Better to churn one bad worker than to serve degraded jobs from it.

    Side-effect: warm workers' RAM and VRAM are pre-staged with all required
    models, so the first real job pays no cold-load cost. RunPod's flashboot
    snapshot taken later in the worker's life captures this hot state.

    SKIP_WARMUP=1 bypasses for local dev."""
    if os.environ.get("SKIP_WARMUP") == "1":
        print("[handler] WARMUP skipped (SKIP_WARMUP=1)", flush=True)
        return
    job_id = f"warmup-{uuid.uuid4().hex[:8]}"
    t0 = time.time()
    print(f"[handler] WARMUP starting job_id={job_id}", flush=True)
    try:
        p = _validate({
            "prompt": "warmup", "quality": "sd", "aspect_ratio": "16:9",
            "duration_sec": 1.0, "seed": 42, "steps": 5,
        })
        run_pipeline(
            p, job_id,
            progress_cb=None,
            timeout_s=int(os.environ.get("WARMUP_TIMEOUT_S", "900")),
        )
        elapsed = round(time.time() - t0, 1)
        print(f"[handler] WARMUP ok elapsed={_hms(elapsed)}", flush=True)
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        print(f"[handler] WARMUP FAILED elapsed={_hms(elapsed)} "
              f"err={type(e).__name__}: {str(e)[:400]}",
              flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    _warmup()
    runpod.serverless.start({"handler": handler})
