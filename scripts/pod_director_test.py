#!/usr/bin/env python3
"""Integration test of the PRODUCTION custom-audio lip-sync path (no S3).

Uploads image + audio, then calls the real workflow_builder.build(audio_name=...)
— i.e. the Director + use_custom_audio + VAEDecodeTiled graph that ships — queues
it, and saves the result to /comfyui/output for the pod proxy.

Usage (on the pod):
    POD_PROXY=https://<id>-8188.proxy.runpod.net \
    python3 pod_director_test.py <AUDIO_URL> [IMAGE_URL|none] [dur] [quality]
"""
import sys, os, uuid, subprocess
import requests
from pathlib import Path
sys.path.insert(0, "/")
import handler as h                       # noqa: E402
from workflow_builder import build as build_workflow   # noqa: E402

AUDIO_URL = sys.argv[1]                    # URL or local pod path (starts with /)
IMAGE_URL = sys.argv[2] if len(sys.argv) > 2 else ""
if IMAGE_URL in ("none", "-"):
    IMAGE_URL = ""
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
QUALITY = sys.argv[4] if len(sys.argv) > 4 else "sd"
ASPECT = os.environ.get("ASPECT", "16:9")
AUDIO_START = float(os.environ.get("AUDIO_START", "0"))   # take audio from this offset
PROMPT = os.environ.get("PROMPT", "a person talking to the camera, close up portrait")


def prep_audio(src, start, dur, job):
    """Trim `dur` seconds of audio starting at `start` (local path or URL),
    transcode to 48k stereo wav, upload to ComfyUI input/. Returns the name."""
    if not src.startswith("/"):
        local = f"/tmp/{job}_src"
        r = requests.get(src, timeout=120); r.raise_for_status()
        open(local, "wb").write(r.content); src = local
    wav = f"/tmp/{job}_audio.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start),
                    "-i", src, "-t", str(dur), "-af", "apad",
                    "-ac", "2", "-ar", str(h.AUDIO_SR), wav], check=True, timeout=180)
    name = f"ltx_audio_{uuid.uuid4().hex}.wav"
    with open(wav, "rb") as fh:
        up = requests.post(f"{h.COMFY_URL}/upload/image",
                           files={"image": (name, fh, "audio/wav")},
                           data={"type": "input"}, timeout=60)
    up.raise_for_status()
    return up.json()["name"]


job = f"director-{uuid.uuid4().hex[:6]}"
h._wait_comfy_ready()

frames = []
if IMAGE_URL:
    frames = [{"name": h._fetch_and_upload_image(IMAGE_URL), "frame_idx": 0, "strength": 1.0}]
target = h._video_duration_sec(DUR)
audio_name = prep_audio(AUDIO_URL, AUDIO_START, target, job)
print(f"[test] job={job} q={QUALITY} {ASPECT} dur={target}s start={AUDIO_START}s i2v={bool(frames)} audio={audio_name}")

wf, meta = build_workflow(
    prompt=PROMPT, negative_prompt="", quality=QUALITY, aspect_ratio=ASPECT,
    duration_sec=DUR, seed=42, frames=frames, is_i2v=bool(frames),
    audio_name=audio_name)
print(f"[test] meta mode={meta['mode']} {meta['width']}x{meta['height']} frames={meta['num_frames']} nodes={len(wf)}")

cid = uuid.uuid4().hex
pid = h._queue(wf, cid)
print(f"[test] queued {pid}")
rec = h._poll(pid, cid, wf, timeout_s=1800, job_id=job)
mp4 = h._download_video(rec)
out = Path("/comfyui/output") / f"{job}.mp4"
out.write_bytes(mp4)
proxy = os.environ.get("POD_PROXY", "")
print(f"[test] SAVED {out} ({len(mp4)} bytes)")
print(f"[test] VIEW {proxy}/view?filename={job}.mp4&type=output")
