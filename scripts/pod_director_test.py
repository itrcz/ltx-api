#!/usr/bin/env python3
"""Integration test of the PRODUCTION custom-audio lip-sync path (no S3).

Uploads image + audio, then calls the real workflow_builder.build(audio_name=...)
— i.e. the Director + use_custom_audio + VAEDecodeTiled graph that ships — queues
it, and saves the result to /comfyui/output for the pod proxy.

Usage (on the pod):
    POD_PROXY=https://<id>-8188.proxy.runpod.net \
    python3 pod_director_test.py <AUDIO_URL> [IMAGE_URL|none] [dur] [quality]
"""
import sys, os, uuid
from pathlib import Path
sys.path.insert(0, "/")
import handler as h                       # noqa: E402
from workflow_builder import build as build_workflow   # noqa: E402

AUDIO_URL = sys.argv[1]
IMAGE_URL = sys.argv[2] if len(sys.argv) > 2 else ""
if IMAGE_URL in ("none", "-"):
    IMAGE_URL = ""
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
QUALITY = sys.argv[4] if len(sys.argv) > 4 else "sd"
PROMPT = os.environ.get("PROMPT", "a person talking to the camera, close up portrait")

job = f"director-{uuid.uuid4().hex[:6]}"
h._wait_comfy_ready()

frames = []
if IMAGE_URL:
    frames = [{"name": h._fetch_and_upload_image(IMAGE_URL), "frame_idx": 0, "strength": 1.0}]
target = h._video_duration_sec(DUR)
audio_name, _ = h._fetch_and_upload_audio(AUDIO_URL, target, job)
print(f"[test] job={job} q={QUALITY} dur={target}s i2v={bool(frames)} audio={audio_name}")

wf, meta = build_workflow(
    prompt=PROMPT, negative_prompt="", quality=QUALITY, aspect_ratio="16:9",
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
