#!/usr/bin/env python3
"""Pod-side smoke test for input-audio support — NO S3 required.

Runs the real handler pipeline pieces (audio download+trim+upload, workflow
build, queue, poll, download) but writes the mp4 to /runpod-volume/out/ instead
of uploading to S3, so it works on a bare test pod without S3 creds.

Usage (on the pod, after overlaying the audio-input code onto /):
    python3 pod_audio_test.py <AUDIO_URL> [mode] [duration_sec] [quality]
      mode      = mux | reference | lipsync     (default: reference)
      duration  = seconds                        (default: 10)
      quality   = sd | hd | fullhd               (default: sd)

Prints the saved path + an ffprobe summary so you can confirm the audio stream.
"""
import sys, os, json, uuid, subprocess
from pathlib import Path

sys.path.insert(0, "/")          # baked code lives at /handler.py etc.
import handler as h              # noqa: E402

URL = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "reference"
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
QUALITY = sys.argv[4] if len(sys.argv) > 4 else "sd"

job = f"podtest-{uuid.uuid4().hex[:6]}"
print(f"[test] job={job} mode={MODE} dur={DUR}s quality={QUALITY}")
h._wait_comfy_ready()
print("[test] comfy ready")

target = h._video_duration_sec(DUR)
audio_name = None
audio_wav = None
if MODE != "none":
    audio_name, audio_wav = h._prepare_audio(
        URL, target, job, upload=MODE in ("reference", "lipsync"))
    print(f"[test] audio prepared: trimmed to {target}s, comfy_name={audio_name}")

wf, meta = h.build_workflow(
    prompt="a person singing to the camera in a studio, close up",
    negative_prompt="", quality=QUALITY, aspect_ratio="16:9",
    duration_sec=DUR, seed=42, frames=[], is_i2v=False,
    t2v_dummy_name=h._upload_dummy_png(), steps=8,
    lora_strength=None, no_tile_vae=None,
    audio_name=audio_name, audio_mode=MODE)
print("[test] meta:", json.dumps(meta))

cid = uuid.uuid4().hex
pid = h._queue(wf, cid)
print(f"[test] queued comfy prompt {pid}")
rec = h._poll(pid, cid, wf, timeout_s=1800, job_id=job)
mp4 = h._download_video(rec)

outdir = Path("/runpod-volume/out"); outdir.mkdir(parents=True, exist_ok=True)
tmp = Path("/tmp") / f"{job}.mp4"; tmp.write_bytes(mp4)
out = outdir / f"{job}_{MODE}_{QUALITY}.mp4"
if MODE == "mux" and audio_wav is not None:
    muxed = Path("/tmp") / f"{job}_muxed.mp4"
    h._mux_audio(tmp, audio_wav, muxed)
    tmp = muxed
out.write_bytes(tmp.read_bytes())
print(f"[test] SAVED {out} ({out.stat().st_size} bytes)")

# Copy the final (possibly remuxed) file into ComfyUI's output dir so it is
# browsable via the pod proxy, and print a directly-viewable URL.
view_name = f"{job}_{MODE}_{QUALITY}.mp4"
try:
    Path("/comfyui/output").mkdir(parents=True, exist_ok=True)
    Path(f"/comfyui/output/{view_name}").write_bytes(out.read_bytes())
    POD_PROXY = os.environ.get("POD_PROXY", "")
    q = f"filename={view_name}&type=output"
    print(f"[test] VIEW {POD_PROXY}/view?{q}" if POD_PROXY else f"[test] VIEW /view?{q}")
except Exception as e:
    print("[test] view-copy failed:", e)

# Confirm the audio stream landed.
try:
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,duration,sample_rate",
         "-of", "json", str(out)],
        capture_output=True, text=True, timeout=30).stdout
    print("[test] ffprobe:", info)
except Exception as e:
    print("[test] ffprobe failed:", e)
