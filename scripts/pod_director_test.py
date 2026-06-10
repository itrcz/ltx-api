#!/usr/bin/env python3
"""Reproduce the WhatDreamsCost LTXDirector use_custom_audio workflow on our fp8
stack — image + speech audio -> talking head whose lips follow the supplied clip.

The Director node builds a custom-audio latent with noise_mask=0 (keep), so the
sampler renders VIDEO under the fixed audio (true audio->video), and CreateVideo
muxes the Director's combined_audio (your track). NO S3.

Usage (on the pod):
    POD_PROXY=https://<id>-8188.proxy.runpod.net \
    python3 pod_director_test.py <AUDIO_URL> <IMAGE_URL> [dur] [quality]
"""
import sys, os, json, uuid
from pathlib import Path
sys.path.insert(0, "/")
import handler as h   # noqa: E402

AUDIO_URL = sys.argv[1]
IMAGE_URL = sys.argv[2] if len(sys.argv) > 2 else ""
if IMAGE_URL in ("none", "-"):
    IMAGE_URL = ""
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
QUALITY = sys.argv[4] if len(sys.argv) > 4 else "sd"
PROMPT = os.environ.get("PROMPT", "a person talking to the camera, close up portrait")

CKPT = "ltx-2.3-22b-dev-fp8.safetensors"
TE = "gemma_3_12B_it_fp8_e4m3fn.safetensors"
DISTILLED = "ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
TALKVID = "ltxv/ltx2/ltx-2.3-id-lora-talkvid-3k.safetensors"
UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DIMS = {"sd": (1024, 576), "hd": (1344, 768), "fullhd": (1920, 1088)}
W, H = DIMS[QUALITY]
FPS = 24.0

job = f"director-{uuid.uuid4().hex[:6]}"
nf = h._num_frames(DUR)
dur_frames = nf - 1
target = round(dur_frames / FPS, 3)
print(f"[test] job={job} {W}x{H} frames={nf} dur={target}s")
h._wait_comfy_ready()

img_name = h._fetch_and_upload_image(IMAGE_URL) if IMAGE_URL else None
audio_name, _ = h._prepare_audio(AUDIO_URL, target, job, upload=True)
print(f"[test] image={img_name or '(t2v, none)'} audio={audio_name}")

seg = {"id": "s1", "start": 0, "length": dur_frames, "prompt": PROMPT, "type": "image"}
if img_name:                       # i2v: face guide. Omit -> Director t2v (dummy@0)
    seg["imageFile"] = img_name
timeline = json.dumps({
    "segments": [seg],
    "audioSegments": [{"id": "a1", "type": "audio", "start": 0, "length": dur_frames,
                       "trimStart": 0, "audioFile": audio_name, "fileName": audio_name}],
})

wf = {
    "ck":  {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
    "te":  {"class_type": "LTXAVTextEncoderLoader",
            "inputs": {"text_encoder": TE, "ckpt_name": CKPT, "device": "default"}},
    "avae": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
    "lora1": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["ck", 0], "lora_name": DISTILLED, "strength_model": 0.45}},
    "lora2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["lora1", 0], "lora_name": TALKVID, "strength_model": 0.45}},
    "ups_model": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPSCALER}},

    "dir": {"class_type": "LTXDirector", "inputs": {
        "model": ["lora2", 0], "clip": ["te", 0],
        "global_prompt": PROMPT, "duration_frames": dur_frames, "duration_seconds": DUR,
        "timeline_data": timeline, "local_prompts": PROMPT,
        "segment_lengths": str(dur_frames), "epsilon": 0.001, "guide_strength": "1.00",
        "audio_vae": ["avae", 0], "use_custom_audio": True, "frame_rate": FPS,
        "display_mode": "seconds", "custom_width": W, "custom_height": H,
        "resize_method": "maintain aspect ratio", "divisible_by": 32, "img_compression": 0,
    }},

    # --- stage 1 ---
    "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["dir", 1]}},
    "cond": {"class_type": "LTXVConditioning",
             "inputs": {"positive": ["dir", 1], "negative": ["zero", 0], "frame_rate": ["dir", 5]}},
    "dg1": {"class_type": "LTXDirectorGuide", "inputs": {
        "positive": ["cond", 0], "negative": ["cond", 1], "vae": ["ck", 2],
        "latent": ["dir", 2], "guide_data": ["dir", 4],
        "scale_by": 0.5, "upscale_method": "bicubic"}},
    "cav1": {"class_type": "LTXVConcatAVLatent",
             "inputs": {"video_latent": ["dg1", 2], "audio_latent": ["dir", 3]}},
    "cfg1": {"class_type": "CFGGuider",
             "inputs": {"model": ["dir", 0], "positive": ["dg1", 0], "negative": ["dg1", 1], "cfg": 2.0}},
    "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
    "sched1": {"class_type": "BasicScheduler",
               "inputs": {"model": ["dir", 0], "scheduler": "linear_quadratic", "steps": 12, "denoise": 0.96}},
    "ks1": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "samp1": {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["noise", 0], "guider": ["cfg1", 0], "sampler": ["ks1", 0],
        "sigmas": ["sched1", 0], "latent_image": ["cav1", 0]}},
    "sep1": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["samp1", 0]}},

    # --- stage 2 (upscale + refine) ---
    "cg1": {"class_type": "LTXVCropGuides",
            "inputs": {"positive": ["dg1", 0], "negative": ["dg1", 1], "latent": ["sep1", 0]}},
    "ups": {"class_type": "LTXVLatentUpsampler",
            "inputs": {"samples": ["cg1", 2], "upscale_model": ["ups_model", 0], "vae": ["ck", 2]}},
    "dg2": {"class_type": "LTXDirectorGuide", "inputs": {
        "positive": ["cg1", 0], "negative": ["cg1", 1], "vae": ["ck", 2],
        "latent": ["ups", 0], "guide_data": ["dir", 4],
        "scale_by": 1.0, "upscale_method": "bicubic"}},
    "cav2": {"class_type": "LTXVConcatAVLatent",
             "inputs": {"video_latent": ["dg2", 2], "audio_latent": ["sep1", 1]}},
    "cfg2": {"class_type": "CFGGuider",
             "inputs": {"model": ["dir", 0], "positive": ["dg2", 0], "negative": ["dg2", 1], "cfg": 1.0}},
    "sched2": {"class_type": "BasicScheduler",
               "inputs": {"model": ["dir", 0], "scheduler": "linear_quadratic", "steps": 4, "denoise": 0.42}},
    "ks2": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "samp2": {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["noise", 0], "guider": ["cfg2", 0], "sampler": ["ks2", 0],
        "sigmas": ["sched2", 0], "latent_image": ["cav2", 0]}},
    "sep2": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["samp2", 0]}},
    "cg2": {"class_type": "LTXVCropGuides",
            "inputs": {"positive": ["dg2", 0], "negative": ["dg2", 1], "latent": ["sep2", 0]}},

    # --- decode + mux the Director's combined (your) audio ---
    "vdec": {"class_type": "VAEDecode", "inputs": {"samples": ["cg2", 2], "vae": ["ck", 2]}},
    "cv": {"class_type": "CreateVideo",
           "inputs": {"images": ["vdec", 0], "audio": ["dir", 6], "fps": ["dir", 5]}},
    "save": {"class_type": "SaveVideo",
             "inputs": {"filename_prefix": job, "format": "auto", "codec": "auto", "video": ["cv", 0]}},
}

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
