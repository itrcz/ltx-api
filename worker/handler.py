"""
RunPod Serverless entrypoint for LTX-2.3 distilled.

Weights mount expected at /workspace/models (RunPod Network Volume):
  ltx23/ltx-2.3-22b-distilled-1.1.safetensors
  ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
  gemma/ (full HF snapshot of unsloth/gemma-3-12b-it)
"""
from __future__ import annotations

import io
import os
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

import requests
import runpod
import torch
from PIL import Image

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

from s3_upload import upload_and_presign

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/workspace/models"))
DISTILLED = MODELS_DIR / "ltx23" / "ltx-2.3-22b-distilled-1.1.safetensors"
UPSAMPLER = MODELS_DIR / "ltx23" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA = MODELS_DIR / "gemma"

# Resolution presets — portrait native; landscape just swaps dims.
# Draft = stage-1-only (no upsampler pass) for faster previews.
PORTRAIT_HD = (960, 1728)
PORTRAIT_DRAFT = (768, 1344)

ALLOWED_STEPS = {5, 8, 10, 15, 20}
ALLOWED_QUALITY = {"draft", "hd"}
ALLOWED_AR = {"9:16", "16:9"}  # start narrow; extend once base works
FPS = 24.0

_pipeline: Optional[DistilledPipeline] = None


def _load_pipeline() -> DistilledPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    for p in (DISTILLED, UPSAMPLER, GEMMA):
        if not Path(p).exists():
            raise RuntimeError(f"weights missing: {p}")
    t0 = time.time()
    _pipeline = DistilledPipeline(
        distilled_checkpoint_path=str(DISTILLED),
        spatial_upsampler_path=str(UPSAMPLER),
        gemma_root=str(GEMMA),
        loras=(),
        quantization=QuantizationPolicy.fp8_cast(),
    )
    print(f"[ltx] pipeline loaded in {time.time() - t0:.1f}s", flush=True)
    return _pipeline


def _dims(quality: str, aspect_ratio: str) -> tuple[int, int]:
    w, h = PORTRAIT_HD if quality == "hd" else PORTRAIT_DRAFT
    if aspect_ratio == "16:9":
        w, h = h, w
    return w, h


def _num_frames(duration_sec: int) -> int:
    n = int(duration_sec * FPS)
    return ((n - 1) // 8) * 8 + 1


def _fetch_image(url: str) -> Path:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    tmp = Path(tempfile.gettempdir()) / f"ltx_in_{uuid.uuid4().hex}.png"
    img.save(tmp)
    return tmp


def _validate(i: dict) -> dict:
    prompt = i.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise ValueError("prompt is required")
    quality = i.get("quality", "draft")
    if quality not in ALLOWED_QUALITY:
        raise ValueError(f"quality must be one of {sorted(ALLOWED_QUALITY)}")
    ar = i.get("aspect_ratio", "9:16")
    if ar not in ALLOWED_AR:
        raise ValueError(f"aspect_ratio must be one of {sorted(ALLOWED_AR)}")
    steps = int(i.get("steps", 8))
    if steps not in ALLOWED_STEPS:
        raise ValueError(f"steps must be one of {sorted(ALLOWED_STEPS)}")
    duration = int(i.get("duration_sec", 4))
    if not 1 <= duration <= 10:
        raise ValueError("duration_sec must be 1..10")
    if not i.get("first_frame_url"):
        raise ValueError("first_frame_url is required (LTX-2.3 is image-to-video)")
    return {
        "prompt": prompt,
        "negative_prompt": i.get("negative_prompt") or "",
        "quality": quality,
        "aspect_ratio": ar,
        "steps": steps,
        "duration_sec": duration,
        "seed": int(i["seed"]) if i.get("seed") is not None else 42,
        "first_frame_url": i["first_frame_url"],
        "last_frame_url": i.get("last_frame_url"),
    }


def handler(event):
    t0 = time.time()
    try:
        p = _validate(event.get("input") or {})
    except ValueError as e:
        return {"error": str(e)}

    job_id = event.get("id") or uuid.uuid4().hex

    def progress(frac: float):
        try:
            runpod.serverless.progress_update(event, {"progress": round(frac, 3)})
        except Exception:
            pass

    try:
        pipeline = _load_pipeline()
        progress(0.02)

        w, h = _dims(p["quality"], p["aspect_ratio"])
        nf = _num_frames(p["duration_sec"])

        images = [ImageConditioningInput(
            path=str(_fetch_image(p["first_frame_url"])),
            frame_idx=0, strength=1.0,
        )]
        if p.get("last_frame_url"):
            images.append(ImageConditioningInput(
                path=str(_fetch_image(p["last_frame_url"])),
                frame_idx=nf - 1, strength=0.5,
            ))

        tiling = TilingConfig.default()
        chunks = get_video_chunks_number(nf, tiling)
        progress(0.05)

        with torch.inference_mode():
            video, audio = pipeline(
                prompt=p["prompt"],
                seed=p["seed"],
                height=h,
                width=w,
                num_frames=nf,
                frame_rate=FPS,
                images=images,
                tiling_config=tiling,
            )
        progress(0.85)

        out_path = Path(tempfile.gettempdir()) / f"ltx_out_{job_id}.mp4"
        encode_video(
            video=video, fps=FPS, audio=audio,
            output_path=str(out_path),
            video_chunks_number=chunks,
        )
        progress(0.95)

        key = f"ltx/{time.strftime('%Y/%m/%d')}/{job_id}.mp4"
        url = upload_and_presign(
            out_path, key,
            expires_sec=int(os.environ.get("PRESIGN_TTL", "3600")),
        )
        progress(1.0)

        return {
            "video_url": url,
            "quality": p["quality"],
            "aspect_ratio": p["aspect_ratio"],
            "width": w, "height": h,
            "num_frames": nf,
            "steps": p["steps"],
            "elapsed_sec": round(time.time() - t0, 2),
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
