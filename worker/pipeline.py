"""
LTX-2.3 inference pipeline.

Stage 1: base text-to-video (or i2v / keyframe-conditioned) at 720p
Stage 2 (optional, quality=hd): spatial upscale 720p -> 1080p via tiled VAE

NOTE: Exact LTX-2.3 API surface depends on what ships in the HF repo.
This module centralizes all model calls so we can swap to the official
Lightricks inference code path after the first vast.ai smoke test.
"""
from __future__ import annotations

import io
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
import torch
from PIL import Image

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models/ltx"))
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Aspect ratio -> (width, height) at base (stage 1) resolution.
# Base tier ~720p short side; upscale stage x1.5 -> ~1080p short side.
BASE_DIMS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1":  (960, 960),
    "4:3":  (960, 720),
}
UPSCALE_FACTOR = 1.5  # 720p -> 1080p

# Lazy globals so the model loads once per worker process and stays in VRAM.
_pipe = None
_upscaler = None


@dataclass
class GenParams:
    prompt: str
    negative_prompt: Optional[str]
    steps: int
    duration_sec: int
    aspect_ratio: str
    quality: str  # "draft" | "hd"
    seed: Optional[int]
    first_frame_url: Optional[str]
    last_frame_url: Optional[str]


def _load_pipelines():
    """Load LTX-2.3 base + upscaler. Called once per worker."""
    global _pipe, _upscaler
    if _pipe is not None:
        return _pipe, _upscaler

    # TODO: swap to official Lightricks inference entrypoint once we
    # confirm the repo layout. Likely one of:
    #   from diffusers import LTXPipeline
    #   _pipe = LTXPipeline.from_pretrained(MODEL_DIR, torch_dtype=DTYPE).to(DEVICE)
    # or a custom class exposed by the repo.
    from diffusers import LTXPipeline  # type: ignore

    _pipe = LTXPipeline.from_pretrained(str(MODEL_DIR), torch_dtype=DTYPE)
    _pipe.to(DEVICE)
    _pipe.vae.enable_tiling()  # keep VAE decode in 32GB

    # Upscaler: LTX ships a separate latent upsampler in the same repo
    # (subfolder tbd). Load lazily so draft-only jobs don't pay for it.
    _upscaler = None
    return _pipe, _upscaler


def _load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def _resolve_dims(aspect_ratio: str) -> tuple[int, int]:
    if aspect_ratio not in BASE_DIMS:
        raise ValueError(f"unsupported aspect_ratio: {aspect_ratio}")
    return BASE_DIMS[aspect_ratio]


def _num_frames(duration_sec: int, fps: int = 24) -> int:
    # LTX typically wants (N*8 + 1) frames; clamp to model constraint.
    n = duration_sec * fps
    return ((n - 1) // 8) * 8 + 1


def generate(params: GenParams, on_progress=None) -> Path:
    """Run the pipeline end-to-end, return path to encoded .mp4 file."""
    pipe, _ = _load_pipelines()
    width, height = _resolve_dims(params.aspect_ratio)
    num_frames = _num_frames(params.duration_sec)

    generator = None
    if params.seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(params.seed)

    kwargs = dict(
        prompt=params.prompt,
        negative_prompt=params.negative_prompt or "",
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=params.steps,
        generator=generator,
    )

    if params.first_frame_url:
        kwargs["image"] = _load_image_from_url(params.first_frame_url)
    if params.last_frame_url:
        # LTX-2.3 supports keyframe conditioning; exact kwarg name tbd
        kwargs["last_image"] = _load_image_from_url(params.last_frame_url)

    if on_progress:
        on_progress(0.05)

    with torch.inference_mode():
        result = pipe(**kwargs)
    frames = result.frames[0]  # list[PIL.Image] or np.ndarray

    if on_progress:
        on_progress(0.7)

    if params.quality == "hd":
        frames = _upscale(frames, on_progress=on_progress)

    out_path = _encode_mp4(frames, fps=24)
    if on_progress:
        on_progress(1.0)
    return out_path


def _upscale(frames, on_progress=None):
    """Stage 2: tiled spatial upscale 720p -> 1080p."""
    # TODO: wire up Lightricks latent/spatial upsampler from the repo.
    # Placeholder: PIL bicubic so the end-to-end path works during
    # smoke tests. Real upsampler is swapped in after we confirm
    # the subfolder / class name in the HF repo.
    out = []
    for i, f in enumerate(frames):
        img = f if isinstance(f, Image.Image) else Image.fromarray(f)
        w, h = img.size
        out.append(img.resize((int(w * UPSCALE_FACTOR), int(h * UPSCALE_FACTOR)), Image.BICUBIC))
        if on_progress and i % 8 == 0:
            on_progress(0.7 + 0.25 * (i / max(len(frames), 1)))
    return out


def _encode_mp4(frames, fps: int = 24) -> Path:
    import imageio
    tmp = Path(tempfile.gettempdir()) / f"ltx_{uuid.uuid4().hex}.mp4"
    writer = imageio.get_writer(
        str(tmp), fps=fps, codec="libx264",
        quality=8, macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        for f in frames:
            arr = f if not isinstance(f, Image.Image) else f
            writer.append_data(_to_np(arr))
    finally:
        writer.close()
    return tmp


def _to_np(img):
    import numpy as np
    if isinstance(img, Image.Image):
        return np.array(img)
    return img
