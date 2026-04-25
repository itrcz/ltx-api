"""Smoke test: 480p portrait, 1 sec, DistilledPipeline on PRO 6000.

Run on pod via:
    python /workspace/pod_smoke.py
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video


DISTILLED = "/workspace/models/ltx23/ltx-2.3-22b-distilled-1.1.safetensors"
UPSAMPLER = "/workspace/models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA = "/workspace/models/gemma"

WIDTH = 576
HEIGHT = 1024
NUM_FRAMES = 25
FPS = 24.0
SEED = 42
OUT = "/workspace/smoke_480p_1s.mp4"
IMG = "/workspace/smoke_first_frame.jpg"

PROMPT = (
    "An orange vintage Volkswagen T2 camper van drives slowly along a winding "
    "alpine mountain road through lush green Dolomite meadows, distant jagged "
    "peaks under a soft cloudy sky. Cinematic camera smoothly follows the van "
    "with a gentle parallax pan; grass and wildflowers sway in a light breeze; "
    "warm summer light. A calm warm male voice narrates in Russian off-screen, "
    "clearly audible: «Inogda luchshaya doroga — ta, u kotoroy net "
    "kontsa. Tolko gory, tishina i ty.» Subtle wind, distant birdsong, "
    "no music."
)


def make_placeholder_image(path: str, w: int, h: int) -> None:
    """Generate a simple alpine-ish placeholder so pipeline runs end-to-end."""
    img = Image.new("RGB", (w, h), (170, 200, 230))
    d = ImageDraw.Draw(img)
    d.rectangle((0, int(h * 0.55), w, h), fill=(120, 160, 90))
    d.polygon([(0, int(h * 0.55)), (w // 2, int(h * 0.30)), (w, int(h * 0.55))], fill=(110, 110, 120))
    d.rectangle((int(w * 0.35), int(h * 0.70), int(w * 0.65), int(h * 0.80)), fill=(220, 120, 50))
    img.save(path, "JPEG", quality=92)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("smoke")

    make_placeholder_image(IMG, WIDTH, HEIGHT)

    log.info("loading DistilledPipeline (this loads ~50GB weights)...")
    t0 = time.time()
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=DISTILLED,
        spatial_upsampler_path=UPSAMPLER,
        gemma_root=GEMMA,
        loras=(),
        quantization=QuantizationPolicy.fp8_cast(),
    )
    t_load = time.time() - t0
    log.info("pipeline loaded in %.1fs", t_load)

    images = [ImageConditioningInput(path=IMG, frame_idx=0, strength=1.0)]
    tiling = TilingConfig.default()
    chunks = get_video_chunks_number(NUM_FRAMES, tiling)

    log.info("generating: %dx%d, %d frames @ %.1f fps, seed=%d",
             WIDTH, HEIGHT, NUM_FRAMES, FPS, SEED)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        video, audio = pipeline(
            prompt=PROMPT,
            seed=SEED,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            frame_rate=FPS,
            images=images,
            tiling_config=tiling,
        )
    t_gen = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    log.info("generation done in %.1fs (peak VRAM %.1f GB)", t_gen, peak_gb)

    t0 = time.time()
    encode_video(video=video, fps=FPS, audio=audio, output_path=OUT,
                 video_chunks_number=chunks)
    t_enc = time.time() - t0
    out_size_mb = Path(OUT).stat().st_size / 1024 / 1024
    log.info("encoded in %.1fs -> %s (%.1f MB)", t_enc, OUT, out_size_mb)

    summary = {
        "resolution": f"{WIDTH}x{HEIGHT}",
        "num_frames": NUM_FRAMES,
        "duration_sec": round((NUM_FRAMES - 1) / FPS, 2),
        "load_sec": round(t_load, 1),
        "gen_sec": round(t_gen, 1),
        "encode_sec": round(t_enc, 1),
        "total_sec": round(t_load + t_gen + t_enc, 1),
        "peak_vram_gb": round(peak_gb, 1),
        "output_mp4": OUT,
        "output_mb": round(out_size_mb, 1),
    }
    print("SUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
