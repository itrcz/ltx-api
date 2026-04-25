"""Parameterized LTX-2.3 distilled benchmark."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video


DISTILLED = "/workspace/models/ltx23/ltx-2.3-22b-distilled-1.1.safetensors"
UPSAMPLER = "/workspace/models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA = "/workspace/models/gemma"
FPS = 24.0

PROMPT = (
    "An orange vintage Volkswagen T2 camper van drives slowly along a winding "
    "alpine mountain road through lush green Dolomite meadows, distant jagged "
    "peaks under a soft cloudy sky. Cinematic camera smoothly follows the van "
    "with a gentle parallax pan; grass and wildflowers sway in a light breeze; "
    "warm summer light. A calm warm male voice narrates in Russian off-screen, "
    "clearly audible: «Inogda luchshaya doroga — ta, u kotoroy net "
    "kontsa. Tolko gory, tishina i ty.» Subtle wind, distant birdsong, no music."
)


def make_placeholder_image(path: str, w: int, h: int) -> None:
    img = Image.new("RGB", (w, h), (170, 200, 230))
    d = ImageDraw.Draw(img)
    d.rectangle((0, int(h * 0.55), w, h), fill=(120, 160, 90))
    d.polygon([(0, int(h * 0.55)), (w // 2, int(h * 0.30)), (w, int(h * 0.55))], fill=(110, 110, 120))
    d.rectangle((int(w * 0.35), int(h * 0.70), int(w * 0.65), int(h * 0.80)), fill=(220, 120, 50))
    img.save(path, "JPEG", quality=92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--duration", type=float, required=True, help="seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--img", type=str, default=None,
                    help="path to first frame image (placeholder generated if absent)")
    ap.add_argument("--tile-px", type=int, default=512, help="VAE spatial tile size in pixels")
    ap.add_argument("--tile-frames", type=int, default=64, help="VAE temporal tile size in frames")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("test")

    nf_raw = max(1, int(round(args.duration * FPS)))
    num_frames = ((nf_raw - 1) // 8) * 8 + 1

    if args.img and Path(args.img).exists():
        img_path = args.img
        log.info("using user image: %s", img_path)
    else:
        img_path = "/workspace/_placeholder.jpg"
        make_placeholder_image(img_path, args.width, args.height)
        log.info("generated placeholder image at %s", img_path)

    log.info("loading DistilledPipeline...")
    t0 = time.time()
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=DISTILLED,
        spatial_upsampler_path=UPSAMPLER,
        gemma_root=GEMMA,
        loras=(),
        quantization=QuantizationPolicy.fp8_cast(),
    )
    t_load = time.time() - t0
    log.info("pipeline ready in %.1fs", t_load)

    images = [ImageConditioningInput(path=img_path, frame_idx=0, strength=1.0)]
    tiling = TilingConfig.default()
    chunks = get_video_chunks_number(num_frames, tiling)

    log.info("generating: %dx%d, %d frames @ %.1f fps (%.2fs), seed=%d",
             args.width, args.height, num_frames, FPS, (num_frames - 1) / FPS, args.seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        video, audio = pipeline(
            prompt=PROMPT,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=num_frames,
            frame_rate=FPS,
            images=images,
            tiling_config=tiling,
        )
    t_gen = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    log.info("gen done in %.1fs (peak VRAM %.1f GB)", t_gen, peak_gb)

    t0 = time.time()
    encode_video(video=video, fps=FPS, audio=audio, output_path=args.out,
                 video_chunks_number=chunks)
    t_enc = time.time() - t0
    out_mb = Path(args.out).stat().st_size / 1024 / 1024
    log.info("encoded in %.1fs -> %s (%.1f MB)", t_enc, args.out, out_mb)

    print("SUMMARY:", json.dumps({
        "resolution": f"{args.width}x{args.height}",
        "num_frames": num_frames,
        "duration_sec": round((num_frames - 1) / FPS, 2),
        "load_sec": round(t_load, 1),
        "gen_sec": round(t_gen, 1),
        "encode_sec": round(t_enc, 1),
        "total_sec": round(t_load + t_gen + t_enc, 1),
        "peak_vram_gb": round(peak_gb, 1),
        "output_mp4": args.out,
        "output_mb": round(out_mb, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
