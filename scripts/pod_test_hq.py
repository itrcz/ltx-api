"""LTX-2.3 HQ pipeline benchmark: dev + distilled-LoRA, fp8_cast."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS
from ltx_pipelines.utils.media_io import encode_video


DEV_CKPT = "/workspace/models/ltx23/ltx-2.3-22b-dev.safetensors"
LORA = "/workspace/models/ltx23/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
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
NEG = "worst quality, static, blurry, ugly, cartoon, low resolution, jpeg artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--img", type=str, required=True)
    ap.add_argument("--stage1-steps", type=int, default=8)
    ap.add_argument("--lora-strength-stage1", type=float, default=1.0)
    ap.add_argument("--lora-strength-stage2", type=float, default=1.0)
    ap.add_argument("--tile-px", type=int, default=512)
    ap.add_argument("--tile-frames", type=int, default=64)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("hq")

    nf_raw = max(1, int(round(args.duration * FPS)))
    num_frames = ((nf_raw - 1) // 8) * 8 + 1

    log.info("loading TI2VidTwoStagesHQPipeline (dev + distilled-LoRA, fp8_cast)...")
    t0 = time.time()
    pipeline = TI2VidTwoStagesHQPipeline(
        checkpoint_path=DEV_CKPT,
        distilled_lora=[LoraPathStrengthAndSDOps(path=LORA, strength=1.0, sd_ops=None)],
        distilled_lora_strength_stage_1=args.lora_strength_stage1,
        distilled_lora_strength_stage_2=args.lora_strength_stage2,
        spatial_upsampler_path=UPSAMPLER,
        gemma_root=GEMMA,
        loras=(),
        quantization=QuantizationPolicy.fp8_cast(),
    )
    t_load = time.time() - t0
    log.info("pipeline ready in %.1fs", t_load)

    images = [ImageConditioningInput(path=args.img, frame_idx=0, strength=1.0)]
    tiling = TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=args.tile_px, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(tile_size_in_frames=args.tile_frames, tile_overlap_in_frames=24),
    )
    chunks = get_video_chunks_number(num_frames, tiling)

    log.info("generating: %dx%d, %d frames @ %.1f fps (%.2fs), stage1=%d steps, stage2=3 steps (default)",
             args.width, args.height, num_frames, FPS, (num_frames - 1) / FPS, args.stage1_steps)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    # Wrap BOTH pipeline call and encode_video in inference_mode to match LTX-2's main().
    # The pipeline returns a lazy iterator whose decode runs during encode_video; if that
    # runs outside of an inference/no_grad context, internal inference tensors leak and
    # trigger "Inference tensors cannot be saved for backward".
    with torch.inference_mode():
        video, audio = pipeline(
            prompt=PROMPT,
            negative_prompt=NEG,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=num_frames,
            frame_rate=FPS,
            num_inference_steps=args.stage1_steps,
            video_guider_params=LTX_2_3_HQ_PARAMS.video_guider_params,
            audio_guider_params=LTX_2_3_HQ_PARAMS.audio_guider_params,
            images=images,
            tiling_config=tiling,
        )
        t_gen = time.time() - t0
        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        log.info("gen done in %.1fs (peak alloc %.1f GB, reserved %.1f GB)",
                 t_gen, peak_alloc_gb, peak_reserved_gb)

        t0 = time.time()
        encode_video(video=video, fps=FPS, audio=audio, output_path=args.out,
                     video_chunks_number=chunks)
    t_enc = time.time() - t0
    peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    out_mb = Path(args.out).stat().st_size / 1024 / 1024
    log.info("encoded in %.1fs -> %s (%.1f MB)", t_enc, args.out, out_mb)

    print("SUMMARY:", json.dumps({
        "pipeline": "HQ (dev + distilled-LoRA, fp8_cast)",
        "resolution": f"{args.width}x{args.height}",
        "num_frames": num_frames,
        "duration_sec": round((num_frames - 1) / FPS, 2),
        "stage1_steps": args.stage1_steps,
        "stage2_steps": 3,
        "tile_px": args.tile_px,
        "tile_frames": args.tile_frames,
        "load_sec": round(t_load, 1),
        "gen_sec": round(t_gen, 1),
        "encode_sec": round(t_enc, 1),
        "total_sec": round(t_load + t_gen + t_enc, 1),
        "peak_vram_alloc_gb": round(peak_alloc_gb, 1),
        "peak_vram_reserved_gb": round(peak_reserved_gb, 1),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        "output_mp4": args.out,
        "output_mb": round(out_mb, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
