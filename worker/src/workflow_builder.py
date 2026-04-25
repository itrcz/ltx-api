"""Build an LTX-2.3 ComfyUI API-format workflow for a given request.

Starts from the hand-authored template (UI format exported via "Save (API)"),
then surgically wires in Gemma prompt enhancement and optional last_frame.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

TEMPLATE_PATH = Path(__file__).parent / "workflow_template_api.json"

# Only fp8 weights live on the network volume. Hardcoded — bf16 path was
# removed because we never deployed an endpoint with the headroom for it.
CKPT = "ltx-2.3-22b-dev-fp8.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp8_e4m3fn.safetensors"
# VAE_NO_TILE=1 disables tiled VAE decode (full-frame decode in one pass).
# Only set on cards with enough VRAM (≥48GB recommended for sd, ≥80GB for fullhd).
VAE_NO_TILE = os.environ.get("VAE_NO_TILE", "0") == "1"
# LTX_LORA_STRENGTH=0 disables the distilled LoRA. With LoRA disabled, the
# model runs in plain dev-fp8 mode and you should bump steps to ≥20 for
# comparable quality (DISTILLED_SIGMAS_8 only fits the LoRA-augmented model).
LTX_LORA_STRENGTH = os.environ.get("LTX_LORA_STRENGTH")  # e.g. "0" or "0.5"
I2V_SP_PATH = Path(__file__).parent.parent / "system_prompts" / "gemma_i2v_system_prompt.txt"
T2V_SP_PATH = Path(__file__).parent.parent / "system_prompts" / "gemma_t2v_system_prompt.txt"

# Final (stage-2) dims per quality + aspect. All dims /64.
# Stage 1 runs at half. Keep landscape and derive portrait by swap.
DIMS = {
    "sd":     (1024, 576),   # ≈480-576p
    "hd":     (1344, 768),   # ≈720p
    "fullhd": (1920, 1088),  # ≈1080p
}
FPS = 24.0
NEG_DEFAULT = "worst quality, static, blurry, ugly, cartoon, low resolution, jpeg artifacts"

# Distilled LoRA was trained for this exact 8-step schedule — use it verbatim
# when steps == 8 for best quality at that step count.
DISTILLED_SIGMAS_8 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STEPS_MIN, STEPS_MAX, STEPS_DEFAULT = 5, 30, 8


def _stage1_sigmas(steps: int) -> str:
    """Return a comma-separated sigma string with `steps + 1` values.

    For the canonical 8-step distilled schedule, reuse the trained values.
    For other counts, fall back to a linear 1.0→0.0 schedule — workable across
    5..30 but without the fine-tuned stops of the distilled curve.
    """
    if steps == 8:
        return DISTILLED_SIGMAS_8
    # Linear from 1.0 to 0.0 inclusive
    vals = [round(1.0 - i / steps, 6) for i in range(steps + 1)]
    vals[0] = 1.0
    vals[-1] = 0.0
    return ", ".join(str(v) for v in vals)


def _dims(quality: str, aspect: str) -> tuple[int, int]:
    w, h = DIMS[quality]
    if aspect == "9:16":
        w, h = h, w
    return w, h


def _num_frames(duration_sec: float) -> int:
    # LTX-2 latents are temporal blocks of 8 frames + 1 reference frame, so
    # frame count must be of the form 8k + 1. Round to the *nearest* valid
    # value (the previous floor variant always under-shot — 15s @24fps gave
    # 14.667s instead of 15.04s).
    target = max(1, int(round(duration_sec * FPS)))
    k = max(0, round((target - 1) / 8))
    return int(k) * 8 + 1


def build(
    *,
    prompt: str,
    negative_prompt: str,
    quality: str,
    aspect_ratio: str,
    duration_sec: float,
    seed: int,
    frames: Optional[list[dict]] = None,
    is_i2v: Optional[bool] = None,
    t2v_dummy_name: Optional[str] = None,
    steps: int = STEPS_DEFAULT,
    lora_strength: Optional[float] = None,   # 0.0..1.0, None = env default
    no_tile_vae: Optional[bool] = None,      # None = env default
) -> tuple[dict, dict]:
    """Return (workflow, meta). Workflow is API-format dict for ComfyUI /prompt.

    frames: ordered list of {name, frame_idx, strength}. `name` must already be
        uploaded to ComfyUI. frame_idx: absolute index within num_frames
        (0 = first). strength: 0..1 guide weight.
    is_i2v: explicit mode. If None, inferred from (frames is non-empty).
    t2v_dummy_name: when is_i2v=False, the LoadImage node still needs a real
        filename on server — caller supplies a dummy PNG name.
    """
    wf = json.loads(TEMPLATE_PATH.read_text())

    w, h = _dims(quality, aspect_ratio)
    nf = _num_frames(duration_sec)
    frames = list(frames or [])
    use_image = (len(frames) > 0) if is_i2v is None else is_i2v
    sp = (I2V_SP_PATH if use_image else T2V_SP_PATH).read_text().strip()

    # --- Stage 2 (final) dims are width/height here; stage 1 latent = /2 ---
    # EmptyLTXVLatentVideo widget: [width, height, length, batch] — these are STAGE-1 dims
    wf["3059"]["inputs"]["width"] = w // 2
    wf["3059"]["inputs"]["height"] = h // 2
    # num frames (both stage 1 latent + audio latent use same INT source via node 4988)
    wf["4988"]["inputs"]["value"] = nf
    # fps (shared float source via node 4989)
    wf["4989"]["inputs"]["value"] = FPS

    # bypass_i2v: true → t2v (image ignored by conditioning nodes). false → i2v.
    wf["4987"]["inputs"]["value"] = not use_image

    # Noise seed — stage 1 (4832) and stage 2 (4967) both use the same request seed.
    wf["4832"]["inputs"]["seed"] = seed
    wf["4967"]["inputs"]["seed"] = seed

    # Stage-1 sigma schedule (step count). Stage 2 kept at trained 3-step default.
    wf["4984"]["inputs"]["sigmas"] = _stage1_sigmas(steps)

    # Negative prompt (plain CLIPTextEncode path)
    wf["2612"]["inputs"]["text"] = negative_prompt or NEG_DEFAULT

    eff_no_tile = VAE_NO_TILE if no_tile_vae is None else bool(no_tile_vae)

    if lora_strength is not None:
        eff_lora = float(lora_strength)
    elif LTX_LORA_STRENGTH is not None:
        try:
            eff_lora = float(LTX_LORA_STRENGTH)
        except (TypeError, ValueError):
            eff_lora = None
    else:
        eff_lora = None

    wf["4982"]["inputs"]["text_encoder"] = TEXT_ENCODER
    wf["4982"]["inputs"]["ckpt_name"] = CKPT
    wf["3940"]["inputs"]["ckpt_name"] = CKPT
    wf["4010"]["inputs"]["ckpt_name"] = CKPT

    if eff_no_tile and "4995" in wf:
        # Setting tiles=1 effectively disables tiling; overlap stays at template
        # default since the node validates overlap >= 1 even when unused.
        wf["4995"]["inputs"]["horizontal_tiles"] = 1
        wf["4995"]["inputs"]["vertical_tiles"] = 1

    if eff_lora is not None and "4922" in wf:
        wf["4922"]["inputs"]["strength_model"] = eff_lora

    # --- Gemma enhance node ---
    # Wire: clip from 4982, image optional, prompt = user's raw text.
    wf["5001"] = {
        "class_type": "LTXVGemmaEnhancePrompt",
        "inputs": {
            "clip": ["4982", 0],
            "prompt": prompt or "",
            "system_prompt": sp,
            "max_tokens": 768,
            "bypass_i2v": not use_image,
            "seed": seed,
        },
    }
    # --- First frame (frame_idx==0, goes through existing LTXVImgToVideoConditionOnly) ---
    # Handler always supplies a filename in LoadImage — real image for i2v first frame
    # or a dummy for t2v. Extra keyframes (if any) are loaded via new LoadImage nodes below.
    first_frame = next((f for f in frames if f["frame_idx"] == 0), None)
    extra_frames = [f for f in frames if f["frame_idx"] != 0]
    wf["2004"]["inputs"]["image"] = (first_frame["name"] if first_frame
                                      else t2v_dummy_name or "ltx_dummy.png")
    if use_image:
        # Enhance receives the resized first frame for I2V scene analysis
        wf["5001"]["inputs"]["image"] = ["4990", 0]
        # First-frame strength may differ from template default 0.7
        if first_frame and "strength" in first_frame:
            wf["3159"]["inputs"]["strength"] = float(first_frame["strength"])
            wf["4970"]["inputs"]["strength"] = float(first_frame["strength"])

    # Positive CLIPTextEncode consumes the enhanced string
    wf["2483"]["inputs"]["text"] = ["5001", 0]

    # --- Extra keyframes via chained LTXVAddGuide BEFORE LTXVConditioning ---
    # Why before LTXVConditioning: the Conditioning node wraps positive/negative
    # into a NestedTensor (for audio+video joint path). LTXVAddGuide cannot clone
    # NestedTensor (upstream bug), but it works fine on the plain CONDITIONING
    # output of CLIPTextEncode. So we chain AddGuides on plain conditioning,
    # then feed the final (pos, neg) into LTXVConditioning for both stages.
    # Latents are per-stage (different resolutions) → duplicate AddGuide for latent
    # modification on stage 2.
    pos_src = ["2483", 0]
    neg_src = ["2612", 0]
    # IMPORTANT: AddGuide operates on plain VIDEO latent, BEFORE LTXVConcatAVLatent
    # (post-concat is a NestedTensor that AddGuide can't clone).
    # Stage 1 plain video latent source = LTXVImgToVideoConditionOnly output (3159).
    # Stage 2 plain video latent source = LTXVImgToVideoConditionOnly output (4970).
    latent_s1_src = ["3159", 0]
    latent_s2_src = ["4970", 0]

    next_id = 5100
    keyframes_meta = []
    for kf in extra_frames:
        idx = int(kf["frame_idx"])
        if idx < 0:
            idx = nf + idx  # -1 → N-1
        if not (0 <= idx < nf):
            raise ValueError(f"frame_idx {kf['frame_idx']} out of range 0..{nf - 1}")
        strength = float(kf.get("strength", 0.5))
        image_name = kf["name"]

        # LoadImage + Resize for this keyframe
        li_id = str(next_id); next_id += 1
        rs_id = str(next_id); next_id += 1
        wf[li_id] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        wf[rs_id] = {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "resize_type": "scale longer dimension",
                "resize_type.longer_size": 1536,
                "scale_method": "lanczos",
                "input": [li_id, 0],
            },
        }

        # Stage-1 AddGuide: modifies (pos, neg, stage1_latent)
        ag1_id = str(next_id); next_id += 1
        wf[ag1_id] = {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": pos_src, "negative": neg_src,
                "vae": ["3940", 2], "latent": latent_s1_src,
                "image": [rs_id, 0],
                "frame_idx": idx, "strength": strength,
            },
        }
        # The modified conditioning is used for BOTH stages (shared LTXVConditioning).
        pos_src = [ag1_id, 0]
        neg_src = [ag1_id, 1]
        latent_s1_src = [ag1_id, 2]

        # Stage-2 AddGuide: only latent output is used; conditioning output ignored.
        ag2_id = str(next_id); next_id += 1
        wf[ag2_id] = {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": pos_src, "negative": neg_src,
                "vae": ["3940", 2], "latent": latent_s2_src,
                "image": [rs_id, 0],
                "frame_idx": idx, "strength": strength,
            },
        }
        latent_s2_src = [ag2_id, 2]

        keyframes_meta.append({"frame_idx": idx, "strength": strength})

    # If any extra frames, rewire:
    #   - final (pos, neg) → LTXVConditioning(1241)
    #   - guided video latent → LTXVConcatAVLatent video input (so audio concat still happens)
    if extra_frames:
        wf["1241"]["inputs"]["positive"] = pos_src
        wf["1241"]["inputs"]["negative"] = neg_src
        wf["4528"]["inputs"]["video_latent"] = latent_s1_src   # stage 1 ConcatAV
        wf["4969"]["inputs"]["video_latent"] = latent_s2_src   # stage 2 ConcatAV

    meta = {
        "width": w, "height": h, "num_frames": nf, "fps": FPS,
        "duration_sec": round((nf - 1) / FPS, 3),
        "quality": quality, "aspect_ratio": aspect_ratio,
        "mode": "i2v" if use_image else "t2v",
        "steps": steps,
        "seed": seed,
        "ckpt": CKPT,
        "text_encoder": TEXT_ENCODER,
        "vae_tiled": not eff_no_tile,
        "lora_strength": eff_lora,
        "precision": "fp8",
        "keyframes": ([{"frame_idx": 0, "strength": first_frame["strength"]
                        if first_frame and "strength" in first_frame else 1.0}]
                      if first_frame else []) + keyframes_meta,
    }
    return wf, meta
