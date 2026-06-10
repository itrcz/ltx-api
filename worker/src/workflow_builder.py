"""Build an LTX-2.3 ComfyUI API-format workflow for a given request.

Pipeline (3-pass distilled, audio-clean):
  stage-1a: 4 of 8 distilled steps, LoRA strength 0.2, CFG 3.0
  stage-1b: remaining 4 of 8 distilled steps, LoRA strength 1.0, CFG 1.0
  stage-2 : ×2 latent upscale + 3-step refine (0.85→0), LoRA 0.5, CFG 1.0

The 1.0→0.5 LoRA taper across stages keeps audio clean — late refining at
strength 0.5 with low sigmas re-denoises the audio block of the joint A/V
latent, which is where the "hiss" issue lives at uniform LoRA=1.0.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

TEMPLATE_PATH = Path(__file__).parent / "workflow_template_api.json"

CKPT = "ltx-2.3-22b-dev-fp8.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp8_e4m3fn.safetensors"
LORA_NAME = "ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
# ID-LoRA for audio-driven talking-head lip-sync (Comfy-Org/ltx-2.3 TalkVid-3K,
# == AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K). Trained "audio_ref_only_ic with
# negative temporal positions" → it pairs with LTXVSetAudioRefTokens (audio ref
# tokens) and needs NO video IC-LoRA guide. Public (no gating), ~1.16 GB.
# Only loaded for audio_mode="lipsync"; must be present on the volume.
TALKVID_LORA = "ltxv/ltx2/ltx-2.3-id-lora-talkvid-3k.safetensors"
TALKVID_STRENGTH = 1.0

# audio_mode values. "mux" needs no graph change (handler remuxes the mp4);
# "reference"/"lipsync" wire the input audio in as LTXV ref tokens.
AUDIO_MODES = {"none", "mux", "reference", "lipsync"}

# LTX_LORA_STRENGTH=N applies as a multiplier across the three pipeline LoRAs:
# base strengths (0.2, 1.0, 0.5) → all × N. N=1.0 is default; N=0 disables.
# With LoRA disabled the 8-step distilled schedule is off-distribution; bump
# steps to ≥20 if you intend to run that mode in anger.
LTX_LORA_STRENGTH = os.environ.get("LTX_LORA_STRENGTH")
# VAE_NO_TILE=1 raises VAEDecodeTiled.tile_size enough to skip tiling. Cheap
# correctness win at fullhd only on cards with large VRAM headroom (≥48 GB).
VAE_NO_TILE = os.environ.get("VAE_NO_TILE", "0") == "1"

# RunPod layout has /system_prompts (copied from worker/system_prompts) at the
# image root; the source tree puts them at worker/system_prompts. Vast layout
# bakes them at the same image-root path. Anything else can override via
# LTX_SYSTEM_PROMPTS_DIR (used by local pytest setups).
_SP_DIR = Path(os.environ.get(
    "LTX_SYSTEM_PROMPTS_DIR",
    str(Path(__file__).parent.parent / "system_prompts"),
))
if not _SP_DIR.exists():
    _alt = Path("/system_prompts")
    if _alt.exists():
        _SP_DIR = _alt
I2V_SP_PATH = _SP_DIR / "gemma_i2v_system_prompt.txt"
T2V_SP_PATH = _SP_DIR / "gemma_t2v_system_prompt.txt"

# Final (stage-2) dims per quality + aspect. All dims /64. Stage-1 latent = /2.
DIMS = {
    "sd":     (1024, 576),   # ≈480-576p
    "hd":     (1344, 768),   # ≈720p
    "fullhd": (1920, 1088),  # ≈1080p
}
FPS = 24.0
NEG_DEFAULT = "worst quality, static, blurry, ugly, cartoon, low resolution, jpeg artifacts"

# Distilled LoRA was trained for this exact 8-step schedule.
DISTILLED_SIGMAS_8 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
# Stage-2 refinement: 3 sub-steps starting from 0.85 (the trained schedule's
# last full-step sigma) — short low-sigma sweep cleans up audio + adds detail.
REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
REFINE_STEPS = 3
STEPS_MIN, STEPS_MAX, STEPS_DEFAULT = 5, 30, 8

# Per-stage base LoRA strengths. Empirical sweet spot for clean audio +
# preserved fidelity. See module docstring.
LORA_S1A_BASE = 0.2   # node 4968 — first 4 distilled steps, CFG 3.0
LORA_S1B_BASE = 1.0   # node 5026 — last 4 distilled steps, CFG 1.0
LORA_S2_BASE  = 0.5   # node 5015 — refinement pass after upscale, CFG 1.0


def _stage1_sigmas(steps: int) -> str:
    """`steps + 1` sigma values. Canonical 8-step schedule when steps==8,
    linear 1.0→0.0 otherwise."""
    if steps == 8:
        return DISTILLED_SIGMAS_8
    vals = [round(1.0 - i / steps, 6) for i in range(steps + 1)]
    vals[0] = 1.0
    vals[-1] = 0.0
    return ", ".join(str(v) for v in vals)


def _split_step(steps: int) -> int:
    """Index at which to split stage-1 sigmas into 1a/1b passes."""
    return max(1, steps // 2)


def _dims(quality: str, aspect: str) -> tuple[int, int]:
    w, h = DIMS[quality]
    if aspect == "9:16":
        w, h = h, w
    return w, h


def _num_frames(duration_sec: float) -> int:
    # LTX-2.x latents are temporal blocks of 8 frames + 1 reference.
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
    lora_strength: Optional[float] = None,   # multiplier on per-stage bases
    no_tile_vae: Optional[bool] = None,
    audio_name: Optional[str] = None,        # ComfyUI-side filename (already uploaded)
    audio_mode: str = "none",
) -> tuple[dict, dict]:
    """Return (workflow, meta). `workflow` is API-format dict for /prompt.

    frames: ordered list of {name, frame_idx, strength}. `name` must already
        be uploaded to ComfyUI. frame_idx is absolute (0 = first; -1 = last).
    is_i2v: explicit. If None, inferred from `frames` non-empty.
    t2v_dummy_name: when is_i2v=False the LoadImage node still needs a real
        filename — caller supplies a 64×64 black PNG name.
    lora_strength: multiplier × (0.2, 1.0, 0.5) base. None falls back to env.
    audio_name: when set with audio_mode in {reference, lipsync}, the input
        audio (already uploaded to ComfyUI input/) is encoded and attached as
        LTXV audio reference tokens to the conditioning. "lipsync" additionally
        loads the lip-dub IC-LoRA for tight audio→lip synchronisation. "mux"
        needs no graph change (the handler remuxes the soundtrack onto the mp4).
    """
    wf = json.loads(TEMPLATE_PATH.read_text())

    w, h = _dims(quality, aspect_ratio)
    nf = _num_frames(duration_sec)
    frames = list(frames or [])
    use_image = (len(frames) > 0) if is_i2v is None else is_i2v
    sp = (I2V_SP_PATH if use_image else T2V_SP_PATH).read_text().strip()

    # Stage-1 pixel dims live on EmptyLTXVLatentVideo. ×2 latent upsampler
    # later brings the final video to (w, h).
    wf["3059"]["inputs"]["width"] = w // 2
    wf["3059"]["inputs"]["height"] = h // 2
    wf["4979"]["inputs"]["value"] = nf
    wf["4978"]["inputs"]["value"] = FPS

    # bypass_i2v: true → t2v (image input ignored by both LTXVImgToVideoConditionOnly
    # nodes). false → i2v.
    wf["4977"]["inputs"]["value"] = bool(not use_image)

    # Same seed across the three RandomNoise nodes — deterministic across stages.
    for nid in ("5029", "5032", "5051"):
        wf[nid]["inputs"]["noise_seed"] = int(seed)

    # Stage-1 schedule + split point. 5030 holds the full sigma list, 5027
    # SplitSigmas slices it into [0..k] (1a) and [k..end] (1b).
    wf["5030"]["inputs"]["sigmas"] = _stage1_sigmas(steps)
    wf["5027"]["inputs"]["step"] = _split_step(steps)

    # Negative prompt is a static string on CLIPTextEncode.
    wf["2612"]["inputs"]["text"] = negative_prompt or NEG_DEFAULT

    # Loaders (these have stable filenames on the volume; see CLAUDE.md).
    wf["4960"]["inputs"]["text_encoder"] = TEXT_ENCODER
    wf["4960"]["inputs"]["ckpt_name"] = CKPT
    wf["3940"]["inputs"]["ckpt_name"] = CKPT
    wf["4010"]["inputs"]["ckpt_name"] = CKPT
    wf["5012"]["inputs"]["model_name"] = UPSCALER
    for nid in ("4968", "5015", "5026"):
        wf[nid]["inputs"]["lora_name"] = LORA_NAME

    # LoRA-strength multiplier — applied to each base.
    if lora_strength is not None:
        mult = float(lora_strength)
    elif LTX_LORA_STRENGTH is not None:
        try:
            mult = float(LTX_LORA_STRENGTH)
        except (TypeError, ValueError):
            mult = 1.0
    else:
        mult = 1.0
    wf["4968"]["inputs"]["strength_model"] = round(LORA_S1A_BASE * mult, 4)
    wf["5026"]["inputs"]["strength_model"] = round(LORA_S1B_BASE * mult, 4)
    wf["5015"]["inputs"]["strength_model"] = round(LORA_S2_BASE  * mult, 4)

    eff_no_tile = VAE_NO_TILE if no_tile_vae is None else bool(no_tile_vae)
    if eff_no_tile:
        # The pipeline tiled-decodes only at stage-2 output (node 5039).
        # tile_size large enough to cover any single frame disables tiling.
        wf["5039"]["inputs"]["tile_size"] = 4096
        wf["5039"]["inputs"]["overlap"] = 0
        wf["5039"]["inputs"]["temporal_size"] = max(nf, 512)
        wf["5039"]["inputs"]["temporal_overlap"] = 0

    # First-frame image: real for i2v, dummy png for t2v (LoadImage requires a
    # filename either way; bypass_i2v=true tells the conditioning to ignore it).
    first_frame = next((f for f in frames if f["frame_idx"] == 0), None)
    extra_frames = [f for f in frames if f["frame_idx"] != 0]
    wf["2004"]["inputs"]["image"] = (first_frame["name"] if first_frame
                                      else t2v_dummy_name or "ltx_dummy.png")
    if use_image and first_frame and "strength" in first_frame:
        # Stage-1 conditioning strength (was 0.7 default); stage-2 stays at 1.0.
        wf["3159"]["inputs"]["strength"] = float(first_frame["strength"])

    # Gemma prompt enhancement — wraps user prompt with system prompt + (for
    # i2v) the resized first frame for scene context. Output replaces 2483.text.
    wf["5001"] = {
        "class_type": "LTXVGemmaEnhancePrompt",
        "inputs": {
            "clip": ["4960", 0],
            "prompt": prompt or "",
            "system_prompt": sp,
            "max_tokens": 768,
            "bypass_i2v": bool(not use_image),
            "seed": int(seed),
        },
    }
    if use_image:
        wf["5001"]["inputs"]["image"] = ["4981", 0]
    wf["2483"]["inputs"]["text"] = ["5001", 0]

    # --- Extra keyframes via LTXVAddGuide chain ---
    # Why before LTXVConditioning(1241): Conditioning wraps pos/neg into a
    # NestedTensor for the joint A/V path; LTXVAddGuide can't clone NestedTensor.
    # Plain CLIPTextEncode outputs are clonable, so we chain there. Latents are
    # per-stage (different resolutions) — duplicate AddGuide for stage-2 latent.
    pos_src = ["2483", 0]
    neg_src = ["2612", 0]
    latent_s1_src = ["3159", 0]   # stage-1 video latent (post-img-condition)
    latent_s2_src = ["5044", 0]   # stage-2 video latent (post-img-condition + upscale)

    next_id = 5100
    keyframes_meta = []
    for kf in extra_frames:
        idx = int(kf["frame_idx"])
        if idx < 0:
            idx = nf + idx
        if not (0 <= idx < nf):
            raise ValueError(f"frame_idx {kf['frame_idx']} out of range 0..{nf - 1}")
        strength = float(kf.get("strength", 0.5))
        image_name = kf["name"]

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
        pos_src = [ag1_id, 0]
        neg_src = [ag1_id, 1]
        latent_s1_src = [ag1_id, 2]

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

    if extra_frames:
        # Re-route shared conditioning + per-stage latents to the chain tail.
        wf["1241"]["inputs"]["positive"] = pos_src
        wf["1241"]["inputs"]["negative"] = neg_src
        wf["4528"]["inputs"]["video_latent"] = latent_s1_src   # stage-1 ConcatAV
        wf["5043"]["inputs"]["video_latent"] = latent_s2_src   # stage-2 ConcatAV

    # --- Input-audio conditioning (reference / lipsync) ---
    # The model conditioning hub is LTXVConditioning(1241); every guider reads
    # its pos/neg off [1241,0]/[1241,1]. We splice LTXVSetAudioRefTokens between
    # the hub and the guiders so the input audio is attached as reference tokens
    # ("speaker identity context"), then re-point the guiders at the ref-token
    # outputs. The audio VAE loader (4010) is already in the template.
    #
    # Stage-1 ref tokens use the freshly-encoded input audio. Stage-2 reuses the
    # stage-1-generated audio latent ([5025,1]) and routes its frozen output
    # (noise_mask=0) into the stage-2 ConcatAV, so the upscale/refine pass keeps
    # the audio verbatim instead of re-denoising it at sigma 0.85 (this matches
    # the official LTX-2.3 two-stage lip-dub graph).
    use_audio = bool(audio_name) and audio_mode in ("reference", "lipsync")
    audio_iclora = use_audio and audio_mode == "lipsync"
    if use_audio:
        wf["6001"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_name}}
        wf["6002"] = {
            "class_type": "LTXVAudioVAEEncode",
            "inputs": {"audio": ["6001", 0], "audio_vae": ["4010", 0]},
        }
        # Stage-1: ref tokens from the encoded input audio.
        wf["6010"] = {
            "class_type": "LTXVSetAudioRefTokens",
            "inputs": {"positive": ["1241", 0], "negative": ["1241", 1],
                       "audio_latent": ["6002", 0]},
        }
        # Stage-2: ref tokens from the stage-1 audio latent; frozen output feeds
        # the stage-2 ConcatAV so audio is preserved across the refine pass.
        wf["6011"] = {
            "class_type": "LTXVSetAudioRefTokens",
            "inputs": {"positive": ["1241", 0], "negative": ["1241", 1],
                       "audio_latent": ["5025", 1]},
        }
        # Re-point stage-1 guiders (5020 = s1a, 5033 = s1b) and the stage-2
        # guider (5054) at the ref-token conditioning.
        for nid in ("5020", "5033"):
            wf[nid]["inputs"]["positive"] = ["6010", 0]
            wf[nid]["inputs"]["negative"] = ["6010", 1]
        wf["5054"]["inputs"]["positive"] = ["6011", 0]
        wf["5054"]["inputs"]["negative"] = ["6011", 1]
        wf["5043"]["inputs"]["audio_latent"] = ["6011", 2]   # frozen stage-1 audio

        if audio_iclora:
            # TalkVid ID-LoRA stacks on top of each per-stage distilled-LoRA
            # branch (4968 s1a, 5026 s1b, 5015 s2). It is "audio_ref_only_ic":
            # the audio ref tokens above + this LoRA drive lip-sync; the first
            # frame (img-cond) supplies identity — NO video IC-LoRA guide.
            for ic_id, lora_src, guider in (
                ("6020", "4968", "5020"),
                ("6021", "5026", "5033"),
                ("6022", "5015", "5054"),
            ):
                wf[ic_id] = {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "model": [lora_src, 0],
                        "lora_name": TALKVID_LORA,
                        "strength_model": TALKVID_STRENGTH,
                    },
                }
                wf[guider]["inputs"]["model"] = [ic_id, 0]

    meta = {
        "width": w, "height": h, "num_frames": nf, "fps": FPS,
        "duration_sec": round((nf - 1) / FPS, 3),
        "quality": quality, "aspect_ratio": aspect_ratio,
        "mode": "i2v" if use_image else "t2v",
        "steps": steps,
        "refine_steps": REFINE_STEPS,
        "seed": seed,
        "ckpt": CKPT,
        "text_encoder": TEXT_ENCODER,
        "vae_tiled": not eff_no_tile,
        "lora_strength_multiplier": mult,
        "lora_strengths": [
            wf["4968"]["inputs"]["strength_model"],
            wf["5026"]["inputs"]["strength_model"],
            wf["5015"]["inputs"]["strength_model"],
        ],
        "precision": "fp8",
        "audio_mode": audio_mode if (use_audio or audio_mode == "mux") else "none",
        "audio_input": bool(use_audio or (audio_name and audio_mode == "mux")),
        "audio_talkvid_lora": audio_iclora,
        "keyframes": ([{"frame_idx": 0, "strength": first_frame["strength"]
                        if first_frame and "strength" in first_frame else 1.0}]
                      if first_frame else []) + keyframes_meta,
    }
    return wf, meta
