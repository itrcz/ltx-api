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
# IC-LoRA for Multiple-Subject-Reference (LiconStudio/LTX-2.3-Multiple-Subject-Reference,
# Apache-2.0). Trained against the canonical 8-step DISTILLED_SIGMAS_8 schedule —
# stacks on top of the distilled LoRA the same way TALKVID_LORA does. Requires the
# LiconMSR custom node (packs 1-4 subject images + a background image into a
# pseudo-video IMAGE batch) and LTXICLoRALoaderModelOnly/LTXAddVideoICLoRAGuide
# (both ship in the pinned ComfyUI-LTXVideo). Only loaded when reference_image_urls
# is set; must be present on the volume.
MSR_LORA = "ltxv/ltx2/ltx-2.3-licon-msr-v1.safetensors"
# LiconMSR's guide-clip length is independent of the output video's frame count —
# it's a compact reference clip that gets VAE-encoded and injected as IC-LoRA
# guide conditioning at frame_idx=0, not the visible output. Always use the
# richest (max) option for best identity fidelity.
MSR_FRAME_COUNT = 41
# LiconMSR only exposes 4 numbered subject slots ("1".."4") plus a separate
# required background slot — see ComfyUI-Licon-MSR's licon_msr.py.
MSR_MAX_SUBJECTS = 4

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


def _build_director_av(
    *, prompt: str, quality: str, aspect_ratio: str, duration_sec: float,
    seed: int, audio_name: str, image_name: Optional[str],
) -> tuple[dict, dict]:
    """Custom-audio lip-sync graph (audio_url present).

    Uses the WhatDreamsCost LTXDirector node with use_custom_audio=True: it
    encodes the supplied speech, attaches a noise_mask=0 (keep) audio latent, and
    the joint A/V sampler renders VIDEO whose lips follow that audio. CreateVideo
    muxes the Director's combined_audio (the user's track). Distilled + TalkVid
    ID-LoRA stack on the fp8 model; VAEDecodeTiled keeps fullhd within VRAM.

    A face image (image_name) acts as identity guide (i2v); without it the
    Director generates a speaker from the prompt (t2v). 2-stage 12+4 schedule
    mirrors the reference workflow."""
    w, h = _dims(quality, aspect_ratio)
    nf = _num_frames(duration_sec)
    dur_frames = nf - 1

    seg = {"id": "s1", "start": 0, "length": dur_frames, "prompt": prompt, "type": "image"}
    if image_name:
        seg["imageFile"] = image_name
    timeline = json.dumps({
        "segments": [seg],
        "audioSegments": [{"id": "a1", "type": "audio", "start": 0, "length": dur_frames,
                           "trimStart": 0, "audioFile": audio_name, "fileName": audio_name}],
    })

    wf = {
        "ck": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "te": {"class_type": "LTXAVTextEncoderLoader",
               "inputs": {"text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"}},
        "avae": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CKPT}},
        "lora1": {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["ck", 0], "lora_name": LORA_NAME, "strength_model": 0.45}},
        "lora2": {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["lora1", 0], "lora_name": TALKVID_LORA, "strength_model": 0.45}},
        "ups_model": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPSCALER}},
        "dir": {"class_type": "LTXDirector", "inputs": {
            "model": ["lora2", 0], "clip": ["te", 0], "global_prompt": prompt,
            "duration_frames": dur_frames, "duration_seconds": float(duration_sec),
            "timeline_data": timeline, "local_prompts": prompt,
            "segment_lengths": str(dur_frames), "epsilon": 0.001, "guide_strength": "1.00",
            "audio_vae": ["avae", 0], "use_custom_audio": True, "frame_rate": FPS,
            "display_mode": "seconds", "custom_width": w, "custom_height": h,
            "resize_method": "maintain aspect ratio", "divisible_by": 32, "img_compression": 0}},
        # stage 1
        "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["dir", 1]}},
        "cond": {"class_type": "LTXVConditioning",
                 "inputs": {"positive": ["dir", 1], "negative": ["zero", 0], "frame_rate": ["dir", 5]}},
        "dg1": {"class_type": "LTXDirectorGuide", "inputs": {
            "positive": ["cond", 0], "negative": ["cond", 1], "vae": ["ck", 2],
            "latent": ["dir", 2], "guide_data": ["dir", 4], "scale_by": 0.5, "upscale_method": "bicubic"}},
        "cav1": {"class_type": "LTXVConcatAVLatent",
                 "inputs": {"video_latent": ["dg1", 2], "audio_latent": ["dir", 3]}},
        "cfg1": {"class_type": "CFGGuider",
                 "inputs": {"model": ["dir", 0], "positive": ["dg1", 0], "negative": ["dg1", 1], "cfg": 2.0}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "sched1": {"class_type": "BasicScheduler",
                   "inputs": {"model": ["dir", 0], "scheduler": "linear_quadratic", "steps": 12, "denoise": 0.96}},
        "ks1": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "samp1": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["cfg1", 0], "sampler": ["ks1", 0],
            "sigmas": ["sched1", 0], "latent_image": ["cav1", 0]}},
        "sep1": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["samp1", 0]}},
        # stage 2
        "cg1": {"class_type": "LTXVCropGuides",
                "inputs": {"positive": ["dg1", 0], "negative": ["dg1", 1], "latent": ["sep1", 0]}},
        "ups": {"class_type": "LTXVLatentUpsampler",
                "inputs": {"samples": ["cg1", 2], "upscale_model": ["ups_model", 0], "vae": ["ck", 2]}},
        "dg2": {"class_type": "LTXDirectorGuide", "inputs": {
            "positive": ["cg1", 0], "negative": ["cg1", 1], "vae": ["ck", 2],
            "latent": ["ups", 0], "guide_data": ["dir", 4], "scale_by": 1.0, "upscale_method": "bicubic"}},
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
        # decode (tiled — fits fullhd) + mux the Director's combined (user) audio
        "vdec": {"class_type": "VAEDecodeTiled", "inputs": {
            "tile_size": 512, "overlap": 64, "temporal_size": 512, "temporal_overlap": 4,
            "samples": ["cg2", 2], "vae": ["ck", 2]}},
        "cv": {"class_type": "CreateVideo",
               "inputs": {"images": ["vdec", 0], "audio": ["dir", 6], "fps": ["dir", 5]}},
        "save": {"class_type": "SaveVideo",
                 "inputs": {"filename_prefix": "output", "format": "auto", "codec": "auto", "video": ["cv", 0]}},
    }
    meta = {
        "width": w, "height": h, "num_frames": nf, "fps": FPS,
        "duration_sec": round(dur_frames / FPS, 3),
        "quality": quality, "aspect_ratio": aspect_ratio,
        "mode": "lipsync-i2v" if image_name else "lipsync-t2v",
        "steps": 12, "refine_steps": 4, "seed": seed,
        "ckpt": CKPT, "text_encoder": TEXT_ENCODER, "precision": "fp8",
        "vae_tiled": True, "audio_input": True,
        "loras": [LORA_NAME, TALKVID_LORA], "lora_strengths": [0.45, 0.45],
        "keyframes": [{"frame_idx": 0, "strength": 1.0}] if image_name else [],
    }
    return wf, meta


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
    reference_names: Optional[list[str]] = None,   # MSR: 1-4 ComfyUI-side subject filenames
    background_name: Optional[str] = None,         # MSR: required background filename
) -> tuple[dict, dict]:
    """Return (workflow, meta). `workflow` is API-format dict for /prompt.

    frames: ordered list of {name, frame_idx, strength}. `name` must already
        be uploaded to ComfyUI. frame_idx is absolute (0 = first; -1 = last).
    is_i2v: explicit. If None, inferred from `frames` non-empty.
    t2v_dummy_name: when is_i2v=False the LoadImage node still needs a real
        filename — caller supplies a 64×64 black PNG name.
    lora_strength: multiplier × (0.2, 1.0, 0.5) base. None falls back to env.
    audio_name: when set, the request switches to the custom-audio lip-sync
        graph (see _build_director_av) — the supplied speech drives the lips and
        is the output soundtrack. The face, if any, is the first frame.
    reference_names / background_name: when set, 1-4 subject images plus a
        background image drive identity-preserving generation via IC-LoRA guide
        conditioning (LiconMSR + MSR_LORA), spliced into this same 3-stage
        distilled template — NOT a separate graph — so it keeps the tuned
        half-res→CFG-3.0/1.0 split→×2 upscale→3-step-refine recipe that makes
        the standard t2v/i2v output clean. An earlier version ran MSR through
        its own single-stage full-res graph (mirroring the author's reference
        workflow) and produced visibly degraded, unrealistic video — that
        recipe skips exactly the refinement this template provides. Mutually
        exclusive with audio_name for now.
    """
    # Input audio → custom-audio lip-sync graph (Director + tiled decode).
    if audio_name:
        first = next((f for f in (frames or []) if int(f.get("frame_idx", 0)) == 0), None)
        return _build_director_av(
            prompt=prompt, quality=quality, aspect_ratio=aspect_ratio,
            duration_sec=duration_sec, seed=seed, audio_name=audio_name,
            image_name=(first.get("name") if first else None),
        )

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

    # Multiple-Subject-Reference: stack the MSR IC-LoRA onto each of the three
    # per-stage model branches (mirrors how TALKVID_LORA stacks onto the
    # distilled LoRA in _build_director_av), so identity guidance runs through
    # the same tuned CFG-3.0/1.0 + ×2-upscale + refine recipe as everything
    # else — not a separate, less-refined single-pass graph.
    msr_latent_downscale_factor = None
    if reference_names:
        for stage_nid in ("4968", "5026", "5015"):
            msr_id = f"{stage_nid}_msr"
            wf[msr_id] = {"class_type": "LTXICLoRALoaderModelOnly",
                          "inputs": {"model": [stage_nid, 0], "lora_name": MSR_LORA,
                                     "strength_model": 1.0}}
        # CFGGuiders (5020/5033/5054) currently read model straight off
        # 4968/5026/5015 — redirect each to its MSR-stacked counterpart.
        wf["5020"]["inputs"]["model"] = ["4968_msr", 0]
        wf["5033"]["inputs"]["model"] = ["5026_msr", 0]
        wf["5054"]["inputs"]["model"] = ["5015_msr", 0]
        # latent_downscale_factor is derived from the LoRA file's own metadata —
        # identical across all three loader instances, so any one will do.
        msr_latent_downscale_factor = ["5026_msr", 1]

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

    # --- Multiple-Subject-Reference guide, via the same chain-splice pattern
    # as the keyframe loop below (plain-conditioning first, wrap into 1241
    # last). Applied once per stage, like a single "keyframe" at frame_idx=0.
    if reference_names:
        msr_inputs = {"width": w, "height": h, "frame_count": MSR_FRAME_COUNT}
        for i, name in enumerate(reference_names, start=1):
            li_id = f"msrref{i}"
            wf[li_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            msr_inputs[str(i)] = [li_id, 0]
        wf["msrbg"] = {"class_type": "LoadImage", "inputs": {"image": background_name}}
        msr_inputs["background"] = ["msrbg", 0]
        wf["msrpack"] = {"class_type": "LiconMSR", "inputs": msr_inputs}

        def _msr_guide(node_id, latent_src):
            wf[node_id] = {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
                "positive": pos_src, "negative": neg_src,
                "vae": ["3940", 2], "latent": latent_src, "image": ["msrpack", 0],
                "frame_idx": 0, "strength": 1.0,
                "latent_downscale_factor": msr_latent_downscale_factor,
                "crop": "center", "use_tiled_encode": False,
                "tile_size": 256, "tile_overlap": 64}}
            return [node_id, 0], [node_id, 1], [node_id, 2]

        pos_src, neg_src, latent_s1_src = _msr_guide("msrguide1", latent_s1_src)
        _, _, latent_s2_src = _msr_guide("msrguide2", latent_s2_src)

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

    if extra_frames or reference_names:
        # Re-route shared conditioning + per-stage latents to the chain tail.
        wf["1241"]["inputs"]["positive"] = pos_src
        wf["1241"]["inputs"]["negative"] = neg_src
        wf["4528"]["inputs"]["video_latent"] = latent_s1_src   # stage-1 ConcatAV
        wf["5043"]["inputs"]["video_latent"] = latent_s2_src   # stage-2 ConcatAV

    meta = {
        "width": w, "height": h, "num_frames": nf, "fps": FPS,
        "duration_sec": round((nf - 1) / FPS, 3),
        "quality": quality, "aspect_ratio": aspect_ratio,
        "mode": ("msr-i2v" if reference_names and use_image
                 else "msr-t2v" if reference_names
                 else "i2v" if use_image else "t2v"),
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
        "audio_input": False,
        "keyframes": ([{"frame_idx": 0, "strength": first_frame["strength"]
                        if first_frame and "strength" in first_frame else 1.0}]
                      if first_frame else []) + keyframes_meta,
        "reference_images": len(reference_names) if reference_names else 0,
        "loras": ([LORA_NAME, MSR_LORA] if reference_names else [LORA_NAME]),
    }
    return wf, meta
