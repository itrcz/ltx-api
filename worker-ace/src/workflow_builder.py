"""Build an ACE-Step 1.5 XL-SFT ComfyUI API-format workflow for a given request.

The template (workflow_template_api.json) is the hi-quality 4-pass workflow
exported from ComfyUI's "Save (API Format)". It contains:
  * 1 × AceStepSFTModelLoader  — model + text encoder + VAE selection
  * 1 × AceStepSFTLoraLoader   — LoRA stage; we set strength=0 to disable
  * 1 × AceStepSFTTextEncode   — caption / lyrics / BPM / key / language / duration
  * 4 × AceStepSFTGenerate     — 4 sampling stages (rough → refine → polish → final)
  * 1 × AceStepSFTSaveAudio    — output filename prefix + format

We patch widgets by index. Indices are stable for the custom-node version we
pin in worker-ace/Dockerfile. If they ever shift, run worker-ace/test_builder.py
to surface the mismatch.
"""
from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, Optional

TEMPLATE_PATH = Path(__file__).parent / "workflow_template_api.json"

# Widget index map for AceStepSFTTextEncode
TE = {
    "tags":        0,   # style/genre/instrumentation tags
    "lyrics":      1,   # song text with [Verse]/[Chorus] tags
    "pure_lyrics": 2,   # bool — when True, disables tags-driven generation
    "text_seed":   3,   # int — text encoder noise seed
    "text_seed_mode": 4,  # "fixed" | "randomize"
    "duration":    5,   # seconds
    "bpm":         6,   # beats per minute
    "time_sig":    7,   # "4" | "3" | etc — beats per measure
    "language":    8,   # "ru" | "en" | "auto" | ...
    "key":         9,   # "C minor" | "A major" | ...
    # 10..16: advanced flags + chunking knobs we leave at template defaults
}

# Widget index map for AceStepSFTGenerate
GE = {
    "seed":            0,
    "seed_mode":       1,   # "fixed" | "randomize"
    "steps":           2,
    "cfg":             3,
    "sampler":         4,   # "euler" | "er_sde" | "res_2s" | ...
    "scheduler":       5,   # "linear_quadratic" | "sgm_uniform" | ...
    "shift":           6,
    "duration":        7,
    "inference":       8,   # "ode" | "sde"
    "guidance":        9,   # "apg" | "cfg"
    # 10..30: granular per-stage knobs (sigma_start / sigma_end / strength / etc)
    # — left at template defaults; tweak via ADVANCED_GE if needed
}

# Loader paths — must match what setup-volume-ace.sh provisions on the volume.
DEFAULT_DIT       = "acestep_v1.5_xl_sft_bf16.safetensors"
DEFAULT_QWEN_06B  = "qwen_0.6b_ace15.safetensors"
DEFAULT_QWEN_LM   = "qwen_4b_ace15.safetensors"   # composition LM; switchable to 1.7B for tighter VRAM
DEFAULT_VAE       = "ace_1.5_vae.safetensors"

# Bounds used in handler validation; duplicated here as defaults.
DURATION_MIN, DURATION_MAX, DURATION_DEFAULT = 30, 240, 120
STEPS_MIN, STEPS_MAX, STEPS_DEFAULT = 8, 80, 50
CFG_MIN, CFG_MAX, CFG_DEFAULT = 1.0, 12.0, 6.0
BPM_MIN, BPM_MAX = 60, 220
NEG_SAMPLERS = {"lcm", "ddim", "dpm_fast", "dpm_adaptive"}  # known-bad on ACE-Step
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "linear_quadratic"
DEFAULT_INFERENCE = "sde"
DEFAULT_LANGUAGE = "ru"
DEFAULT_KEY = "C minor"
DEFAULT_TIME_SIG = "4"

# High-level modes. Each maps to a tuple of:
#   (active_passes, steps, cfg, sampler, scheduler, inference)
# Passes that are NOT in active_passes get ComfyUI mode=4 (BYPASS) so the
# latent flows through unchanged; only the kept passes do real work.
MODES = {
    # ~10s on RTX 4090 — preview-grade. Single fast pass with ODE.
    "turbo":   {"passes": 1, "steps": 12, "cfg": 2.0, "sampler": "euler",
                "scheduler": "sgm_uniform", "inference": "ode"},
    # ~30s — decent for iteration. Single SDE pass at moderate settings.
    "fast":    {"passes": 1, "steps": 25, "cfg": 4.0, "sampler": "euler",
                "scheduler": "linear_quadratic", "inference": "sde"},
    # ~90s — production default. Full 4-pass.
    "quality": {"passes": 4, "steps": 50, "cfg": 6.0, "sampler": "euler",
                "scheduler": "linear_quadratic", "inference": "sde"},
    # ~150s — max effort, more refinement steps.
    "max":     {"passes": 4, "steps": 65, "cfg": 7.0, "sampler": "euler",
                "scheduler": "linear_quadratic", "inference": "sde"},
}
DEFAULT_MODE = "quality"

# ComfyUI node mode constants
NODE_MODE_RUN    = 0
NODE_MODE_BYPASS = 4


def _load_template() -> dict[str, Any]:
    if not TEMPLATE_PATH.exists():
        raise RuntimeError(
            f"workflow template missing at {TEMPLATE_PATH}. "
            "Export 'Save (API Format)' from ComfyUI Dev Mode and place it here."
        )
    return json.loads(TEMPLATE_PATH.read_text())


def _find_nodes(wf: dict, class_type: str) -> list[tuple[str, dict]]:
    """Return [(node_id, node_dict), ...] for every node of the given class."""
    return [(nid, n) for nid, n in wf.items() if n.get("class_type") == class_type]


def _set_widget(node: dict, idx: int, value: Any) -> None:
    """API-format nodes use `inputs` (dict). Widget order is preserved by ComfyUI
    via the `inputs` keys when "Save (API)" was used. We patch by KEY when the
    template was exported sensibly; fall back to indexed widgets list otherwise.
    """
    inputs = node.setdefault("inputs", {})
    keys = list(inputs.keys())
    if idx < len(keys):
        inputs[keys[idx]] = value


def build(
    *,
    prompt: str,
    lyrics: str,
    bpm: int,
    duration_sec: int,
    seed: Optional[int] = None,
    key: str = DEFAULT_KEY,
    language: str = DEFAULT_LANGUAGE,
    time_signature: str = DEFAULT_TIME_SIG,
    mode: str = DEFAULT_MODE,
    # advanced — when set, OVERRIDE the mode preset for that field.
    # When None/unset, mode preset wins.
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    inference: Optional[str] = None,
    job_id: Optional[str] = None,
) -> tuple[dict, dict]:
    """Return (workflow, meta) ready to POST to ComfyUI /prompt."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
    preset = MODES[mode]
    # Per-field override semantics: explicit param wins over preset.
    eff_steps     = steps     if steps     is not None else preset["steps"]
    eff_cfg       = cfg       if cfg       is not None else preset["cfg"]
    eff_sampler   = sampler   if sampler   is not None else preset["sampler"]
    eff_scheduler = scheduler if scheduler is not None else preset["scheduler"]
    eff_inference = inference if inference is not None else preset["inference"]
    active_passes = preset["passes"]

    wf = _load_template()

    # --- AceStepSFTTextEncode (single instance) ---
    # Field names per ComfyUI /object_info schema: caption (not "tags"),
    # timesignature (one word), keyscale (not "key").
    te_nodes = _find_nodes(wf, "AceStepSFTTextEncode")
    if len(te_nodes) != 1:
        raise RuntimeError(f"expected 1 AceStepSFTTextEncode, got {len(te_nodes)}")
    _, te = te_nodes[0]
    te_in = te["inputs"]
    te_in["caption"]       = prompt
    te_in["lyrics"]        = lyrics
    te_in["duration"]      = float(duration_sec)
    te_in["bpm"]           = bpm
    te_in["timesignature"] = time_signature
    te_in["language"]      = language
    te_in["keyscale"]      = key

    # --- AceStepSFTGenerate (4 passes in hi-quality template) ---
    # First N passes (where N == active_passes) RUN with the chosen settings.
    # Remaining passes get mode=4 (BYPASS): the latent passes through unchanged
    # and SaveAudio downstream still gets a valid audio_latent. This is how
    # turbo/fast modes skip the refinement stages without rewriting the graph.
    ge_nodes = _find_nodes(wf, "AceStepSFTGenerate")
    if not ge_nodes:
        raise RuntimeError("no AceStepSFTGenerate nodes in template")
    # Sort by node id (stable across ComfyUI exports) so pass ordering is consistent.
    ge_nodes_sorted = sorted(ge_nodes, key=lambda kv: int(kv[0]))
    base_seed = seed if seed is not None else secrets.randbelow(2**32)
    for i, (_, ge) in enumerate(ge_nodes_sorted):
        gi = ge["inputs"]
        # Each active pass uses base_seed + i — distinct noise per stage,
        # fully reproducible from one user-supplied seed.
        # Field names per /object_info: sampler_name (not "sampler"),
        # infer_method (not "inference").
        gi["seed"]         = base_seed + i
        gi["steps"]        = eff_steps
        gi["cfg"]          = eff_cfg
        gi["sampler_name"] = eff_sampler
        gi["scheduler"]    = eff_scheduler
        gi["duration"]     = float(duration_sec)
        gi["infer_method"] = eff_inference
        # Set node-level mode to BYPASS for stages beyond active_passes
        ge["mode"] = NODE_MODE_RUN if i < active_passes else NODE_MODE_BYPASS

    # --- AceStepSFTLoraLoader — bypass at node level (strengths to 0 too,
    # belt-and-suspenders in case the bypass routing mis-resolves) ---
    for _, ll in _find_nodes(wf, "AceStepSFTLoraLoader"):
        li = ll["inputs"]
        li["strength_model"] = 0.0
        li["strength_clip"]  = 0.0
        li["lora_name"]      = ""
        ll["mode"] = NODE_MODE_BYPASS

    # --- AceStepSFTModelLoader — pin filenames to what's on the volume.
    # Schema names: diffusion_model, text_encoder_1, text_encoder_2, vae_name.
    # text_encoder_1 = Qwen 0.6B (caption); text_encoder_2 = Qwen 4B (composition LM).
    for _, ml in _find_nodes(wf, "AceStepSFTModelLoader"):
        mi = ml["inputs"]
        mi["diffusion_model"] = DEFAULT_DIT
        mi["text_encoder_1"]  = DEFAULT_QWEN_06B
        mi["text_encoder_2"]  = DEFAULT_QWEN_LM
        mi["vae_name"]        = DEFAULT_VAE

    # --- AceStepSFTSaveAudio — namespace output by job_id ---
    job_id = job_id or secrets.token_hex(8)
    for _, sn in _find_nodes(wf, "AceStepSFTSaveAudio"):
        si = sn["inputs"]
        si["filename_prefix"] = f"ace/{job_id}"

    meta = {
        "job_id": job_id,
        "seed": base_seed,
        "mode": mode,
        "active_passes": active_passes,
        "duration_sec": duration_sec,
        "bpm": bpm,
        "key": key,
        "language": language,
        "steps": eff_steps,
        "cfg": eff_cfg,
        "sampler": eff_sampler,
        "scheduler": eff_scheduler,
        "inference": eff_inference,
    }
    return wf, meta
