"""RunPod Serverless entrypoint."""
import os
import time
import uuid
import traceback

import runpod

from pipeline import GenParams, generate
from s3_upload import upload_and_presign

ALLOWED_STEPS = {5, 8, 10, 15, 20}
ALLOWED_QUALITY = {"draft", "hd"}
ALLOWED_AR = {"16:9", "9:16", "1:1", "4:3"}


def _validate(i: dict) -> GenParams:
    quality = i.get("quality", "draft")
    ar = i.get("aspect_ratio", "16:9")
    steps = int(i.get("steps", 8))
    duration = int(i.get("duration_sec", 4))
    prompt = i.get("prompt")

    if not prompt or not isinstance(prompt, str):
        raise ValueError("prompt is required")
    if quality not in ALLOWED_QUALITY:
        raise ValueError(f"quality must be one of {ALLOWED_QUALITY}")
    if ar not in ALLOWED_AR:
        raise ValueError(f"aspect_ratio must be one of {ALLOWED_AR}")
    if steps not in ALLOWED_STEPS:
        raise ValueError(f"steps must be one of {ALLOWED_STEPS}")
    if not 1 <= duration <= 10:
        raise ValueError("duration_sec must be 1..10")

    seed = i.get("seed")
    return GenParams(
        prompt=prompt,
        negative_prompt=i.get("negative_prompt"),
        steps=steps,
        duration_sec=duration,
        aspect_ratio=ar,
        quality=quality,
        seed=int(seed) if seed is not None else None,
        first_frame_url=i.get("first_frame_url"),
        last_frame_url=i.get("last_frame_url"),
    )


def handler(event):
    t0 = time.time()
    try:
        params = _validate(event.get("input") or {})
    except ValueError as e:
        return {"error": str(e)}

    job_id = event.get("id") or uuid.uuid4().hex

    def progress(p: float):
        runpod.serverless.progress_update(event, {"progress": round(p, 3)})

    try:
        video_path = generate(params, on_progress=progress)
        key = f"ltx/{time.strftime('%Y/%m/%d')}/{job_id}.mp4"
        url = upload_and_presign(
            video_path, key,
            expires_sec=int(os.environ.get("PRESIGN_TTL", "3600")),
        )
        return {
            "video_url": url,
            "quality": params.quality,
            "aspect_ratio": params.aspect_ratio,
            "duration_sec": params.duration_sec,
            "steps": params.steps,
            "elapsed_sec": round(time.time() - t0, 2),
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
