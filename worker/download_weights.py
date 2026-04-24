"""
Build-time weight fetcher. Runs inside `docker build` so weights are baked
into the image layer — no cold-start download on RunPod.

HF token is passed via buildx secret mount at /run/secrets/hf_token.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = os.environ.get("HF_REPO_ID", "Lightricks/LTX-2.3")
TARGET = Path(os.environ.get("MODEL_DIR", "/models/ltx"))

def read_token() -> str | None:
    secret_path = Path("/run/secrets/hf_token")
    if secret_path.exists():
        return secret_path.read_text().strip()
    return os.environ.get("HF_TOKEN")

def main() -> None:
    token = read_token()
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} -> {TARGET}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(TARGET),
        token=token,
        allow_patterns=None,
        max_workers=8,
    )
    print("Done.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Weight download failed: {e}", file=sys.stderr)
        sys.exit(1)
