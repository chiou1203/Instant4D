import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = os.environ.get("UNIDEPTH_REPO_ID", "lpiccinelli/unidepth-v2-vitl14")
REVISION = os.environ.get(
    "UNIDEPTH_MODEL_REVISION",
    "1d0d3c52f60b5164629d279bb9a7546458e6dcc4",
)
TARGET = Path(
    os.environ.get(
        "UNIDEPTH_MODEL_PATH",
        "/content/Instant4D-native/checkpoints/hf/unidepth-v2-vitl14",
    )
)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading UniDepth snapshot {REPO_ID}@{REVISION} to {TARGET}", flush=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=str(TARGET),
        local_dir_use_symlinks=False,
    )
    print(f"UniDepth snapshot ready at {path}", flush=True)


if __name__ == "__main__":
    main()
