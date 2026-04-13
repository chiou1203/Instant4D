import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, hf_hub_download


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
    print(f"Downloading UniDepth files {REPO_ID}@{REVISION} to {TARGET}", flush=True)
    print("HF_HUB_DISABLE_XET=1", flush=True)

    files = HfApi().list_repo_files(repo_id=REPO_ID, revision=REVISION, repo_type="model")
    files = [name for name in files if not name.endswith("/")]
    if not files:
        raise RuntimeError(f"No files found for {REPO_ID}@{REVISION}")

    for idx, filename in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Downloading {filename}", flush=True)
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            revision=REVISION,
            repo_type="model",
            local_dir=str(TARGET),
        )
        size = Path(path).stat().st_size
        print(f"[{idx}/{len(files)}] Ready {filename} ({size / 1024**2:.2f} MiB)", flush=True)

    print(f"UniDepth snapshot ready at {TARGET}", flush=True)


if __name__ == "__main__":
    main()
