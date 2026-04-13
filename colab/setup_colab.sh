#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/content/Instant4D}"
cd "$REPO_ROOT"

echo "==> Initializing submodules"
git submodule update --init --recursive

echo "==> Python and CUDA sanity check"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo "==> Installing Python dependencies"
python -m pip install -U pip setuptools wheel ninja
python - <<'PY'
from pathlib import Path

source = Path("requirement.txt")
target = Path("/tmp/instant4d_colab_requirements.txt")
skip = {"torch-scatter"}

lines = []
for raw in source.read_text().splitlines():
    package = raw.strip()
    if not package or package.startswith("#"):
        continue
    if package.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0] in skip:
        print(f"Skipping {package}: not used by the Colab smoke-test runtime and often unavailable for Colab's newest Torch/Python combo.")
        continue
    lines.append(package)

target.write_text("\n".join(lines) + "\n")
PY
python -m pip install -r /tmp/instant4d_colab_requirements.txt
python -m pip install open3d
python -m pip install -r SLAM/mega-sam/UniDepth/requirements.txt
python -m pip install -e SLAM/mega-sam/UniDepth

if [ "${INSTALL_TORCH_SCATTER:-0}" = "1" ]; then
  echo "==> Installing optional torch-scatter"
  python -m pip install torch-scatter || echo "torch-scatter install failed; continuing because the runtime path does not import it."
fi

if [ "${INSTALL_XFORMERS:-0}" = "1" ]; then
  echo "==> Installing optional xformers"
  python -m pip install xformers || echo "xformers install failed; continuing because some Colab torch builds need a matching wheel."
else
  echo "==> Skipping optional xformers. Set INSTALL_XFORMERS=1 if UniDepth later requires it."
fi

echo "==> Downloading external checkpoints when missing"
mkdir -p SLAM/mega-sam/Depth-Anything/checkpoints
if [ ! -f SLAM/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth ]; then
  curl -L "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth" \
    -o SLAM/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth
fi

if [ ! -f SLAM/mega-sam/cvd_opt/raft-things.pth ]; then
  mkdir -p /tmp/instant4d_raft
  curl -L "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip" \
    -o /tmp/instant4d_raft/models.zip
  python -m zipfile -e /tmp/instant4d_raft/models.zip /tmp/instant4d_raft
  cp /tmp/instant4d_raft/models/raft-things.pth SLAM/mega-sam/cvd_opt/raft-things.pth
fi

if [ ! -f SLAM/mega-sam/checkpoints/megasam_final.pth ]; then
  echo "ERROR: missing SLAM/mega-sam/checkpoints/megasam_final.pth. Check that the mega-sam submodule cloned correctly." >&2
  exit 1
fi

echo "==> Applying torch > 2.7 CUDA extension compatibility patch if needed"
python - <<'PY'
from pathlib import Path

for path in Path("SLAM/mega-sam/base").rglob("*"):
    if path.suffix not in {".cu", ".cpp", ".h", ".hpp"}:
        continue
    if not path.exists():
        continue
    text = path.read_text()
    patched = text.replace(".type()", ".scalar_type()")
    if patched != text:
        path.write_text(patched)
        print("patched", path)
PY

echo "==> Building Mega-SAM CUDA extensions"
pushd SLAM/mega-sam/base >/dev/null
python setup.py install
popd >/dev/null

echo "==> Building Gaussian Splatting helper extensions"
python -m pip install ./submodule/fussed-ssim
python -m pip install ./submodule/simple-knn
pushd submodule/pointops2 >/dev/null
python setup.py install
popd >/dev/null

echo "==> Colab setup complete"
