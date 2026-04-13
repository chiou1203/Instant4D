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
cat > /tmp/instant4d_colab_constraints.txt <<'EOF'
numpy<2.0
EOF
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
        print(f"Skipping {package}: Colab's default Python/Torch stack often lacks a matching wheel; the runtime applies a compatibility fallback.")
        continue
    lines.append(package)

target.write_text("\n".join(lines) + "\n")
PY
python -m pip install -c /tmp/instant4d_colab_constraints.txt -r /tmp/instant4d_colab_requirements.txt
python -m pip install -c /tmp/instant4d_colab_constraints.txt open3d
python - <<'PY'
from pathlib import Path

source = Path("SLAM/mega-sam/UniDepth/requirements.txt")
target = Path("/tmp/unidepth_colab_requirements.txt")
skip = {"torch", "torchvision", "torchaudio", "triton", "xformers"}

lines = []
for raw in source.read_text().splitlines():
    package = raw.strip()
    if not package or package.startswith("#"):
        continue
    normalized = package.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0]
    if normalized in skip:
        print(f"Skipping UniDepth requirement {package}: Colab already provides Torch, or the runtime applies a compatibility fallback for this package.")
        continue
    lines.append(package)

target.write_text("\n".join(lines) + "\n")
PY
python -m pip install -c /tmp/instant4d_colab_constraints.txt -r /tmp/unidepth_colab_requirements.txt
# UniDepth's pyproject dynamically reads its original requirements.txt, which
# pins old Torch/Triton/xformers builds. Dependencies were installed from the
# filtered Colab requirements file above, so keep the editable install no-deps.
python -m pip install --no-deps -e SLAM/mega-sam/UniDepth

if [ "${INSTALL_TORCH_SCATTER:-0}" = "1" ]; then
  echo "==> Installing optional torch-scatter"
  python -m pip install torch-scatter || echo "torch-scatter install failed; using the Colab compatibility fallback instead."
fi

if [ "${INSTALL_XFORMERS:-0}" = "1" ]; then
  echo "==> Installing optional xformers"
  python -m pip install xformers || echo "xformers install failed; continuing because some Colab torch builds need a matching wheel."
else
  echo "==> Skipping optional xformers. Set INSTALL_XFORMERS=1 on a compatible Torch/Python stack for a native xformers run."
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
python colab/patch_megasam_colab.py
python colab/patch_unidepth_colab.py
python colab/patch_torch_scatter_colab.py

echo "==> Building Mega-SAM CUDA extensions"
pushd SLAM/mega-sam/base >/dev/null
python setup.py install
popd >/dev/null

echo "==> Building Gaussian Splatting helper extensions"
python -m pip install --no-build-isolation ./submodule/fussed-ssim
python -m pip install --no-build-isolation ./submodule/simple-knn
pushd submodule/pointops2 >/dev/null
python setup.py install
popd >/dev/null

echo "==> Colab setup complete"
