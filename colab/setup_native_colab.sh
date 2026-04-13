#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/content/Instant4D}"
ENV_NAME="${ENV_NAME:-instant4d310}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/content/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-/content/micromamba-bin/micromamba}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

export MAMBA_ROOT_PREFIX
export TORCH_CUDA_ARCH_LIST
export CUDA_HOME
export FORCE_CUDA=1

cd "$REPO_ROOT"

echo "==> Installing micromamba if needed"
if [ ! -x "$MICROMAMBA_BIN" ]; then
  mkdir -p "$(dirname "$MICROMAMBA_BIN")"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "$(dirname "$MICROMAMBA_BIN")" --strip-components=1 bin/micromamba
fi

echo "==> Creating Python 3.10 native environment: $ENV_NAME"
if "$MICROMAMBA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists; reusing it."
else
  "$MICROMAMBA_BIN" create -y -n "$ENV_NAME" -c conda-forge python=3.10 pip
fi

run_env() {
  "$MICROMAMBA_BIN" run -n "$ENV_NAME" "$@"
}

echo "==> Initializing submodules"
git submodule update --init --recursive

echo "==> Removing generated fallback modules from previous Colab smoke-test runs"
rm -f SLAM/mega-sam/torch_scatter.py
rm -f SLAM/mega-sam/base/droid_slam/torch_scatter.py

echo "==> Installing pinned PyTorch, xformers, and torch-scatter wheels"
run_env python -m pip install -U "pip<26" setuptools wheel ninja packaging
run_env python -m pip install \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
run_env python -m pip install xformers==0.0.24 --extra-index-url https://download.pytorch.org/whl/cu121
run_env python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

cat > /tmp/instant4d_native_constraints.txt <<'EOF'
numpy<2.0
EOF

echo "==> Installing Instant4D Python requirements without replacing native Torch stack"
run_env python - <<'PY'
from pathlib import Path

source = Path("requirement.txt")
target = Path("/tmp/instant4d_native_requirements.txt")
skip = {"torch-scatter"}

lines = []
for raw in source.read_text().splitlines():
    package = raw.strip()
    if not package or package.startswith("#"):
        continue
    normalized = package.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0]
    if normalized in skip:
        print(f"Skipping {package}: installed from the PyG native wheel index already.")
        continue
    lines.append(package)

target.write_text("\n".join(lines) + "\n")
PY
run_env python -m pip install -c /tmp/instant4d_native_constraints.txt -r /tmp/instant4d_native_requirements.txt
run_env python -m pip install -c /tmp/instant4d_native_constraints.txt open3d

echo "==> Installing UniDepth requirements without replacing native Torch/xformers stack"
run_env python - <<'PY'
from pathlib import Path

source = Path("SLAM/mega-sam/UniDepth/requirements.txt")
target = Path("/tmp/unidepth_native_requirements.txt")
skip = {"torch", "torchvision", "torchaudio", "triton", "xformers"}

lines = []
for raw in source.read_text().splitlines():
    package = raw.strip()
    if not package or package.startswith("#"):
        continue
    normalized = package.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0]
    if normalized in skip:
        print(f"Skipping UniDepth requirement {package}: native stack installed explicitly.")
        continue
    lines.append(package)

target.write_text("\n".join(lines) + "\n")
PY
run_env python -m pip install -c /tmp/instant4d_native_constraints.txt -r /tmp/unidepth_native_requirements.txt
run_env python -m pip install --no-deps -e SLAM/mega-sam/UniDepth

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
  run_env python -m zipfile -e /tmp/instant4d_raft/models.zip /tmp/instant4d_raft
  cp /tmp/instant4d_raft/models/raft-things.pth SLAM/mega-sam/cvd_opt/raft-things.pth
fi

if [ ! -f SLAM/mega-sam/checkpoints/megasam_final.pth ]; then
  echo "ERROR: missing SLAM/mega-sam/checkpoints/megasam_final.pth. Check that the mega-sam submodule cloned correctly." >&2
  exit 1
fi

echo "==> Applying Mega-SAM Torch API compatibility patch"
run_env python colab/patch_megasam_colab.py

echo "==> Verifying native dependency stack before compiling extensions"
run_env python colab/verify_native_deps.py

echo "==> Building Mega-SAM CUDA extensions"
pushd SLAM/mega-sam/base >/dev/null
run_env python setup.py install
popd >/dev/null

echo "==> Building Gaussian Splatting helper extensions"
run_env python -m pip install --no-build-isolation ./submodule/fussed-ssim
run_env python -m pip install --no-build-isolation ./submodule/simple-knn
pushd submodule/pointops2 >/dev/null
run_env python setup.py install
popd >/dev/null

echo "==> Verifying native dependency stack after compiling extensions"
run_env python colab/verify_native_deps.py

echo "==> Native Colab setup complete"
echo "Use: $MICROMAMBA_BIN run -n $ENV_NAME bash colab/run_native_colab.sh"
