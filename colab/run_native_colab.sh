#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/content/Instant4D}"
SCENE_NAME="${SCENE_NAME:-panda}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/example}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/SLAM/medium_native}"
PRUNE_DIR="${PRUNE_DIR:-$REPO_ROOT/SLAM/voxel_filter/output/native}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/output/native/$SCENE_NAME}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/sora/panda.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

if [ ! -d "$DATA_DIR/$SCENE_NAME" ]; then
  echo "ERROR: could not find sequence directory: $DATA_DIR/$SCENE_NAME" >&2
  exit 1
fi

echo "==> Verifying native dependencies"
rm -f SLAM/mega-sam/torch_scatter.py
rm -f SLAM/mega-sam/base/droid_slam/torch_scatter.py
python colab/verify_native_deps.py

mkdir -p "$WORK_DIR" "$PRUNE_DIR" "$MODEL_DIR"

echo "==> Applying Mega-SAM Torch API compatibility patch"
python colab/patch_megasam_colab.py

echo "==> Running Mega-SAM preprocessing for $SCENE_NAME"
pushd SLAM/mega-sam >/dev/null
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/UniDepth"

python UniDepth/scripts/demo_mega-sam.py \
  --scene-name "$SCENE_NAME" \
  --img-path "$DATA_DIR/$SCENE_NAME" \
  --outdir "$WORK_DIR/UniDepth"

python Depth-Anything/run_videos.py --encoder vitl \
  --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  --img-path "$DATA_DIR/$SCENE_NAME" \
  --outdir "$WORK_DIR/Depth-Anything/$SCENE_NAME"

python camera_tracking_scripts/test_demo.py \
  --datapath "$DATA_DIR/$SCENE_NAME" \
  --weights checkpoints/megasam_final.pth \
  --scene_name "$SCENE_NAME" \
  --mono_depth_path "$WORK_DIR/Depth-Anything" \
  --metric_depth_path "$WORK_DIR/UniDepth" \
  --disable_vis

python cvd_opt/preprocess_flow.py \
  --datapath "$DATA_DIR/$SCENE_NAME" \
  --model cvd_opt/raft-things.pth \
  --scene_name "$SCENE_NAME" \
  --mixed_precision

python cvd_opt/cvd_opt.py \
  --scene_name "$SCENE_NAME" \
  --output_dir outputs_cvd \
  --w_grad 2.0 \
  --w_normal 5.0

popd >/dev/null

echo "==> Voxel filtering and transform rewrite"
python colab/prune_colab.py \
  --scene_name "$SCENE_NAME" \
  --image_dir "$DATA_DIR/$SCENE_NAME" \
  --droid_dir "$REPO_ROOT/SLAM/mega-sam/outputs_cvd" \
  --motion_dir "$REPO_ROOT/SLAM/mega-sam/reconstructions" \
  --save_dir "$PRUNE_DIR" \
  --config "$CONFIG_PATH" \
  --clean

echo "==> 4DGS optimization"
python colab/optimize_colab.py \
  --config "$CONFIG_PATH" \
  --source_path "$PRUNE_DIR/$SCENE_NAME" \
  --model_path "$MODEL_DIR"

echo "==> Done. Outputs: $MODEL_DIR"
