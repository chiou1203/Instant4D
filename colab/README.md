# Instant4D Colab Workflows

This folder contains two Colab-oriented workflows for Instant4D. The quick-test workflow targets Colab's default runtime and uses compatibility fallbacks where Colab's newest Python/Torch stack lacks matching wheels. The native workflow creates a separate Python 3.10 environment with the repo-pinned Torch/xformers stack and the real PyG torch-scatter wheel.

## Quick Compatibility Test

Use a Colab Pro/Pro+ A100 runtime and run the bundled `example/panda` sequence first. That avoids dataset download friction and answers the main question: can the code build, reconstruct, prune, and optimize end-to-end in a Colab-style Linux environment?

Open `colab/Instant4D_Colab_A100.ipynb` in Colab, then run cells top to bottom.

## Native Comparison Run

Use `colab/Instant4D_Colab_Native_Py310.ipynb` when you want a more faithful environment for judging Instant4D against other methods. It creates:

```text
python=3.10
torch==2.2.0
torchvision==0.17.0
torchaudio==2.2.0
triton==2.2.0
xformers==0.0.24
torch-scatter from https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

The native notebook clones into `/content/Instant4D-native` so it does not inherit generated fallback modules from the quick-test run. It still applies `patch_megasam_colab.py` before compiling Mega-SAM CUDA extensions because that patch fixes Torch C++ API compatibility in source code. It does not apply the UniDepth xformers fallback or the torch-scatter fallback.

If the native run still reports `Loss=nan` or `Lssim=nan`, treat the result as not evaluation-ready and inspect the loss numerics before making quality claims.

## Drive Layout

The notebook assumes this persistent Drive layout:

```text
/content/drive/MyDrive/Instant4D/
  checkpoints/
  datasets/
  outputs/
```

You can put a custom sequence at:

```text
/content/drive/MyDrive/Instant4D/datasets/<scene_name>/*.png
```

or:

```text
/content/drive/MyDrive/Instant4D/datasets/<scene_name>/*.jpg
```

Then set:

```bash
SCENE_NAME=<scene_name>
DATA_DIR=/content/drive/MyDrive/Instant4D/datasets
```

## Manual Commands

If you do not want to use the notebooks, these are the core commands for the quick compatibility workflow:

```bash
cd /content
# Use your fork that contains this colab/ folder.
git clone --recursive https://github.com/<your-user>/Instant4D.git
cd /content/Instant4D
bash colab/setup_colab.sh
```

For the bundled sample:

```bash
cd /content/Instant4D
SCENE_NAME=panda DATA_DIR=/content/Instant4D/example bash colab/run_instant4d_colab.sh
```

For a Drive-hosted sequence:

```bash
cd /content/Instant4D
SCENE_NAME=my_scene DATA_DIR=/content/drive/MyDrive/Instant4D/datasets bash colab/run_instant4d_colab.sh
```

Outputs are written to:

```text
/content/Instant4D/output/colab/<scene_name>/
```

Copy that folder to Drive if you want it to survive Colab runtime reset.

For the native workflow, use:

```bash
cd /content
git clone --recursive --branch codex-colab-a100-workflow https://github.com/chiou1203/Instant4D.git /content/Instant4D-native
cd /content/Instant4D-native
REPO_ROOT=/content/Instant4D-native ENV_NAME=instant4d310 MAMBA_ROOT_PREFIX=/content/micromamba bash colab/setup_native_colab.sh
MAMBA_ROOT_PREFIX=/content/micromamba REPO_ROOT=/content/Instant4D-native SCENE_NAME=panda DATA_DIR=/content/Instant4D-native/example CONFIG_PATH=/content/Instant4D-native/configs/sora/panda.yaml MODEL_DIR=/content/Instant4D-native/output/native/panda /content/micromamba-bin/micromamba run -n instant4d310 bash colab/run_native_colab.sh
```

## Notes

- The setup script downloads `depth_anything_vitl14.pth` and `raft-things.pth` if they are missing.
- UniDepth downloads its Hugging Face model cache on first use.
- `megasam_final.pth` is expected to come from the Mega-SAM submodule; the setup script checks for it.
- A100 is the intended quick-test GPU. Smaller GPUs may need lower image resolution, fewer frames, or lower `num_pts`.
- The upstream scripts use local paths and `CUDA_VISIBLE_DEVICES=3`; the Colab runner uses GPU `0`.
- Colab's OpenCV build may not support H.264 (`avc1`) encoding. The evaluator now falls back to `mp4v` and raises an error if no video writer can be opened.
