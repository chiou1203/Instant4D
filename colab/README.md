# Instant4D Colab Quick Test

This folder contains a Colab-oriented workflow for quickly testing whether Instant4D is worth using as a reference implementation. It keeps the upstream files mostly untouched and adds wrappers for the parts that were hardcoded to the authors' local paths.

## Recommended First Test

Use a Colab Pro/Pro+ A100 runtime and run the bundled `example/panda` sequence first. That avoids dataset download friction and answers the main question: can the code build, reconstruct, prune, and optimize end-to-end in a Colab-style Linux environment?

Open `colab/Instant4D_Colab_A100.ipynb` in Colab, then run cells top to bottom.

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

If you do not want to use the notebook, these are the core commands for Colab:

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

## Notes

- The setup script downloads `depth_anything_vitl14.pth` and `raft-things.pth` if they are missing.
- UniDepth downloads its Hugging Face model cache on first use.
- `megasam_final.pth` is expected to come from the Mega-SAM submodule; the setup script checks for it.
- A100 is the intended quick-test GPU. Smaller GPUs may need lower image resolution, fewer frames, or lower `num_pts`.
- The upstream scripts use local paths and `CUDA_VISIBLE_DEVICES=3`; the Colab runner uses GPU `0`.
