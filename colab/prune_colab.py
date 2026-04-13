import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.prune import voxel_filter


def sorted_images(image_dir: Path) -> list[Path]:
    images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError(f"No .png/.jpg/.jpeg images found in {image_dir}")
    return images


def normalize_images(image_dir: Path, output_dir: Path) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted_images(image_dir)

    width = height = None
    for idx, image_path in enumerate(images, start=1):
        dst = output_dir / f"{idx:05d}.png"
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            if width is None:
                width, height = rgb.size
            rgb.save(dst)

    return width, height


def load_time_duration(config_path: Path | None) -> tuple[float, float]:
    if config_path is None:
        return 0.0, 3.0

    text = config_path.read_text()
    match = re.search(r"(?m)^\s*time_duration:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]", text)
    if not match:
        raise ValueError(f"Could not read time_duration from {config_path}")
    return float(match.group(1)), float(match.group(2))


def rewrite_transforms(
    droid_path: Path,
    image_dir: Path,
    output_scene_dir: Path,
    time_start: float,
    time_end: float,
) -> None:
    images_dir = output_scene_dir / "images"
    width, height = normalize_images(image_dir, images_dir)

    droid_data = np.load(droid_path)
    intrinsic = droid_data["intrinsic"]
    cam_c2w = droid_data["cam_c2w"]
    image_count = len(sorted_images(image_dir))
    frame_count = min(image_count, cam_c2w.shape[0])

    frames = []
    denom = max(frame_count - 1, 1)
    for idx in range(frame_count):
        timestamp = time_start + (idx / denom) * (time_end - time_start)
        frames.append(
            {
                "file_path": f"images/{idx + 1:05d}",
                "transform_matrix": cam_c2w[idx].tolist(),
                "time": timestamp,
            }
        )

    transforms = {
        "w": width,
        "h": height,
        "fl_x": float(intrinsic[0, 0]),
        "fl_y": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "frames": frames,
    }

    for split in ["train", "test"]:
        with (output_scene_dir / f"transforms_{split}.json").open("w") as f:
            json.dump(transforms, f, indent=2)


def validate_filtered_cvd_times(output_scene_dir: Path, time_start: float, time_end: float) -> None:
    filtered_path = output_scene_dir / "filtered_cvd.npz"
    data = dict(np.load(filtered_path))
    time_stamp = data.get("time_stamp")
    scale_time = data.get("scale_time")
    if time_stamp is None:
        return

    changed = False
    invalid_time = ~np.isfinite(time_stamp) | (time_stamp < time_start) | (time_stamp > time_end)
    if np.any(invalid_time):
        midpoint = (time_start + time_end) / 2.0
        print(
            f"WARNING: correcting {int(np.count_nonzero(invalid_time))} filtered_cvd.npz "
            f"time values outside [{time_start}, {time_end}]"
        )
        time_stamp = np.nan_to_num(time_stamp, nan=midpoint, posinf=time_end, neginf=time_start)
        time_stamp = np.clip(time_stamp, time_start, time_end)
        data["time_stamp"] = time_stamp
        changed = True

    if scale_time is not None:
        min_scale = max((time_end - time_start) / 10_000.0, 1e-6)
        invalid_scale = ~np.isfinite(scale_time) | (scale_time <= 0)
        if np.any(invalid_scale):
            print(f"WARNING: correcting {int(np.count_nonzero(invalid_scale))} invalid scale_time values")
            scale_time = np.nan_to_num(scale_time, nan=min_scale, posinf=(time_end - time_start), neginf=min_scale)
            scale_time = np.clip(scale_time, min_scale, time_end - time_start)
            data["scale_time"] = scale_time
            changed = True

    if changed:
        np.savez(filtered_path, **data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab-safe Instant4D voxel filtering wrapper")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--droid_dir", required=True, type=Path)
    parser.add_argument("--motion_dir", required=True, type=Path)
    parser.add_argument("--save_dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--clean", action="store_true", help="Remove the existing output scene folder before writing")
    args = parser.parse_args()

    output_scene_dir = args.save_dir / args.scene_name
    if args.clean and output_scene_dir.exists():
        shutil.rmtree(output_scene_dir)
    output_scene_dir.mkdir(parents=True, exist_ok=True)

    droid_path = args.droid_dir / f"{args.scene_name}_sgd_cvd_hr.npz"
    motion_path = args.motion_dir / args.scene_name / "motion_prob.npy"

    if not droid_path.exists():
        raise FileNotFoundError(f"Missing CVD output: {droid_path}")
    if not motion_path.exists():
        raise FileNotFoundError(f"Missing Mega-SAM motion probabilities: {motion_path}")

    time_start, time_end = load_time_duration(args.config)
    voxel_filter(str(droid_path), str(motion_path), str(output_scene_dir), args.scene_name, use_mask=False)
    validate_filtered_cvd_times(output_scene_dir, time_start, time_end)
    rewrite_transforms(droid_path, args.image_dir, output_scene_dir, time_start, time_end)
    print(f"Wrote Colab-ready Mega-SAM dataset to {output_scene_dir}")


if __name__ == "__main__":
    main()
