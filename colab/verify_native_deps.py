import importlib
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_FALLBACKS = (
    REPO_ROOT / "SLAM/mega-sam/torch_scatter.py",
    REPO_ROOT / "SLAM/mega-sam/base/droid_slam/torch_scatter.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    print("python:", sys.version.replace("\n", " "))
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))

    require(sys.version_info[:2] == (3, 10), "Native Colab env must use Python 3.10")
    require(torch.__version__.startswith("2.2.0"), "Native Colab env must use torch==2.2.0")
    require(torch.cuda.is_available(), "CUDA is not available in the native Colab env")

    for fallback in FORBIDDEN_FALLBACKS:
        require(not fallback.exists(), f"Fallback module is present and would shadow the native wheel: {fallback}")

    xformers = importlib.import_module("xformers")
    torch_scatter = importlib.import_module("torch_scatter")
    triton = importlib.import_module("triton")

    print("xformers:", getattr(xformers, "__version__", "unknown"), getattr(xformers, "__file__", ""))
    print("torch_scatter:", getattr(torch_scatter, "__version__", "unknown"), getattr(torch_scatter, "__file__", ""))
    print("triton:", getattr(triton, "__version__", "unknown"), getattr(triton, "__file__", ""))

    from torch_scatter import scatter_mean, scatter_sum

    src = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    idx = torch.tensor([0, 0, 1, 1], device="cuda")
    require(torch.allclose(scatter_sum(src, idx, dim=0), torch.tensor([3.0, 7.0], device="cuda")), "scatter_sum check failed")
    require(torch.allclose(scatter_mean(src, idx, dim=0), torch.tensor([1.5, 3.5], device="cuda")), "scatter_mean check failed")
    print("Native dependency verification passed.")


if __name__ == "__main__":
    main()
