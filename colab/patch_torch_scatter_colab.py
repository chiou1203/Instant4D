from pathlib import Path


TARGETS = (
    Path("SLAM/mega-sam/torch_scatter.py"),
    Path("SLAM/mega-sam/base/droid_slam/torch_scatter.py"),
)

MODULE = '''"""Small Colab fallback for torch_scatter APIs used by Mega-SAM.

This is intended for smoke testing on Colab runtimes where torch-scatter wheels
are unavailable for the preinstalled Python/Torch/CUDA stack.
"""

import torch


def _broadcast_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = src.dim() + dim

    index = index.to(device=src.device, dtype=torch.long)
    if index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = index.numel()
        index = index.view(shape)

    while index.dim() < src.dim():
        index = index.unsqueeze(-1)

    return index.expand_as(src)


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    if dim < 0:
        dim = src.dim() + dim

    expanded_index = _broadcast_index(index, src, dim)
    if out is None:
        if dim_size is None:
            dim_size = int(expanded_index.max().item()) + 1 if expanded_index.numel() else 0
        size = list(src.shape)
        size[dim] = dim_size
        out = src.new_zeros(size)

    return out.scatter_add_(dim, expanded_index, src)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    if dim < 0:
        dim = src.dim() + dim

    if out is not None and dim_size is None:
        dim_size = out.shape[dim]
    summed = scatter_sum(src, index, dim=dim, out=None, dim_size=dim_size)
    expanded_index = _broadcast_index(index, src, dim)
    counts = scatter_sum(torch.ones_like(src), expanded_index, dim=dim, dim_size=summed.shape[dim])
    result = summed / counts.clamp_min(1)

    if out is not None:
        out.copy_(result)
        return out
    return result
'''


def main() -> None:
    wrote = []
    for target in TARGETS:
        if target.parent.exists():
            target.write_text(MODULE)
            wrote.append(str(target))

    if not wrote:
        raise FileNotFoundError("Mega-SAM folders are missing. Run submodule init first.")

    print("Wrote Colab torch_scatter fallback to:")
    for target in wrote:
        print(f"  {target}")


if __name__ == "__main__":
    main()
