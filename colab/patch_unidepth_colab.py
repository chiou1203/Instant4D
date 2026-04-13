from pathlib import Path


TARGET = Path("SLAM/mega-sam/UniDepth/unidepth/layers/nystrom_attention.py")

ORIGINAL = "from xformers.components.attention import NystromAttention"
PATCHED = """try:
    from xformers.components.attention import NystromAttention
except ModuleNotFoundError:
    class NystromAttention(nn.Module):
        def __init__(self, num_landmarks: int = 128, num_heads: int = 4, dropout: float = 0.0):
            super().__init__()
            self.dropout = dropout

        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            q = rearrange(q, "b n h d -> b h n d")
            k = rearrange(k, "b n h d -> b h n d")
            v = rearrange(v, "b n h d -> b h n d")
            try:
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout, attn_mask=key_padding_mask
                )
            except RuntimeError:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
            return rearrange(x, "b h n d -> b n h d")
"""


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Missing UniDepth Nystrom source: {TARGET}")

    text = TARGET.read_text()
    if "class NystromAttention(nn.Module):" in text:
        print("UniDepth xformers fallback patch already present.")
        return

    if ORIGINAL not in text:
        raise SystemExit(f"Could not find expected xformers import in {TARGET}")

    TARGET.write_text(text.replace(ORIGINAL, PATCHED))
    print(f"Patched UniDepth xformers fallback in {TARGET}")


if __name__ == "__main__":
    main()

