"""Neural network building blocks: sinusoidal embeddings, residual blocks,
self-attention, ResAttnBlock.

Vendored from the research repo
(ddpm/neural_networks/unets/unet_xl_attn.py). Only the symbols needed by
the Helmholtz split-head UNet are copied here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(n: int, d: int) -> torch.Tensor:
    """Fixed sinusoidal position embedding (timestep → vector)."""
    emb = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    t = torch.arange(n).unsqueeze(1)
    emb[:, 0::2] = torch.sin(t * wk[0::2])
    emb[:, 1::2] = torch.cos(t * wk[1::2])
    return emb


class ResBlock(nn.Module):
    """Residual block with GroupNorm + AdaGN time-embedding modulation."""

    def __init__(self, in_c: int, out_c: int, time_emb_dim: int,
                 num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_c), in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_c), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.SiLU()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * out_c),
        )

        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        ts = self.time_mlp(t_emb)
        scale, shift = ts.chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over spatial dims (H*W sequence length)."""

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.permute(0, 1, 3, 2)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return x + out


class ResAttnBlock(nn.Module):
    """ResBlock optionally followed by self-attention."""

    def __init__(self, in_c: int, out_c: int, time_emb_dim: int,
                 use_attn: bool = False, num_heads: int = 4):
        super().__init__()
        self.res = ResBlock(in_c, out_c, time_emb_dim)
        self.attn = SelfAttention2d(out_c, num_heads=num_heads) if use_attn else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.res(x, t_emb)
        if self.attn is not None:
            h = self.attn(h)
        return h
