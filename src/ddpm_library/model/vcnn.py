"""Voronoi-CNN baseline (Fukami et al. 2021) — vendored from the research repo.

Original location: scripts/voronoi_cnn_model.py in the diffusionInpainting
research repo. This is a faithful copy of just the parts needed for
inference: the U-Net-lite architecture and the Voronoi pre-processing
helper. Training utilities are not vendored.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree


def build_voronoi_input(
    vel: np.ndarray,
    mask: np.ndarray,
    ocean_mask: np.ndarray,
) -> np.ndarray:
    """Build the 5-channel Voronoi input from sparse observations.

    Parameters
    ----------
    vel : np.ndarray, shape (2, H, W)
        Velocity field. Only entries where ``mask`` is 1 are used.
    mask : np.ndarray, shape (H, W)
        1 where observed, 0 where missing (opposite of the inpaint
        convention used elsewhere in this library).
    ocean_mask : np.ndarray, shape (H, W)
        1 where ocean, 0 where land.

    Returns
    -------
    np.ndarray, shape (5, H, W), dtype float32
        Channels: voronoi_u, voronoi_v, normalised distance, sensor mask,
        ocean mask.
    """
    H, W = mask.shape
    ky, kx = np.where(mask > 0.5)
    if len(ky) == 0:
        return np.zeros((5, H, W), dtype=np.float32)

    obs_coords = np.stack([ky, kx], axis=1).astype(np.float64)
    tree = cKDTree(obs_coords)

    gy, gx = np.mgrid[0:H, 0:W]
    grid_coords = np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64)
    dist, idx = tree.query(grid_coords, k=1)

    dist = dist.reshape(H, W)
    idx = idx.reshape(H, W)

    obs_u = vel[0, ky, kx]
    obs_v = vel[1, ky, kx]
    voronoi_u = obs_u[idx]
    voronoi_v = obs_v[idx]

    max_dist = dist.max() + 1e-8
    dist_norm = dist / max_dist

    voronoi_u *= ocean_mask
    voronoi_v *= ocean_mask
    dist_norm *= ocean_mask

    sensor_mask = mask.astype(np.float32)

    return np.stack(
        [voronoi_u, voronoi_v, dist_norm, sensor_mask, ocean_mask],
        axis=0,
    ).astype(np.float32)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VoronoiCNN(nn.Module):
    """U-Net-lite encoder–decoder; Voronoi (5-ch) → velocity field (2-ch)."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 2,
        base_ch: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.depth = depth

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(_ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        self.bottleneck = _ConvBlock(ch, ch * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = ch * 2
        for i in range(depth - 1, -1, -1):
            out_ch = base_ch * (2 ** i)
            self.upconvs.append(
                nn.ConvTranspose2d(ch, out_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(_ConvBlock(out_ch * 2, out_ch))
            ch = out_ch

        self.head = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]
        factor = 2 ** self.depth
        pad_h = (factor - orig_h % factor) % factor
        pad_w = (factor - orig_w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.head(x)
        return x[:, :, :orig_h, :orig_w]
