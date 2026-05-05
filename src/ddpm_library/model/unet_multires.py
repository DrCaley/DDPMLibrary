"""Multi-resolution FiLM-conditioned Helmholtz dual-head UNet.

Vendored from the research repo
(ddpm/neural_networks/unets/unet_helmholtz_split_film_multires.py).

Same backbone as MyUNet_Helmholtz_Split_FiLM but replaces the CNN
conditioning encoder with a Feature Pyramid Network (FPN) encoder that
directly processes sparse observations at multiple scales — no Voronoi
fill needed.

Conditioning input is SPARSE observations
[missing_mask(1), sparse_u(1), sparse_v(1)] (5-channel total with x_t).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_film import MyUNet_Helmholtz_Split_FiLM


class MultiResCondEncoder(nn.Module):
    """Feature Pyramid conditioning encoder for sparse observations."""

    def __init__(self, ch=(64, 128, 256, 256), use_distance_field=False,
                 use_bathymetry=False):
        super().__init__()
        self.use_distance_field = use_distance_field
        self.use_bathymetry = use_bathymetry
        pool_ch = 3
        if use_distance_field:
            pool_ch += 1
        if use_bathymetry:
            pool_ch += 1

        self.proc5 = nn.Sequential(
            nn.Conv2d(pool_ch, ch[3], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.SiLU(),
        )
        self.proc4 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[3], ch[3], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.SiLU(),
        )
        self.proc3 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[3], ch[2], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[2], ch[2], 3, 1, 1), nn.SiLU(),
        )
        self.proc2 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[2], ch[1], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[1], ch[1], 3, 1, 1), nn.SiLU(),
        )
        self.proc1 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[1], ch[0], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[0], ch[0], 3, 1, 1), nn.SiLU(),
        )

    @staticmethod
    def _pool_sparse(obs_uv, known_mask, target_h, target_w):
        H, W = obs_uv.shape[-2:]
        kh, kw = H // target_h, W // target_w

        if kh == 1 and kw == 1:
            density = known_mask
            normalized = obs_uv
        else:
            density = F.avg_pool2d(known_mask, (kh, kw))
            pooled = F.avg_pool2d(obs_uv, (kh, kw))
            safe_density = density.clamp(min=1e-8)
            normalized = pooled / safe_density
            has_obs = (density > 1e-7).float()
            normalized = normalized * has_obs

        return torch.cat([normalized, density], dim=1)

    @staticmethod
    def _pool_dense(field, target_h, target_w):
        H, W = field.shape[-2:]
        kh, kw = H // target_h, W // target_w
        if kh == 1 and kw == 1:
            return field
        return F.avg_pool2d(field, (kh, kw))

    def forward(self, cond):
        missing_mask = cond[:, :1]
        known_mask = 1.0 - missing_mask
        obs_uv = cond[:, 1:3]

        p5 = self._pool_sparse(obs_uv, known_mask, 4, 8)
        p4 = self._pool_sparse(obs_uv, known_mask, 8, 16)
        p3 = self._pool_sparse(obs_uv, known_mask, 16, 32)
        p2 = self._pool_sparse(obs_uv, known_mask, 32, 64)
        p1 = self._pool_sparse(obs_uv, known_mask, 64, 128)

        dense_idx = 3
        if self.use_distance_field:
            dist = cond[:, dense_idx:dense_idx + 1]
            dense_idx += 1
            p5 = torch.cat([p5, self._pool_dense(dist, 4, 8)], dim=1)
            p4 = torch.cat([p4, self._pool_dense(dist, 8, 16)], dim=1)
            p3 = torch.cat([p3, self._pool_dense(dist, 16, 32)], dim=1)
            p2 = torch.cat([p2, self._pool_dense(dist, 32, 64)], dim=1)
            p1 = torch.cat([p1, self._pool_dense(dist, 64, 128)], dim=1)

        if self.use_bathymetry:
            bathy = cond[:, dense_idx:dense_idx + 1]
            dense_idx += 1
            p5 = torch.cat([p5, self._pool_dense(bathy, 4, 8)], dim=1)
            p4 = torch.cat([p4, self._pool_dense(bathy, 8, 16)], dim=1)
            p3 = torch.cat([p3, self._pool_dense(bathy, 16, 32)], dim=1)
            p2 = torch.cat([p2, self._pool_dense(bathy, 32, 64)], dim=1)
            p1 = torch.cat([p1, self._pool_dense(bathy, 64, 128)], dim=1)

        c5 = self.proc5(p5)

        c5_up = F.interpolate(c5, size=(8, 16), mode='bilinear', align_corners=False)
        c4 = self.proc4(torch.cat([p4, c5_up], dim=1))

        c4_up = F.interpolate(c4, size=(16, 32), mode='bilinear', align_corners=False)
        c3 = self.proc3(torch.cat([p3, c4_up], dim=1))

        c3_up = F.interpolate(c3, size=(32, 64), mode='bilinear', align_corners=False)
        c2 = self.proc2(torch.cat([p2, c3_up], dim=1))

        c2_up = F.interpolate(c2, size=(64, 128), mode='bilinear', align_corners=False)
        c1 = self.proc1(torch.cat([p1, c2_up], dim=1))

        return c1, c2, c3, c4, c5


class MyUNet_Helmholtz_Split_FiLM_MultiRes(MyUNet_Helmholtz_Split_FiLM):
    """FiLM-conditioned Helmholtz UNet with multi-resolution sparse conditioning."""

    def __init__(self, n_steps=1000, time_emb_dim=256, in_channels=5,
                 detach_heads: bool = False, use_distance_field: bool = False,
                 use_bathymetry: bool = False):
        super().__init__(
            n_steps=n_steps, time_emb_dim=time_emb_dim, in_channels=in_channels,
            detach_heads=detach_heads,
        )
        self.cond_encoder = MultiResCondEncoder(
            ch=[64, 128, 256, 256], use_distance_field=use_distance_field,
            use_bathymetry=use_bathymetry,
        )
