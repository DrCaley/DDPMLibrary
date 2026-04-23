"""FiLM-conditioned Helmholtz dual-head UNet (base class).

Vendored from the research repo
(ddpm/neural_networks/unets/unet_helmholtz_split_film.py). Only the import
path of `sinusoidal_embedding` / `ResAttnBlock` is changed to the
library-local `unet_blocks` module.

Input:  (N, 5, H, W) = [x_t(2ch), mask(1ch), cond_u(1ch), cond_v(1ch)]
Output: (N, 2, H, W) — v = curl(ψ) + grad(φ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import sinusoidal_embedding, ResAttnBlock


# ─── FiLM building blocks ──────────────────────────────────────────────────


class FiLMLayer(nn.Module):
    """Spatial FiLM with GroupNorm (per-pixel modulation)."""

    def __init__(self, cond_channels, feature_channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, feature_channels)
        self.scale_conv = nn.Conv2d(cond_channels, feature_channels, 1)
        self.shift_conv = nn.Conv2d(cond_channels, feature_channels, 1)

        nn.init.zeros_(self.scale_conv.weight)
        nn.init.zeros_(self.scale_conv.bias)
        nn.init.zeros_(self.shift_conv.weight)
        nn.init.zeros_(self.shift_conv.bias)

    def forward(self, h, cond):
        gamma = self.scale_conv(cond)
        beta = self.shift_conv(cond)
        return (1 + gamma) * self.norm(h) + beta


class HelmholtzCondEncoder(nn.Module):
    """Encodes conditioning into multi-scale features (CNN version)."""

    def __init__(self, ch=(64, 128, 256, 256)):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[0], ch[0], 3, 1, 1), nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch[1], ch[1], 3, 1, 1), nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch[2], ch[2], 3, 1, 1), nn.SiLU(),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.SiLU(),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(ch[3], ch[3], 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.SiLU(),
        )

    def forward(self, cond):
        c1 = self.enc1(cond)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)
        c5 = self.enc5(c4)
        return c1, c2, c3, c4, c5


# ─── Main model ────────────────────────────────────────────────────────────


class MyUNet_Helmholtz_Split_FiLM(nn.Module):
    """AdaGN-FiLM-conditioned Helmholtz UNet with independent high-res decoders."""

    def __init__(self, n_steps: int = 1000, time_emb_dim: int = 256,
                 in_channels: int = 5, detach_heads: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.detach_heads = detach_heads

        ch = [64, 128, 256, 256]

        self.cond_encoder = HelmholtzCondEncoder(ch=ch)

        self.film_enc1 = FiLMLayer(cond_channels=ch[0], feature_channels=ch[0])
        self.film_enc2 = FiLMLayer(cond_channels=ch[1], feature_channels=ch[1])
        self.film_enc3 = FiLMLayer(cond_channels=ch[2], feature_channels=ch[2])
        self.film_enc4 = FiLMLayer(cond_channels=ch[3], feature_channels=ch[3])
        self.film_mid = FiLMLayer(cond_channels=ch[3], feature_channels=ch[3])
        self.film_dec4 = FiLMLayer(cond_channels=ch[3], feature_channels=ch[2])
        self.film_dec3 = FiLMLayer(cond_channels=ch[2], feature_channels=ch[1])
        self.film_dec2_psi = FiLMLayer(cond_channels=ch[1], feature_channels=ch[0])
        self.film_dec1_psi = FiLMLayer(cond_channels=ch[0], feature_channels=ch[0])
        self.film_dec2_phi = FiLMLayer(cond_channels=ch[1], feature_channels=ch[0])
        self.film_dec1_phi = FiLMLayer(cond_channels=ch[0], feature_channels=ch[0])

        # Time embedding
        self.time_embed_table = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed_table.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed_table.requires_grad_(False)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Encoder (2-channel input: just x_t)
        self.enc1 = nn.ModuleList([
            ResAttnBlock(2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, 2, 1)

        self.enc2 = nn.ModuleList([
            ResAttnBlock(ch[0], ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[1], time_emb_dim, use_attn=False),
        ])
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, 2, 1)

        self.enc3 = nn.ModuleList([
            ResAttnBlock(ch[1], ch[2], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[2], ch[2], time_emb_dim, use_attn=True, num_heads=4),
        ])
        self.down3 = nn.Conv2d(ch[2], ch[2], 4, 2, 1)

        self.enc4 = nn.ModuleList([
            ResAttnBlock(ch[2], ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
        ])
        self.down4 = nn.Conv2d(ch[3], ch[3], 4, 2, 1)

        # Bottleneck
        self.mid = nn.ModuleList([
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
        ])

        # Shared low-res decoder
        self.up4 = nn.ConvTranspose2d(ch[3], ch[3], 4, 2, 1)
        self.dec4 = nn.ModuleList([
            ResAttnBlock(ch[3] * 2, ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[2], time_emb_dim, use_attn=True, num_heads=4),
        ])

        self.up3 = nn.ConvTranspose2d(ch[2], ch[2], 4, 2, 1)
        self.dec3 = nn.ModuleList([
            ResAttnBlock(ch[2] * 2, ch[2], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[2], ch[1], time_emb_dim, use_attn=True, num_heads=4),
        ])

        # ψ branch
        self.up2_psi = nn.ConvTranspose2d(ch[1], ch[1], 4, 2, 1)
        self.dec2_psi = nn.ModuleList([
            ResAttnBlock(ch[1] * 2, ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[0], time_emb_dim, use_attn=False),
        ])

        self.up1_psi = nn.ConvTranspose2d(ch[0], ch[0], 4, 2, 1)
        self.dec1_psi = nn.ModuleList([
            ResAttnBlock(ch[0] * 2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])

        self.psi_norm = nn.GroupNorm(8, ch[0])
        self.psi_act = nn.SiLU()
        self.psi_conv = nn.Conv2d(ch[0], 1, 3, 1, 1)

        # φ branch
        self.up2_phi = nn.ConvTranspose2d(ch[1], ch[1], 4, 2, 1)
        self.dec2_phi = nn.ModuleList([
            ResAttnBlock(ch[1] * 2, ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[0], time_emb_dim, use_attn=False),
        ])

        self.up1_phi = nn.ConvTranspose2d(ch[0], ch[0], 4, 2, 1)
        self.dec1_phi = nn.ModuleList([
            ResAttnBlock(ch[0] * 2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])

        self.phi_norm = nn.GroupNorm(8, ch[0])
        self.phi_act = nn.SiLU()
        self.phi_conv = nn.Conv2d(ch[0], 1, 3, 1, 1)

        nn.init.zeros_(self.phi_conv.weight)
        nn.init.zeros_(self.phi_conv.bias)

        # Gradient kernels
        dx = torch.tensor([[[[0.0, -0.5, 0.5]]]])
        dy = torch.tensor([[[[0.0], [-0.5], [0.5]]]])
        self.register_buffer("_dx_kernel", dx)
        self.register_buffer("_dy_kernel", dy)

        # Diagnostics (unused in inference but kept so state_dict layout matches)
        self.last_psi = None
        self.last_phi = None
        self.last_v_sol = None
        self.last_v_irr = None

    @staticmethod
    def _curl_from_streamfunction(psi: torch.Tensor) -> torch.Tensor:
        u = psi[:, 1:, :-1] - psi[:, :-1, :-1]
        v = -(psi[:, :-1, 1:] - psi[:, :-1, :-1])
        return torch.stack([u, v], dim=1)

    def _grad_potential(self, phi: torch.Tensor) -> torch.Tensor:
        u_irr = F.conv2d(phi, self._dx_kernel, padding=(0, 1))
        v_irr = F.conv2d(phi, self._dy_kernel, padding=(1, 0))
        return torch.cat([u_irr, v_irr], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        x_t = x[:, :2]
        cond = x[:, 2:]

        c1, c2, c3, c4, c5 = self.cond_encoder(cond)

        t_emb = self.time_embed_table(t)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        t_emb = self.time_mlp(t_emb)

        h = x_t
        for block in self.enc1:
            h = block(h, t_emb)
        h = self.film_enc1(h, c1)
        skip1 = h

        h = self.down1(h)
        for block in self.enc2:
            h = block(h, t_emb)
        h = self.film_enc2(h, c2)
        skip2 = h

        h = self.down2(h)
        for block in self.enc3:
            h = block(h, t_emb)
        h = self.film_enc3(h, c3)
        skip3 = h

        h = self.down3(h)
        for block in self.enc4:
            h = block(h, t_emb)
        h = self.film_enc4(h, c4)
        skip4 = h

        h = self.down4(h)

        for block in self.mid:
            h = block(h, t_emb)
        h = self.film_mid(h, c5)

        h = self.up4(h)
        h = torch.cat([skip4, h], dim=1)
        for block in self.dec4:
            h = block(h, t_emb)
        h = self.film_dec4(h, c4)

        h = self.up3(h)
        h = torch.cat([skip3, h], dim=1)
        for block in self.dec3:
            h = block(h, t_emb)
        h = self.film_dec3(h, c3)

        # ψ branch
        h_psi = self.up2_psi(h)
        h_psi = torch.cat([skip2, h_psi], dim=1)
        for block in self.dec2_psi:
            h_psi = block(h_psi, t_emb)
        h_psi = self.film_dec2_psi(h_psi, c2)

        h_psi = self.up1_psi(h_psi)
        h_psi = torch.cat([skip1, h_psi], dim=1)
        for block in self.dec1_psi:
            h_psi = block(h_psi, t_emb)
        h_psi = self.film_dec1_psi(h_psi, c1)

        psi = self.psi_conv(self.psi_act(self.psi_norm(h_psi)))
        psi = psi.squeeze(1)
        psi_padded = F.pad(psi, (0, 1, 0, 1), mode="constant", value=0.0)
        v_sol = self._curl_from_streamfunction(psi_padded)

        # φ branch
        h_phi = self.up2_phi(h)
        h_phi = torch.cat([skip2, h_phi], dim=1)
        for block in self.dec2_phi:
            h_phi = block(h_phi, t_emb)
        h_phi = self.film_dec2_phi(h_phi, c2)

        h_phi = self.up1_phi(h_phi)
        h_phi = torch.cat([skip1, h_phi], dim=1)
        for block in self.dec1_phi:
            h_phi = block(h_phi, t_emb)
        h_phi = self.film_dec1_phi(h_phi, c1)

        phi = self.phi_conv(self.phi_act(self.phi_norm(h_phi)))
        v_irr = self._grad_potential(phi)

        if self.detach_heads:
            return v_sol.detach() + v_irr.detach()
        return v_sol + v_irr
