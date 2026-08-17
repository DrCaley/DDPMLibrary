"""v-prediction diffusion core for the CorrDiff residual model.

Vendored verbatim from the research pipeline (``Conditional DDPM/train_corrdiff.py``)
so inference here is bit-for-bit identical to the published evaluation:

  * cosine alpha-bar schedule (Nichol & Dhariwal 2021)
  * v-parameterisation (Salimans & Ho 2022) -- the network predicts
    ``v = a*eps - s*x0``; ``x0`` is recovered as ``a*x_t - s*v``
  * deterministic DDIM sampling (Song et al. 2021)

Nothing in this module depends on the research repo.
"""

from __future__ import annotations

import math

import torch


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule; returns alpha_bar for t = 1..T, shape (T,)."""
    t = torch.linspace(0, T, T + 1) / T
    ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (ab / ab[0])[1:]


class VDiffusion:
    """Minimal v-prediction diffusion helper (inference subset).

    Only the quantities needed for sampling are kept; the training-time helpers
    (``q_sample``, ``v_target``, ``minsnr_w``) are intentionally omitted since
    this library is inference-only.
    """

    def __init__(self, T: int = 1000, device: str | torch.device = "cpu"):
        self.T = T
        self.ab = cosine_alpha_bar(T).to(device)
        self.device = device

    def x0_from_v(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Recover the clean signal from the predicted v at timestep t."""
        a = self.ab[t].sqrt().view(-1, 1, 1, 1)
        s = (1 - self.ab[t]).sqrt().view(-1, 1, 1, 1)
        return a * xt - s * v


@torch.no_grad()
def ddim_sample_residual(model, cond_aug, ocean_t, diff, device,
                         n_draws: int = 8, steps: int = 50, seed: int = 0): 
    """Sample ``n_draws`` residual fields for ONE conditioning stack.

    ``cond_aug`` is the full conditioning tensor (1, total_cond, H, W) -- it already
    includes the deterministic mean field, the distance channel and (for the
    sensor-noise model) the sigma channel. Draws differ only in the initial noise,
    which is seeded from ``seed`` so results are reproducible.

    Returns (n_draws, 2, H, W): residuals to be ADDED to the deterministic mean.
    """
    B = n_draws
    H, W = ocean_t.shape[-2:]
    condB = cond_aug.expand(B, -1, -1, -1).to(device)
    oc = ocean_t.to(device).float()
    
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)
    else:
        g = None

    x = torch.randn(B, 2, H, W, generator=g, device=device) * oc
    ts = torch.linspace(diff.T - 1, 0, steps, device=device).long()
    for i in range(steps):
        t = ts[i]
        tb = t.repeat(B)
        v = model(torch.cat([x, condB], dim=1), tb)
        x0 = diff.x0_from_v(x, v, tb) * oc
        a_t = diff.ab[t].sqrt()
        s_t = (1 - diff.ab[t]).sqrt()
        eps = (x - a_t * x0) / s_t.clamp(min=1e-4)
        if i < steps - 1:
            t2 = ts[i + 1]
            a2 = diff.ab[t2].sqrt()
            s2 = (1 - diff.ab[t2]).sqrt()
            x = (a2 * x0 + s2 * eps) * oc
        else:
            x = x0
    return x
