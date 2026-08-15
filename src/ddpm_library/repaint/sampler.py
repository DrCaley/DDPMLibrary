"""Guided samplers for the RePaint model: DPS and MCG, with conditioning support.

Ported from the research code (``Linear Best Model - Not Time Conditioned/
run_mcg_dps_z004.py``) with one change: the ``cond`` tensor (temporal priors) is
threaded through to the network, which the original scripts did not need because
that model was unconditional. The model and diffusion classes already accepted a
``cond`` argument, so nothing else changes.

NOTE ON PROVENANCE. The exact script that produced the published time-conditioned
numbers (``run_mcg_dps_z004_time_cond.py``, referenced in the research code's own
comments) was not included in the repository. This is a faithful reconstruction of
it from the unconditional sampler plus the model's conditioning hook. It has been
validated by reproducing the published RMSE -- see ``scripts/validate_repaint.py``.

Paradigm note: unlike CorrDiff, this model is NOT trained on the observations. Its
conditioning is only the temporal priors; the sparse measurements are imposed at
sampling time by guidance (a likelihood gradient) plus, for MCG, hard replacement
of the observed cells. That is a genuinely different approach, which is exactly
why it is worth comparing head-to-head.
"""

from __future__ import annotations

import numpy as np
import torch


def _prep(x0_known, path_mask, land_mask, device):
    x0_known_t = x0_known.unsqueeze(0).to(device)
    known_t = torch.from_numpy(np.asarray(path_mask, bool)).float().to(device)[None, None]
    land_t = torch.from_numpy(np.asarray(land_mask, bool)).float().to(device)[None, None]
    return x0_known_t, known_t, 1.0 - land_t


@torch.enable_grad()
def dps_infer(model, diffusion, x0_known, path_mask, land_mask, cond=None,
              device="cpu", stride=1, step_size=0.04):
    """Diffusion Posterior Sampling: a likelihood gradient steers every reverse step.

    x0_known : (2, H, W) tensor, observed values on the path and 0 elsewhere,
               in PHYSICAL m/s (this model is not z-scored).
    cond     : (1, cond_ch, H, W) temporal priors, or None for an unconditional model.
    Returns  : (2, H, W) numpy array, m/s.
    """
    H, W = x0_known.shape[1:]
    x0_known_t, known_t, ocean_t = _prep(x0_known, path_mask, land_mask, device)
    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        xt_in = xt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        pred_noise = model(xt_in, t_vec, cond) if cond is not None else model(xt_in, t_vec)
        ab = diffusion.alpha_bar[t_int]
        x0_hat = ((xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()).clamp(-1.5, 1.5)

        residual = known_t * (x0_hat - x0_known_t)
        norm_sq = (residual ** 2).sum()
        grad = torch.autograd.grad(norm_sq, xt_in)[0]

        with torch.no_grad():
            xt_next = diffusion.p_sample_step(model, xt_in.detach(), t_int, t_prev_int,
                                              cond=cond)
            norm = norm_sq.sqrt().item() + 1e-8
            xt = (xt_next - (step_size / norm) * grad.detach()) * ocean_t

    return xt.squeeze(0).detach().cpu().numpy()


@torch.enable_grad()
def mcg_infer(model, diffusion, x0_known, path_mask, land_mask, cond=None,
              device="cpu", stride=1, step_size=0.04):
    """Manifold-Constrained Gradient: DPS guidance PLUS hard replacement of the
    observed cells with a correctly-noised version of the measurements at each step.

    Same arguments as :func:`dps_infer`.
    """
    H, W = x0_known.shape[1:]
    x0_known_t, known_t, ocean_t = _prep(x0_known, path_mask, land_mask, device)
    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean_t
    timesteps = list(range(0, diffusion.T, stride))

    for i in reversed(range(len(timesteps))):
        t_int = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        xt_in = xt.detach().requires_grad_(True)
        t_vec = torch.full((1,), t_int, device=device, dtype=torch.long)

        pred_noise = model(xt_in, t_vec, cond) if cond is not None else model(xt_in, t_vec)
        ab = diffusion.alpha_bar[t_int]
        x0_hat = ((xt_in - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()).clamp(-1.5, 1.5)

        residual = known_t * (x0_hat - x0_known_t)
        norm_sq = (residual ** 2).sum()
        grad = torch.autograd.grad(norm_sq, xt_in)[0]

        with torch.no_grad():
            xt_unknown = diffusion.p_sample_step(model, xt_in.detach(), t_int, t_prev_int,
                                                 cond=cond)
            norm = norm_sq.sqrt().item() + 1e-8
            xt_unknown = xt_unknown - (step_size / norm) * grad.detach()

            t_prev_t = torch.full((1,), t_prev_int, device=device, dtype=torch.long)
            xt_known_noisy, _ = diffusion.q_sample(x0_known_t, t_prev_t)
            xt = (known_t * xt_known_noisy + (1.0 - known_t) * xt_unknown) * ocean_t

    return xt.squeeze(0).detach().cpu().numpy()


SAMPLERS = {"dps": dps_infer, "mcg": mcg_infer}
