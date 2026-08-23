"""
Strided DDPM reverse sampler for conditioned models.

Same posterior math as root ddpm.py's DDPM.p_sample_step (effective
alpha/beta for a multi-step jump, x0 clamp to +-1.5, noise scaled by
noise_std), but the noise prediction comes from a caller-supplied
eps_fn(xt, t) that closes over whatever conditioning the model needs
(observation tokens + mask for TimeCondUNet). Striding is exact within the
DDPM framework: the posterior only depends on alpha_bar at the two jump
endpoints, not on the individual betas between them.
"""

import numpy as np
import torch


@torch.no_grad()
def ddpm_sample(eps_fn, diffusion, land_mask: np.ndarray,
                stride: int = 10, device: str = "cpu") -> torch.Tensor:
    """
    Draw one field sample by walking t = T-1 -> 0 in stride-sized jumps.

    Args:
        eps_fn:    callable (xt (1,2,H,W), t (1,) long) -> predicted noise
                   (1,2,H,W); closes over the model and its conditioning
        diffusion: root ddpm.DDPM instance (alpha_bar, T, noise_std)
        land_mask: (H, W) bool, True = land (zeroed every step)
        stride:    timestep jump size (10 -> 100 model calls for T=1000)
        device:    torch device

    Returns:
        x0: (2, H, W) float32 tensor on CPU, land = 0
    """
    H, W = land_mask.shape
    ocean = torch.from_numpy(~land_mask).float().to(device)[None, None]

    xt = torch.randn(1, 2, H, W, device=device) * diffusion.noise_std * ocean

    timesteps = list(range(0, diffusion.T, stride))   # ascending, then reversed
    for i in reversed(range(len(timesteps))):
        t_int      = timesteps[i]
        t_prev_int = timesteps[i - 1] if i > 0 else 0

        t = torch.full((1,), t_int, device=device, dtype=torch.long)
        pred_noise = eps_fn(xt, t)

        ab = diffusion.alpha_bar[t_int]
        ab_prev = (diffusion.alpha_bar[t_prev_int] if t_prev_int > 0
                   else torch.tensor(1.0, device=device))

        # Predicted x0 (clipped), as in DDPM.p_sample_step
        x0_pred = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred = x0_pred.clamp(-1.5, 1.5)

        if t_int == 0:
            xt = x0_pred * ocean
            break

        # Effective alpha/beta for the (possibly multi-step) jump
        alpha_eff = ab / ab_prev
        beta_eff  = 1.0 - alpha_eff

        # DDPM posterior mean and variance
        coef1 = ab_prev.sqrt() * beta_eff / (1.0 - ab)
        coef2 = alpha_eff.sqrt() * (1.0 - ab_prev) / (1.0 - ab)
        mean  = coef1 * x0_pred + coef2 * xt

        var = (1.0 - ab_prev) / (1.0 - ab) * beta_eff
        xt  = mean + var.sqrt() * torch.randn_like(xt) * diffusion.noise_std
        xt  = xt * ocean

    return xt[0].cpu()
