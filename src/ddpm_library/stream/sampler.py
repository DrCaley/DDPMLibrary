"""Posterior samplers for the conditional stream-function DDPM.

Two paths, both operating on the same trained model:

* :func:`ensemble_infer` — the DDPM ancestral sampler used by the research
  uncertainty-map pipeline. Sequential per-member draws with the SAME seeding
  convention as the research repo, so results are byte-for-byte reproducible.
  This is the default (matches ``inference_steps=100`` in the research probes).

* :func:`dpmpp_ensemble` — DPM-Solver++(2M), batched over draws. Deterministic
  ODE; reaches comparable quality in far fewer steps. Optional fast path.
"""

from __future__ import annotations

import numpy as np
import torch

from .diffusion import eps_wrapper_for, x0_from_output


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _x0_hat(stream_model, diffusion, xt, t_t, cond, pred_type):
    out = stream_model(xt, t_t, cond)
    return x0_from_output(diffusion, xt, out, t_t, pred_type)


def _init_latent(diffusion, n, H, W, ocean, device, seed):
    torch.manual_seed(seed)
    x = diffusion._sample_noise(torch.empty(n, 2, H, W, device=device))
    return x * diffusion.noise_scale * ocean


def _members_to_numpy(xt, ocean):
    xt = xt * ocean
    return [xt[i].detach().cpu().numpy().astype(np.float32) for i in range(xt.shape[0])]


# ---------------------------------------------------------------------------
# DDPM ancestral sampler (parity path — matches research ensemble_infer)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_one(stream_model, diffusion, cond, land_np, *, inference_steps,
               device, seed, pred_type):
    """One conditional posterior draw via the DDPM ancestral reverse chain.

    Replicates the no-CFG path of the research ``sample_capture`` exactly: seed
    the RNG, draw x_T, iterate ``p_sample_step`` over the subsampled schedule,
    then take one final x̂₀ estimate at t=0 (the research code returns
    ``frames[-1][2] = x0hat(final_xt, t=0)``, not the raw final ``xt``).
    Returns a (2, H, W) numpy array.
    """
    H, W = land_np.shape
    ocean_f = torch.from_numpy(~land_np).float().to(device)[None, None]
    cond_b = cond.unsqueeze(0).to(device)
    eps_model = eps_wrapper_for(stream_model, diffusion, pred_type,
                                cond=cond_b).to(device)

    torch.manual_seed(seed)
    xt = diffusion._sample_noise(torch.empty(1, 2, H, W, device=device))
    xt = xt * diffusion.noise_scale * ocean_f

    for t_int, t_prev_int in diffusion.build_inference_schedule(inference_steps):
        xt = diffusion.p_sample_step(eps_model, xt, t_int, t_prev_int) * ocean_f

    # Final x̂₀ estimate at t=0 (matches research sample_capture `final_pred`).
    t0 = torch.zeros(1, device=device, dtype=torch.long)
    out = stream_model(xt, t0, cond_b)
    x0 = x0_from_output(diffusion, xt, out, t0, pred_type)
    return (x0 * ocean_f).squeeze(0).cpu().numpy().astype(np.float32)


@torch.no_grad()
def ensemble_infer(stream_model, diffusion, cond, land_np, *, n_members,
                   inference_steps, device, base_seed=0, pred_type=None):
    """Draw ``n_members`` diverse conditional samples from the SAME cond.

    Per-member seed ``(base_seed + 1) * 100003 + k`` matches the research
    ``ensemble_infer`` exactly. Returns a list of (2, H, W) numpy arrays.
    """
    members = []
    for k in range(max(1, n_members)):
        seed = (base_seed + 1) * 100003 + k
        members.append(sample_one(
            stream_model, diffusion, cond, land_np,
            inference_steps=inference_steps, device=device,
            seed=seed, pred_type=pred_type))
    return members


# ---------------------------------------------------------------------------
# DPM-Solver++(2M) — batched, deterministic ODE (fast option)
# ---------------------------------------------------------------------------

@torch.no_grad()
def dpmpp_ensemble(stream_model, diffusion, cond, land_np, *, n_members=8,
                   inference_steps=25, device="cpu", seed=0, pred_type=None):
    """DPM-Solver++(2M) multistep sampler in data-prediction (x̂₀) form.

    All ``n_members`` draws are denoised in a SINGLE batched forward per step
    (diversity from the initial-noise seed). Uses the noise-scaled marginal
    std σ_t = noise_scale · √(1−ᾱ_t) so the log-SNR schedule matches training.
    Returns a list of (2, H, W) numpy arrays.
    """
    H, W = land_np.shape
    ocean = torch.from_numpy(~land_np).float().to(device)[None, None]
    cond_n = cond.unsqueeze(0).to(device).expand(n_members, -1, -1, -1)
    ns = diffusion.noise_scale

    ab = diffusion.alpha_bar
    alpha = ab.sqrt()
    sigma = ns * (1.0 - ab).sqrt()
    lamb = torch.log(alpha) - torch.log(sigma)

    schedule = diffusion.build_inference_schedule(inference_steps)
    xt = _init_latent(diffusion, n_members, H, W, ocean, device, seed)

    prev_x0, prev_lam = None, None
    for t_int, t_prev_int in schedule:
        t_t = torch.full((n_members,), t_int, device=device, dtype=torch.long)
        x0 = _x0_hat(stream_model, diffusion, xt, t_t, cond_n, pred_type)
        x0 = (x0 * ocean).clamp(-3.0 * ns, 3.0 * ns)

        if t_prev_int < 0:
            xt = x0
            break

        lam_s, lam_t = lamb[t_int], lamb[t_prev_int]
        h = lam_t - lam_s
        sig_ratio = sigma[t_prev_int] / sigma[t_int]
        a_t = alpha[t_prev_int]
        phi = torch.expm1(-h)

        if prev_x0 is None:
            D = x0
        else:
            r = (lam_s - prev_lam) / h
            D = (1.0 + 0.5 / r) * x0 - (0.5 / r) * prev_x0

        xt = (sig_ratio * xt - a_t * phi * D) * ocean
        prev_x0, prev_lam = x0, lam_s

    return _members_to_numpy(xt, ocean)
