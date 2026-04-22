"""Helmholtz-split noise schedule for DDPM.

Vendored from the research repo (ddpm/utils/helmholtz_split.py).
No external project dependencies — pure PyTorch.
"""

import torch


def helmholtz_decompose(uv: torch.Tensor):
    """Decompose (B,2,H,W) velocity into solenoidal + irrotational via FFT."""
    u = uv[:, 0]
    v = uv[:, 1]

    u_f = torch.fft.fft2(u)
    v_f = torch.fft.fft2(v)

    H, W = u.shape[-2], u.shape[-1]
    ky = torch.fft.fftfreq(H, device=uv.device).reshape(1, H, 1)
    kx = torch.fft.fftfreq(W, device=uv.device).reshape(1, 1, W)
    k2 = kx ** 2 + ky ** 2
    k2_safe = k2.clone()
    k2_safe[:, 0, 0] = 1.0

    div_f = u_f * kx + v_f * ky
    phi_f = div_f / k2_safe

    u_irr_f = phi_f * kx
    v_irr_f = phi_f * ky

    u_irr = torch.fft.ifft2(u_irr_f).real
    v_irr = torch.fft.ifft2(v_irr_f).real

    u_sol = u - u_irr
    v_sol = v - v_irr

    uv_sol = torch.stack([u_sol, v_sol], dim=1)
    uv_irr = torch.stack([u_irr, v_irr], dim=1)
    return uv_sol, uv_irr


def generate_solenoidal_noise(shape, device):
    raw = torch.randn(shape, device=device)
    sol, _ = helmholtz_decompose(raw)
    std = sol.std()
    if std > 1e-8:
        sol = sol / std
    return sol


def generate_irrotational_noise(shape, device):
    raw = torch.randn(shape, device=device)
    _, irr = helmholtz_decompose(raw)
    std = irr.std()
    if std > 1e-8:
        irr = irr / std
    return irr


class HelmholtzSplitSchedule:
    """Two independent linear-beta DDPM schedules for Helmholtz components."""

    def __init__(self, n_steps=250, min_beta=1e-4, max_beta=0.02,
                 irr_speed=2.0, device=None):
        self.n_steps = n_steps
        self.device = device

        self.betas_sol = torch.linspace(min_beta, max_beta, n_steps, device=device)
        self.alphas_sol = 1.0 - self.betas_sol
        self.alpha_bars_sol = torch.cumprod(self.alphas_sol, dim=0)

        max_beta_irr = min(max_beta * irr_speed, 0.999)
        self.betas_irr = torch.linspace(min_beta, max_beta_irr, n_steps, device=device)
        self.alphas_irr = 1.0 - self.betas_irr
        self.alpha_bars_irr = torch.cumprod(self.alphas_irr, dim=0)

    def q_sample(self, x0, t, eps_sol=None, eps_irr=None):
        B = x0.shape[0]
        x0_sol, x0_irr = helmholtz_decompose(x0)

        if eps_sol is None:
            eps_sol = generate_solenoidal_noise(x0.shape, x0.device)
        if eps_irr is None:
            eps_irr = generate_irrotational_noise(x0.shape, x0.device)

        abar_sol = self.alpha_bars_sol[t].reshape(B, 1, 1, 1)
        abar_irr = self.alpha_bars_irr[t].reshape(B, 1, 1, 1)

        x_t_sol = abar_sol.sqrt() * x0_sol + (1 - abar_sol).sqrt() * eps_sol
        x_t_irr = abar_irr.sqrt() * x0_irr + (1 - abar_irr).sqrt() * eps_irr

        x_t = x_t_sol + x_t_irr
        return x_t, eps_sol, eps_irr

    def p_step(self, x_t, x0_pred, t, noise_sol=None, noise_irr=None):
        B = x_t.shape[0]

        xt_sol, xt_irr = helmholtz_decompose(x_t)
        x0_sol, x0_irr = helmholtz_decompose(x0_pred)

        mu_sol, sigma_sol = self._component_posterior(
            xt_sol, x0_sol, t,
            self.alphas_sol, self.alpha_bars_sol, self.betas_sol,
        )
        mu_irr, sigma_irr = self._component_posterior(
            xt_irr, x0_irr, t,
            self.alphas_irr, self.alpha_bars_irr, self.betas_irr,
        )

        if noise_sol is None:
            noise_sol = generate_solenoidal_noise(x_t.shape, x_t.device)
        if noise_irr is None:
            noise_irr = generate_irrotational_noise(x_t.shape, x_t.device)

        mask_t0 = (t == 0).float().reshape(B, 1, 1, 1)
        x_prev_sol = mu_sol + (1 - mask_t0) * sigma_sol * noise_sol
        x_prev_irr = mu_irr + (1 - mask_t0) * sigma_irr * noise_irr

        return x_prev_sol + x_prev_irr

    @staticmethod
    def _component_posterior(x_t, x0, t, alphas, alpha_bars, betas):
        B = x_t.shape[0]

        alpha_t = alphas[t].reshape(B, 1, 1, 1)
        abar_t = alpha_bars[t].reshape(B, 1, 1, 1)
        beta_t = betas[t].reshape(B, 1, 1, 1)

        t_prev = (t - 1).clamp(min=0)
        abar_prev = alpha_bars[t_prev].reshape(B, 1, 1, 1)
        abar_prev = torch.where(
            t.reshape(B, 1, 1, 1) == 0,
            torch.ones_like(abar_prev),
            abar_prev,
        )

        coeff_x0 = (abar_prev.sqrt() * beta_t) / (1 - abar_t)
        coeff_xt = (alpha_t.sqrt() * (1 - abar_prev)) / (1 - abar_t)
        mu = coeff_x0 * x0 + coeff_xt * x_t

        beta_tilde = ((1 - abar_prev) / (1 - abar_t)) * beta_t
        sigma = beta_tilde.sqrt()

        return mu, sigma
