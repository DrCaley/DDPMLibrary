"""Inference-only DDPM utilities for the stream-function pipeline.

This is a trimmed, self-contained copy of the research repo's
``DDPM/model/diffusion.py`` with the training/loss machinery removed (the
original imported ``loss_functions.py`` at module load, which does not ship
with this library). The reverse-process math — cosine schedule, div-free noise
sampling, ``build_inference_schedule`` and ``p_sample_step`` — is copied
verbatim so inference is byte-for-byte identical to the research pipeline.
"""

from __future__ import annotations

import math

import torch

from .div_free_noise import NOISE_TYPES, divergence_free_noise as _divergence_free_noise


class DDPM:
    """Denoising Diffusion Probabilistic Model — inference subset.

    Only the pieces used by posterior sampling are kept: the cosine noise
    schedule, the (optionally divergence-free) noise sampler, the subsampled
    inference schedule builder, and a single reverse step.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_schedule: str = "cosine",
        device: str = "cpu",
        noise_type: str = "gaussian",
        spectral_filter: torch.Tensor | None = None,
        noise_scale: float = 1.0,
    ):
        self.T = T
        self.device = device
        self.noise_scale = noise_scale

        if noise_type not in NOISE_TYPES:
            raise ValueError(
                f"noise_type must be one of {NOISE_TYPES}, got '{noise_type}'")
        self.noise_type = noise_type

        # Spectral filter for colored div-free noise (CPU tensor, or None)
        if spectral_filter is not None:
            self.spectral_filter = spectral_filter.cpu().float()
        else:
            self.spectral_filter = None

        betas = self._cosine_betas(T) if beta_schedule == "cosine" else \
            torch.linspace(1e-4, 0.02, T)

        self.betas = betas.to(device)
        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.alpha_bar = torch.cumprod(alphas, dim=0)           # ᾱ_t
        self.alpha_bar_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_bar[:-1]]
        )
        self.sqrt_ab = self.alpha_bar.sqrt()
        self.sqrt_one_mab = (1.0 - self.alpha_bar).sqrt()

    # ------------------------------------------------------------------
    # Noise sampler
    # ------------------------------------------------------------------

    def _sample_noise(self, like: torch.Tensor) -> torch.Tensor:
        """Return noise with the same shape/device as `like`."""
        if self.noise_type == "gaussian":
            return torch.randn_like(like)
        return _divergence_free_noise(
            like.shape,
            device=str(like.device),
            spectral_filter=self.spectral_filter,
        )

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def _cosine_betas(self, T: int, s: float = 0.008) -> torch.Tensor:
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        betas = 1.0 - ab[1:] / ab[:-1]
        return betas.clamp(0, 0.999)

    # ------------------------------------------------------------------
    # Inference schedule
    # ------------------------------------------------------------------

    def build_inference_schedule(self, n_steps: int) -> list[tuple[int, int]]:
        """Subsampled list of (t, t_prev) integer pairs, in reverse order.

        For T=1000, n_steps=100: [(999, 989), ..., (9, -1)]. t_prev == -1
        signals the final step (return x̂₀ directly).
        """
        step_size = self.T // n_steps
        ts = list(reversed(range(step_size - 1, self.T, step_size)))
        pairs = [(ts[i], ts[i + 1] if i + 1 < len(ts) else -1)
                 for i in range(len(ts))]
        return pairs

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------

    def p_sample_step(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t_int: int,
        t_prev_int: int = -1,
    ) -> torch.Tensor:
        """One DDPM reverse step, supporting non-consecutive (subsampled) schedules.

        `model` is an eps-predicting network (see :func:`eps_wrapper_for`).
        Returns x_{t_prev}; if t_prev_int < 0 returns the clamped x̂₀.
        """
        B = xt.shape[0]
        t = torch.full((B,), t_int, device=self.device, dtype=torch.long)

        pred_noise = model(xt, t)

        ab = self.alpha_bar[t_int]

        # Predicted x0 — clamp to ±3σ of the data (noise_scale ≈ data std)
        x0_pred = (xt - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred = x0_pred.clamp(-3.0 * self.noise_scale, 3.0 * self.noise_scale)

        if t_prev_int < 0:
            return x0_pred

        ab_prev = self.alpha_bar[t_prev_int]

        beta_eff = 1.0 - ab / ab_prev
        var = (1.0 - ab_prev) / (1.0 - ab) * beta_eff

        coef1 = ab_prev.sqrt() * beta_eff / (1.0 - ab)
        coef2 = (ab / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab)
        mean = coef1 * x0_pred + coef2 * xt

        return mean + var.sqrt() * self.noise_scale * self._sample_noise(xt)


# ---------------------------------------------------------------------------
# Inference adapters: stream-function model → epsilon-equivalent
# ---------------------------------------------------------------------------

class EpsFromStreamFn(torch.nn.Module):
    """Wrap an x0-prediction stream-function model to expose eps-equivalent output.

        x̂₀(x_t, t) = stream_model(x_t, t, cond)
        ε̂(x_t, t)  = (x_t − √ᾱ_t · x̂₀) / √(1 − ᾱ_t)
    """

    def __init__(self, stream_model: torch.nn.Module, diffusion: "DDPM",
                 cond: torch.Tensor | None = None):
        super().__init__()
        self.stream_model = stream_model
        self._ab = diffusion.alpha_bar
        self.cond = cond

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.cond is None:
            x0 = self.stream_model(xt, t)
        else:
            cond = self.cond
            if cond.shape[0] != xt.shape[0]:           # broadcast to batch
                cond = cond.expand(xt.shape[0], *cond.shape[1:])
            x0 = self.stream_model(xt, t, cond)
        ab = self._ab[t][:, None, None, None]
        sqrt_ab = ab.sqrt()
        sqrt_mab = (1.0 - ab).sqrt().clamp(min=1e-8)
        return (xt - sqrt_ab * x0) / sqrt_mab


class EpsFromV(torch.nn.Module):
    """Wrap a v-prediction stream-function model to expose eps-equivalent output.

        v̂  = stream_model(x_t, t, cond)
        ε̂  = √(1−ᾱ_t)·x_t + √ᾱ_t·v̂
    """

    def __init__(self, stream_model: torch.nn.Module, diffusion: "DDPM",
                 cond: torch.Tensor | None = None):
        super().__init__()
        self.stream_model = stream_model
        self._ab = diffusion.alpha_bar
        self.cond = cond

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.cond is None:
            v = self.stream_model(xt, t)
        else:
            cond = self.cond
            if cond.shape[0] != xt.shape[0]:
                cond = cond.expand(xt.shape[0], *cond.shape[1:])
            v = self.stream_model(xt, t, cond)
        ab = self._ab[t][:, None, None, None]
        sqrt_ab = ab.sqrt()
        sqrt_mab = (1.0 - ab).sqrt()
        return sqrt_mab * xt + sqrt_ab * v


# ---------------------------------------------------------------------------
# Parameterization helpers shared by all inference code
# ---------------------------------------------------------------------------

def is_v_pred(pred_type: str | None) -> bool:
    """True if a checkpoint's pred_type denotes v-prediction."""
    return str(pred_type).startswith("v")


def eps_wrapper_for(stream_model: torch.nn.Module, diffusion: "DDPM",
                    pred_type: str | None,
                    cond: torch.Tensor | None = None) -> torch.nn.Module:
    """Return the correct eps-equivalent wrapper for a checkpoint's pred_type."""
    if is_v_pred(pred_type):
        return EpsFromV(stream_model, diffusion, cond=cond)
    return EpsFromStreamFn(stream_model, diffusion, cond=cond)


def x0_from_output(diffusion: "DDPM", xt: torch.Tensor, model_out: torch.Tensor,
                   t: torch.Tensor, pred_type: str | None) -> torch.Tensor:
    """Convert a raw model output to x̂₀ given the parameterization."""
    if not is_v_pred(pred_type):
        return model_out
    ab = diffusion.alpha_bar[t][:, None, None, None]
    return ab.sqrt() * xt - (1.0 - ab).sqrt() * model_out
