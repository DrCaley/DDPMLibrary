"""Linear-schedule DDPM constants for the distance/time-aware attention model.

Inference subset of the collaborator's ``ddpm.py``: the sampler only reads ``T``,
``alpha_bar`` and ``noise_std``, so the training-time helpers (``q_sample``,
``training_loss``, the curl/div structural term) are deliberately omitted.

The schedule is reproduced exactly -- ``betas = linspace(1e-4, 0.02, T)`` and
``alpha_bar = cumprod(1 - betas)``. ``tests/test_distattn.py`` asserts this
matches the original implementation elementwise; changing it silently would
produce plausible but wrong fields.
"""

from __future__ import annotations

import torch


class DDPM:
    """Noise schedule for the distance/time-aware attention model (inference only).

    Parameters
    ----------
    T : int
        Number of diffusion steps the model was trained with.
    device : str or torch.device
        Where ``alpha_bar`` lives; must match the tensors passed to the sampler.
    noise_std : float
        Scale of the latent noise. This pipeline works in PHYSICAL m/s rather than
        z-scored units, so this is the dataset's ocean-cell standard deviation
        (~0.116), NOT 1.0.
    """

    def __init__(self, T: int = 1000, device: str | torch.device = "cpu",
                 noise_std: float = 1.0):
        self.T = T
        self.device = device
        self.noise_std = noise_std

        self.betas = torch.linspace(1e-4, 0.02, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
