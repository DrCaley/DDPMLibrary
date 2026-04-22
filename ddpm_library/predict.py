"""High-level API: DDPM.predict(observations) -> (mean, uncertainty).

Example
-------
    from ddpm_library import DDPM

    obs = [
        # (lat, lon, unix_t_seconds, u_m_per_s, v_m_per_s)
        (18.305, -64.710, 1_700_000_000.0,  0.12, -0.03),
        (18.320, -64.705, 1_700_000_060.0, -0.05,  0.08),
        # ... ~20-60 observations is typical
    ]

    model = DDPM(device="auto")
    mean, uncertainty = model.predict(obs)
    # mean.shape == (44, 94, 2)   # (lat, lon, [u, v]) in m/s
    # uncertainty.shape == (44, 94, 2)   # currently all zeros; see README
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import (
    DEFAULT_RESAMPLE_STEPS, OCEAN_H, OCEAN_W,
)
from .inference import (
    inpaint, load_network, make_schedule, resolve_device,
)
from .rasterize import observations_to_channels


class DDPM:
    """Reusable split-head DDPM predictor.

    Holds the loaded UNet + diffusion schedule so repeated calls don't
    reload weights. Thread-safety: create one instance per thread.
    """

    def __init__(
        self,
        device: str = "auto",
        weights_path: Optional[str | Path] = None,
    ):
        self.device = resolve_device(device)
        self.net = load_network(weights_path=weights_path, device=self.device)
        self.schedule = make_schedule(self.device)

    def predict(
        self,
        observations: Iterable[Sequence[float]],
        *,
        single_step: bool = True,
        t_start: Optional[int] = None,
        resample_steps: int = DEFAULT_RESAMPLE_STEPS,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the full velocity field from scattered observations.

        Parameters
        ----------
        observations : iterable of (lat, lon, unix_t, u, v)
            lat, lon : decimal degrees (must lie within the model's covered
                region — see config.LAT_MIN/MAX, LON_MIN/MAX).
            unix_t : seconds since the Unix epoch. Accepted but ignored —
                the model is not time-conditioned.
            u, v : velocity components in m/s.
        single_step : bool, default True
            If True, run a single UNet forward pass at step `t_start` and
            return the direct x0 prediction (fast: one network call). If
            False, run the full iterative RePaint reverse chain (slower
            but typically higher quality).
        t_start : int, optional
            In single-step mode, the timestep the UNet is queried at.
            In iterative mode, the reverse chain starting step.
            If None (default), uses 50 for single-step and 75 for
            iterative — the settings from scripts/eval_helmholtz_split.py.
        resample_steps : int
            RePaint repaint iterations per timestep (iterative mode only).
        seed : int, optional
            Seed for the diffusion noise. None → non-deterministic.

        Returns
        -------
        mean : np.ndarray, shape (44, 94, 2), dtype float32
            Predicted velocity field on the model grid. Axis order is
            (lat, lon, [u, v]). Use ddpm_library.geo.grid_arrays() to get
            the corresponding lat/lon coordinates.
        uncertainty : np.ndarray, shape (44, 94, 2), dtype float32
            Currently an array of zeros. The underlying model produces a
            single x0 prediction per call; a proper uncertainty estimate
            would require averaging multiple samples. See README limitations.

        Raises
        ------
        ValueError
            If any observation's (lat, lon) is outside the covered region.
        """
        obs_list = list(observations)
        if not obs_list:
            raise ValueError(
                "At least one observation is required; got an empty sequence."
            )

        sparse_u, sparse_v, missing_mask = observations_to_channels(obs_list)

        mean = inpaint(
            sparse_u, sparse_v, missing_mask,
            net=self.net, schedule=self.schedule, device=self.device,
            single_step=single_step,
            t_start=t_start, resample_steps=resample_steps, seed=seed,
        )
        uncertainty = np.zeros_like(mean)
        return mean, uncertainty


# ── Module-level convenience function ─────────────────────────────────

_default_instance: Optional[DDPM] = None


def predict(
    observations: Iterable[Sequence[float]],
    *,
    device: str = "auto",
    single_step: bool = True,
    t_start: Optional[int] = None,
    resample_steps: int = DEFAULT_RESAMPLE_STEPS,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateless wrapper around DDPM.predict. Lazy-loads the model on first call."""
    global _default_instance
    if _default_instance is None or str(_default_instance.device) != str(
        resolve_device(device)
    ):
        _default_instance = DDPM(device=device)
    return _default_instance.predict(
        observations,
        single_step=single_step,
        t_start=t_start, resample_steps=resample_steps, seed=seed,
    )
