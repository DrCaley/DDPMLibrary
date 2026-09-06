"""High-level API: DistAttn.predict(observations) -> (mean, uncertainty).

A collaborator model, packaged with the SAME contract as the other predictors so
it can be compared like for like::

    from ddpm_library import DistAttn

    model = DistAttn(device="auto")
    mean, uncertainty = model.predict(observations, n_draws=10)

How it differs from the others
------------------------------
The observations reach the network as a SET OF TOKENS that every resolution
level cross-attends to, rather than as input channels (:class:`CorrDiff`) or as
a guidance term during sampling (:class:`RePaint`). Each token is
``[x_norm, y_norm, u, v, age_norm]`` and the raw attention score is penalised by
two learned scalars::

    attn = (q @ k^T) / sqrt(d)  -  alpha * distance(query_xy, obs_xy)
                                -  beta  * age(obs)

Both start at zero, so training decided how much physical distance and
observation staleness actually matter.

Observation age
---------------
This is the only predictor here that uses the timestamp in the library's
``(lat, lon, unix_t, u, v)`` tuple. ``age_norm = (t_end - t_obs) / 3600`` hours,
where ``t_end`` is the newest observation supplied, so the freshest reading has
age 0 and a reading taken two hours earlier is marked as such. The others treat
a whole transect as if it were simultaneous; this one does not, which is the
realistic case for a vehicle collecting along a track.

Because age is measured relative to the newest observation, the ABSOLUTE epoch of
the timestamps does not matter -- only their spacing.

It also works in PHYSICAL m/s -- it is not z-scored -- so no normalization is
applied here. Getting that wrong silently produces plausible but wrong fields.

Uncertainty
-----------
The sampler is stochastic, so ``n_draws > 1`` gives a real per-cell ensemble
spread. The returned ``uncertainty`` is the RAW ensemble standard deviation and
is under-dispersed (raw coverage 0.737 at the 0.90 level on the realistic
benchmark). Multiply by :data:`~ddpm_library.config.DISTATTN_SIGMA_SCALE_TIMED`
(1.3592, split-conformal, held-out coverage 0.891) for calibrated intervals on
time-varying collection.

Cost
----
One draw is ``T / stride`` network calls (default 100). Draws are independent,
so cost scales linearly with ``n_draws``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import config as C
from .calibration import resolve_sigma_scale
from .distattn import DDPM, TimeCondUNet, ddpm_sample
from .geo import lat_lon_to_index
from .inference import draw_seed, resolve_device


def _model2lib_field(a: np.ndarray) -> np.ndarray:
    """(2, 94, 44) -> (44, 94, 2)."""
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))


class DistAttn:
    """Distance- and time-aware attention UNet, sampled with a strided DDPM chain.

    Parameters
    ----------
    device : str
        ``"auto"`` (CUDA > MPS > CPU) or an explicit torch device string.
    weights_path : str or Path, optional
        Override the bundled checkpoint.
    """

    #: This model takes no temporal priors; conditioning is entirely via tokens.
    takes_priors = False

    def __init__(self, device: str = "auto", weights_path: Optional[str | Path] = None):
        self.device = resolve_device(device)

        wpath = Path(weights_path) if weights_path else C.DISTATTN_WEIGHTS_PATH
        if not wpath.exists():
            raise FileNotFoundError(
                f"DistAttn weights not found at {wpath} (git-lfs pull?).")
        ck = torch.load(wpath, map_location="cpu", weights_only=False)
        a = ck.get("args", {})

        self.obs_dim = int(ck.get("obs_dim", C.DISTATTN_OBS_DIM))
        if self.obs_dim != C.DISTATTN_OBS_DIM:
            raise RuntimeError(
                f"checkpoint obs_dim={self.obs_dim}, but this wrapper builds "
                f"{C.DISTATTN_OBS_DIM}-dim tokens [x, y, u, v, age].")
        self.age_scale_sec = float(ck.get("age_scale_sec", C.DISTATTN_AGE_SCALE_SEC))
        self.max_age_sec = float(ck.get("dur_max_sec", C.DISTATTN_MAX_AGE_SEC))

        self.model = TimeCondUNet(
            in_ch=2,
            base_ch=a.get("base_ch", C.DISTATTN_BASE_CH),
            time_dim=a.get("time_dim", C.DISTATTN_TIME_DIM),
            obs_dim=self.obs_dim,
            n_heads=a.get("n_heads", C.DISTATTN_N_HEADS),
        ).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

        self.diffusion = DDPM(
            T=a.get("T", C.DISTATTN_T),
            device=str(self.device),
            noise_std=float(ck.get("noise_std", C.DISTATTN_NOISE_STD)),
        )

        # This pipeline ships its own mask, a strict subset of the shared grid.
        # Sampling zeroes land every step, so it must be the one used in training.
        if not C.DISTATTN_OCEAN_MASK_PATH.exists():
            raise FileNotFoundError(
                f"ocean mask not found at {C.DISTATTN_OCEAN_MASK_PATH}.")
        self.ocean_np = np.load(C.DISTATTN_OCEAN_MASK_PATH).astype(bool)  # (94, 44)
        if self.ocean_np.shape != (C.DISTATTN_H, C.DISTATTN_W):
            raise RuntimeError(
                f"ocean mask is {self.ocean_np.shape}, expected "
                f"{(C.DISTATTN_H, C.DISTATTN_W)}.")
        self.land_np = ~self.ocean_np

    # ------------------------------------------------------------------

    @property
    def ocean_mask(self) -> np.ndarray:
        """(44, 94) ocean mask (1 = ocean), library orientation."""
        return self.ocean_np.T.astype(np.float32).copy()

    def _build_tokens(self, obs_list) -> torch.Tensor:
        """(1, K, 5) observation tokens [x_norm, y_norm, u, v, age_hours].

        Coordinates follow the training convention exactly: the model grid is
        (H=94, W=44), ``x_norm = col / (W - 1)`` and ``y_norm = row / (H - 1)``,
        where a library ``(i_lat, j_lon)`` index maps to ``(row, col) =
        (j_lon, i_lat)``. Ages are relative to the NEWEST observation supplied.
        """
        times = np.array([float(o[2]) for o in obs_list], dtype=np.float64)
        t_end = times.max()

        rows, cols, us, vs, ages, dropped = [], [], [], [], [], 0
        for (lat, lon, t, u, v) in obs_list:
            i_lat, j_lon = lat_lon_to_index(float(lat), float(lon))
            row, col = j_lon, i_lat                      # library (44,94) -> model (94,44)
            if self.land_np[row, col]:
                dropped += 1
                continue
            rows.append(row)
            cols.append(col)
            us.append(float(u))
            vs.append(float(v))
            ages.append((t_end - float(t)) / self.age_scale_sec)

        if dropped:
            warnings.warn(
                f"{dropped} of {len(obs_list)} observations snapped onto land cells "
                f"of this model's mask and were dropped.",
                stacklevel=3,
            )
        if not rows:
            raise ValueError(
                "no observation fell on an ocean cell of the model grid; "
                "check the observation coordinates.")

        oldest = max(ages) * self.age_scale_sec
        if oldest > self.max_age_sec:
            warnings.warn(
                f"the oldest observation is {oldest / 3600:.1f} h old, beyond the "
                f"{self.max_age_sec / 3600:.1f} h span this model was trained on; "
                f"the age conditioning is extrapolating.",
                stacklevel=3,
            )

        tokens = np.stack([
            np.asarray(cols, np.float32) / max(C.DISTATTN_W - 1, 1),   # x_norm
            np.asarray(rows, np.float32) / max(C.DISTATTN_H - 1, 1),   # y_norm
            np.asarray(us, np.float32),
            np.asarray(vs, np.float32),
            np.asarray(ages, np.float32),                              # hours
        ], axis=1)
        return torch.from_numpy(tokens).unsqueeze(0).to(self.device)

    def predict(
        self,
        observations: Iterable[Sequence[float]],
        *,
        n_draws: int = C.DISTATTN_DEFAULT_N_DRAWS,
        stride: int = C.DISTATTN_STRIDE,
        seed=None,
        priors=None, # Dummy argument
        calibrate: bool = True,
        sigma_scale: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the field from scattered, time-stamped observations.

        Parameters
        ----------
        observations : iterable of (lat, lon, unix_t, u, v)
            ``unix_t`` is used here: each observation is tokenised with its age
            relative to the newest one supplied. Only the spacing matters, not
            the absolute epoch.
        n_draws : int
            Ensemble members. >1 returns the raw per-cell spread as ``uncertainty``
            (NOT calibrated -- see the module docstring); 1 returns zeros.
        stride : int
            Reverse-chain step size; one draw costs ``T / stride`` network calls.
        seed : int or None
            Base RNG seed; draw ``k`` uses ``(seed + 1) * 100003 + k``, the
            same spread ``stream`` uses. ``None`` (the default)
            leaves the global RNG untouched, so draws are not reproducible.

        Returns
        -------
        mean, uncertainty : np.ndarray, each (44, 94, 2), float32, m/s.
        """
        obs_list = list(observations)
        if not obs_list:
            raise ValueError(
                "At least one observation is required; got an empty sequence.")
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1; got {n_draws}.")
        if stride < 1 or stride > self.diffusion.T:
            raise ValueError(
                f"stride must be in [1, {self.diffusion.T}]; got {stride}.")

        tokens = self._build_tokens(obs_list)
        obs_mask = torch.ones(1, tokens.shape[1], dtype=torch.bool, device=self.device)

        def eps_fn(xt, t):
            return self.model(xt, t, tokens, obs_mask)

        draws = []
        for k in range(n_draws):
            if seed is not None:               # seed=None -> non-reproducible draws
                torch.manual_seed(draw_seed(seed, k))
            draws.append(ddpm_sample(eps_fn, self.diffusion, self.land_np,
                                     stride=stride, device=str(self.device)).numpy())
        arr = np.stack(draws, axis=0)          # (K, 2, 94, 44), m/s

        mean_model = arr.mean(axis=0)
        unc_model = arr.std(axis=0) if n_draws > 1 else np.zeros_like(mean_model)
        if calibrate and n_draws > 1:
            scale, why = resolve_sigma_scale(
                obs_list, timed=C.DISTATTN_SIGMA_SCALE_TIMED, override=sigma_scale,
                n_draws=n_draws, fitted_n_draws=C.DISTATTN_FITTED_N_DRAWS,
                model="DistAttn")
            unc_model = unc_model * scale
            self._last_sigma_scale = (scale, why)
        else:
            self._last_sigma_scale = (1.0, "uncalibrated (calibrate=False)"
                                      if not calibrate else "single draw")
        mean_model[:, self.land_np] = 0.0
        unc_model[:, self.land_np] = 0.0
        return (_model2lib_field(mean_model.astype(np.float32)),
                _model2lib_field(unc_model.astype(np.float32)))


# -- Module-level convenience function ---------------------------------------

_default_instance: Optional[DistAttn] = None


def predict_distattn(
    observations: Iterable[Sequence[float]],
    *,
    device: str = "auto",
    n_draws: int = C.DISTATTN_DEFAULT_N_DRAWS,
    stride: int = C.DISTATTN_STRIDE,
    seed=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateless wrapper around :meth:`DistAttn.predict` (lazy singleton)."""
    global _default_instance
    if _default_instance is None or str(_default_instance.device) != str(
        resolve_device(device)
    ):
        _default_instance = DistAttn(device=device)
    return _default_instance.predict(
        observations, n_draws=n_draws, stride=stride, seed=seed)
