"""High-level API: RePaint.predict(observations, priors) -> (mean, uncertainty).

A collaborator model (the "linear, time-conditioned" RePaint pipeline), packaged
with the SAME contract as the other predictors so it can be compared like for like::

    from ddpm_library import RePaint

    model = RePaint(device="auto")
    mean, uncertainty = model.predict(observations, priors=priors, n_draws=10)

How it differs from :class:`CorrDiff`
------------------------------------
This model is NOT trained on the observations. Its only per-pixel conditioning is
the ocean state ~13 h and ~25 h earlier; the sparse measurements are imposed at
SAMPLING time by a guidance term (DPS, or MCG which additionally replaces the
observed cells each step). CorrDiff instead feeds the observations to the network
as conditioning and generates a residual around a deterministic mean. Comparing
the two is comparing guided sampling against trained conditioning.

It also works in PHYSICAL m/s -- it is not z-scored -- so no normalization is
applied here. Getting that wrong silently produces plausible but incorrect fields.

Uncertainty
-----------
The sampler is stochastic, so ``n_draws > 1`` gives a real per-cell ensemble
spread. Unlike :class:`CorrDiff` there is no fitted calibration factor for this
model, so the returned ``uncertainty`` is the RAW ensemble standard deviation and
is expected to be under-dispersed. Do not read it as a calibrated 1-sigma; fit a
scale factor on held-out data first if you need one.

Cost
----
Guided sampling runs the full reverse chain WITH a gradient at every step, so it is
substantially slower than CorrDiff's DDIM sampler. Use ``stride`` to subsample the
chain when you need speed.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import config as C
from .inference import resolve_device
from .rasterize import observations_to_channels
from .repaint import DDPM, SAMPLERS, Repaint


def _lib2model_2d(a: np.ndarray) -> np.ndarray:
    """(44, 94) lat x lon -> (94, 44) model grid."""
    return np.ascontiguousarray(a.T)


def _lib2model_field(a: np.ndarray) -> np.ndarray:
    """(44, 94, 2) -> (2, 94, 44)."""
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))


def _model2lib_field(a: np.ndarray) -> np.ndarray:
    """(2, 94, 44) -> (44, 94, 2)."""
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))


class RePaint:
    """Linear-schedule RePaint UNet with temporal priors, sampled with DPS/MCG guidance.

    Parameters
    ----------
    device : str
        ``"auto"`` (CUDA > MPS > CPU) or an explicit torch device string.
    weights_path : str or Path, optional
        Override the bundled checkpoint.
    """

    def __init__(self, device: str = "auto", weights_path: Optional[str | Path] = None):
        self.device = resolve_device(device)

        wpath = Path(weights_path) if weights_path else C.REPAINT_WEIGHTS_PATH
        if not wpath.exists():
            raise FileNotFoundError(
                f"RePaint weights not found at {wpath} (git-lfs pull?).")
        ck = torch.load(wpath, map_location="cpu", weights_only=False)
        a = ck.get("args", {})

        # Both of the collaborator's checkpoints are supported. The time-conditioned
        # one stores cond_ch=4 / lags=(13, 25); the unconditional one stores an
        # explicit None for both, so `or` (not a .get default) is what handles it.
        self.cond_ch = int(ck.get("cond_ch") or 0)
        self.lags = tuple(int(x) for x in (ck.get("lags") or ()))
        if self.cond_ch != 2 * len(self.lags):
            raise RuntimeError(
                f"checkpoint cond_ch={self.cond_ch} is inconsistent with lags="
                f"{self.lags} (expected {2 * len(self.lags)} channels).")
        self.conditional = self.cond_ch > 0

        self.model = Repaint(
            in_ch=2, cond_ch=self.cond_ch,
            base_ch=a.get("base_ch", C.REPAINT_BASE_CH),
            time_dim=a.get("time_dim", C.REPAINT_TIME_DIM),
        ).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

        self.diffusion = DDPM(
            T=a.get("T", C.REPAINT_T),
            beta_schedule=ck.get("schedule", C.REPAINT_SCHEDULE),
            device=str(self.device),
            noise_std=float(ck.get("noise_std", C.REPAINT_NOISE_STD)),
        )

        # Land mask: this pipeline ships no grid asset, so borrow the shared one.
        # The domain is identical across models; only the normalization differs.
        grid_path = C.CORRDIFF_GRID_PATH
        if not grid_path.exists():
            raise FileNotFoundError(
                f"grid asset not found at {grid_path}; needed for the land mask.")
        self.land_np = np.asarray(np.load(grid_path)["land_mask"]).astype(bool)
        self.ocean_np = ~self.land_np

    # ------------------------------------------------------------------

    @property
    def ocean_mask(self) -> np.ndarray:
        """(44, 94) ocean mask (1 = ocean), library orientation."""
        return self.ocean_np.T.astype(np.float32).copy()

    def _build_priors(self, priors):
        """(cond_ch, 94, 44) temporal priors in PHYSICAL m/s (no standardization).

        Returns None for an unconditional checkpoint, which takes no priors at all.
        """
        if not self.conditional:
            if priors is not None:
                warnings.warn(
                    "this checkpoint is unconditional (cond_ch=0); the `priors` you "
                    "passed are ignored. Use the time-conditioned checkpoint if you "
                    "want the 13h/25h priors to be used.",
                    stacklevel=3,
                )
            return None
        n_lags = len(self.lags)
        if priors is None:
            warnings.warn(
                "RePaint.predict called without `priors`; this model is conditioned on "
                "the ocean state ~13h and ~25h earlier. Zeroing them degrades quality "
                "-- pass `priors` for best results.",
                stacklevel=3,
            )
            return np.zeros((self.cond_ch, C.REPAINT_H, C.REPAINT_W), dtype=np.float32)

        priors = list(priors)
        if len(priors) != n_lags:
            raise ValueError(
                f"priors must contain {n_lags} fields (one per lag {self.lags}); "
                f"got {len(priors)}.")
        chans = []
        for p in priors:
            p = np.asarray(p, dtype=np.float32)
            if p.shape == (C.OCEAN_H, C.OCEAN_W, 2):
                p_model = _lib2model_field(p)
            elif p.shape == (2, C.REPAINT_H, C.REPAINT_W):
                p_model = p
            else:
                raise ValueError(
                    f"each prior must be (44,94,2) or (2,94,44); got {p.shape}.")
            p_model = p_model.copy()
            p_model[:, self.land_np] = 0.0
            chans.append(p_model)
        return np.concatenate(chans, axis=0).astype(np.float32)

    def predict(
        self,
        observations: Iterable[Sequence[float]],
        priors=None,
        *,
        n_draws: int = C.REPAINT_DEFAULT_N_DRAWS,
        sampler: str = C.REPAINT_SAMPLER,
        step_size: float = C.REPAINT_STEP_SIZE,
        stride: int = C.REPAINT_STRIDE,
        seed=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the field from scattered observations and temporal priors.

        Parameters
        ----------
        observations : iterable of (lat, lon, unix_t, u, v)
        priors : sequence of two (44, 94, 2) fields in m/s (lags 13 h, 25 h).
        n_draws : int
            Ensemble members. >1 returns the raw per-cell spread as ``uncertainty``
            (NOT calibrated -- see the module docstring); 1 returns zeros.
        sampler : {"dps", "mcg"}
            Guidance scheme. "dps" marginally outperformed "mcg" in the published
            evaluation of this model.
        step_size : float
            Guidance strength (``z`` in the research scripts; published value 0.04).
        stride : int
            Step through the reverse chain; 1 uses all T steps (published setting).
        seed : int
            Base RNG seed; draw ``k`` uses ``seed + k``.

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
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {sorted(SAMPLERS)}; got {sampler!r}")

        # Front-end: same rasterizer as every other predictor. NOTE: values stay in
        # physical m/s -- this pipeline is not z-scored.
        sparse_u, sparse_v, missing_mask = observations_to_channels(obs_list)
        obs_field = _lib2model_field(np.stack([sparse_u, sparse_v], axis=-1))
        path_mask = (_lib2model_2d(missing_mask) < 0.5) & self.ocean_np
        if not path_mask.any():
            raise ValueError(
                "no observation fell on an ocean cell of the model grid; "
                "check the observation coordinates.")
        x0_known = obs_field.copy()
        x0_known[:, ~path_mask] = 0.0          # observed cells only, as the sampler expects

        cond_np = self._build_priors(priors)
        cond_t = (None if cond_np is None
                  else torch.from_numpy(cond_np).unsqueeze(0).to(self.device))
        x0_known_t = torch.from_numpy(x0_known)

        infer = SAMPLERS[sampler]
        draws = []
        for k in range(n_draws):
            torch.manual_seed(seed + k)        # the research scripts seed this way
            draws.append(infer(
                self.model, self.diffusion, x0_known_t, path_mask, self.land_np,
                cond=cond_t, device=str(self.device), stride=stride,
                step_size=step_size,
            ))
        arr = np.stack(draws, axis=0)          # (K, 2, 94, 44), m/s

        mean_model = arr.mean(axis=0)
        unc_model = arr.std(axis=0) if n_draws > 1 else np.zeros_like(mean_model)
        mean_model[:, self.land_np] = 0.0
        unc_model[:, self.land_np] = 0.0
        return (_model2lib_field(mean_model.astype(np.float32)),
                _model2lib_field(unc_model.astype(np.float32)))


# -- Module-level convenience function ---------------------------------------

_default_instance: Optional[RePaint] = None


def predict_repaint(
    observations: Iterable[Sequence[float]],
    priors=None,
    *,
    device: str = "auto",
    n_draws: int = C.REPAINT_DEFAULT_N_DRAWS,
    sampler: str = C.REPAINT_SAMPLER,
    step_size: float = C.REPAINT_STEP_SIZE,
    stride: int = C.REPAINT_STRIDE,
    seed= None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateless wrapper around :meth:`RePaint.predict` (lazy singleton)."""
    global _default_instance
    if _default_instance is None or str(_default_instance.device) != str(
        resolve_device(device)
    ):
        _default_instance = RePaint(device=device)
    return _default_instance.predict(
        observations, priors, n_draws=n_draws, sampler=sampler,
        step_size=step_size, stride=stride, seed=seed)
