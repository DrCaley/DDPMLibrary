"""High-level API: CorrDiff.predict(observations, priors) -> (mean, uncertainty).

The research group's best model, packaged with the SAME contract as
:class:`DDPM`, :class:`VCNN` and :class:`StreamDDPM`::

    from ddpm_library import CorrDiff

    model = CorrDiff(device="auto")
    mean, uncertainty = model.predict(observations, priors=priors)
    # mean.shape        == (44, 94, 2)   # (lat, lon, [u, v]) in m/s
    # uncertainty.shape == (44, 94, 2)   # per-cell 1-sigma, m/s

What makes this model different
-------------------------------
It is the first predictor in this library whose ``uncertainty`` output is a
*calibrated* estimate rather than zeros. A deterministic V-CNN supplies the mean
field and a conditional diffusion model generates the RESIDUAL (truth - mean)
[Mardani et al. 2025]; drawing many residuals yields an ensemble whose spread is
a genuine predictive uncertainty. Because uncertainty is the point of this model,
``n_draws`` defaults to a full ensemble (see :data:`~ddpm_library.config.
CORRDIFF_DEFAULT_N_DRAWS`) -- set ``n_draws=1`` for a fast single field.

Sensor-noise dial
-----------------
The model is conditioned on the observation-error level it should assume, so one
set of weights serves any instrument::

    mean, unc = model.predict(obs, priors, sensor_noise=0.05)   # 5%-error sensors

``sensor_noise`` is a fraction of the field standard deviation, in
``[0, CORRDIFF_NOISE_MAX]``. Higher values widen the predictive distribution and
make the reconstruction trust the observations less. Default 0.0 (ideal sensors).

Temporal priors
---------------
Like :class:`StreamDDPM`, this model is conditioned on the ocean state ~13 h and
~25 h earlier. Supply them via ``priors`` as two (44, 94, 2) fields in m/s. If
omitted the priors are zeroed (degraded quality) and a warning is emitted.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import config as C
from .corrdiff import (
    UNet, VDiffusion, assemble_cond, ddim_sample_residual, dist_channel,
    geometry_channels, observation_channels, sigma_channel,
)
from .inference import resolve_device
from .rasterize import observations_to_channels


# --- grid orientation helpers (library lat x lon 44x94  <->  model 94x44) ----

def _lib2model_2d(a: np.ndarray) -> np.ndarray:
    """(44, 94) lat x lon  ->  (94, 44) model grid."""
    return np.ascontiguousarray(a.T)


def _lib2model_field(a: np.ndarray) -> np.ndarray:
    """(44, 94, 2) lat x lon x uv  ->  (2, 94, 44) model field."""
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))


def _model2lib_field(a: np.ndarray) -> np.ndarray:
    """(2, 94, 44) model field  ->  (44, 94, 2) lat x lon x uv."""
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))


class CorrDiff:
    """V-CNN mean + residual diffusion, with a sensor-noise dial.

    Parameters
    ----------
    device : str
        ``"auto"`` (CUDA > MPS > CPU) or an explicit torch device string.
    weights_path : str or Path, optional
        Override the bundled diffusion checkpoint.
    """

    def __init__(
        self,
        device: str = "auto",
        weights_path: Optional[str | Path] = None,
    ):
        self.device = resolve_device(device)

        # --- static grid + normalisation stats ---
        grid_path = C.CORRDIFF_GRID_PATH
        if not grid_path.exists():
            raise FileNotFoundError(f"CorrDiff grid asset not found at {grid_path}.")
        g = np.load(grid_path)
        self.land_np = np.asarray(g["land_mask"]).astype(bool)      # (94, 44) True=land
        self.ocean_np = ~self.land_np
        self.data_mean = float(g["data_mean"])
        self.data_std = float(g["data_std"])
        self.lags = tuple(int(x) for x in g["lags"])
        self.geom = geometry_channels(self.land_np)                  # (3, 94, 44)
        self._ocean_t = torch.from_numpy(self.ocean_np)

        # --- diffusion model ---
        wpath = Path(weights_path) if weights_path else C.CORRDIFF_WEIGHTS_PATH
        if not wpath.exists():
            raise FileNotFoundError(
                f"CorrDiff weights not found at {wpath} (git-lfs pull?).")
        ck = torch.load(wpath, map_location="cpu", weights_only=False)
        if ck.get("mean_type", "vcnn") != "vcnn":
            raise ValueError(
                f"{wpath} is not a V-CNN-anchored CorrDiff checkpoint "
                f"(mean_type={ck.get('mean_type')!r}).")

        self.cond_ch = int(ck["cond_ch"])
        self.total_cond = int(ck["total_cond"])
        self.use_dist = bool(ck.get("use_dist", False))
        self.noise_cond = bool(ck.get("noise_cond", False))
        self.noise_max = float(ck.get("noise_max", 0.0))

        # The assembled input must be exactly what the network was trained on:
        #   cond(cond_ch) + mean(2) + dist(1 if use_dist) + sigma(1 if noise_cond)
        expected = self.cond_ch + 2 + int(self.use_dist) + int(self.noise_cond)
        if expected != self.total_cond:
            raise RuntimeError(
                f"checkpoint channel bookkeeping inconsistent: cond_ch={self.cond_ch} "
                f"+ mean(2) + dist({int(self.use_dist)}) + sigma({int(self.noise_cond)}) "
                f"= {expected}, but total_cond={self.total_cond}.")

        a = ck.get("args", {})
        self.model = UNet(
            in_ch=2 + self.total_cond,
            base_ch=a.get("base_ch", C.CORRDIFF_BASE_CH),
            time_dim=a.get("time_dim", C.CORRDIFF_TIME_DIM),
            out_ch=2,
        ).to(self.device)
        self.model.load_state_dict(ck.get("ema", ck["model"]))       # EMA weights preferred
        self.model.eval()
        self.diffusion = VDiffusion(T=a.get("T", C.CORRDIFF_T), device=self.device)

        self._vcnn = None      # lazy: the deterministic anchor (bundled V-CNN)

    # ------------------------------------------------------------------

    @property
    def ocean_mask(self) -> np.ndarray:
        """A copy of the (44, 94) ocean mask (1 = ocean), library orientation.

        Same API as :attr:`VCNN.ocean_mask`, but NOT bit-identical to it: this
        mask is the union of the u- and v-channel land masks, so it marks ~9 more
        cells (0.2% of the grid) as land than the V-CNN's does. That is the mask
        the diffusion model was trained with, and the anchor is re-masked to match,
        so predictions are consistent internally. When comparing predictors
        head-to-head, score them all on a COMMON mask -- the intersection, i.e.
        this one -- so no method is credited for cells another treats as land.
        """
        return self.ocean_np.T.astype(np.float32).copy()

    def _standardize(self, field_phys: np.ndarray) -> np.ndarray:
        return (field_phys - self.data_mean) / max(self.data_std, 1e-8)

    def _anchor(self, observations) -> np.ndarray:
        """Deterministic mean field from the bundled V-CNN.

        Returns (2, 94, 44) in STANDARDISED units -- the diffusion model was
        trained with the anchor on the same z-scale as its other inputs.
        """
        if self._vcnn is None:
            from .vcnn_predict import VCNN
            self._vcnn = VCNN(device=str(self.device))
        v_lib, _ = self._vcnn.predict(observations)          # (44, 94, 2) m/s
        v_model = _lib2model_field(np.asarray(v_lib, dtype=np.float32))
        v_std = self._standardize(v_model).astype(np.float32)
        v_std[:, self.land_np] = 0.0
        return v_std

    def _build_priors(self, priors) -> np.ndarray:
        """Standardised priors (2 * n_lags, 94, 44), model orientation."""
        n_lags = len(self.lags)
        if priors is None:
            warnings.warn(
                "CorrDiff.predict called without `priors`; the model is conditioned "
                "on the ocean state ~13h and ~25h earlier. Zeroing the priors "
                "degrades quality -- pass `priors` for best results.",
                stacklevel=3,
            )
            return np.zeros((2 * n_lags, C.CORRDIFF_H, C.CORRDIFF_W), dtype=np.float32)

        priors = list(priors)
        if len(priors) != n_lags:
            raise ValueError(
                f"priors must contain {n_lags} fields (one per lag {self.lags}); "
                f"got {len(priors)}.")
        chans = []
        for p in priors:
            p = np.asarray(p, dtype=np.float32)
            if p.shape == (C.OCEAN_H, C.OCEAN_W, 2):            # library (44, 94, 2)
                p_model = _lib2model_field(p)
            elif p.shape == (2, C.CORRDIFF_H, C.CORRDIFF_W):    # already model-oriented
                p_model = p
            else:
                raise ValueError(
                    f"each prior must be (44,94,2) or (2,94,44); got {p.shape}.")
            p_std = self._standardize(p_model).astype(np.float32)
            p_std[:, self.land_np] = 0.0
            chans.append(p_std)
        return np.concatenate(chans, axis=0).astype(np.float32)

    @torch.no_grad()
    def predict(
        self,
        observations: Iterable[Sequence[float]],
        priors=None,
        *,
        sensor_noise: float = 0.0,
        n_draws: int = C.CORRDIFF_DEFAULT_N_DRAWS,
        steps: int = C.CORRDIFF_STEPS,
        seed: int = 0,
        calibrate: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the full velocity field and its uncertainty.

        Parameters
        ----------
        observations : iterable of (lat, lon, unix_t, u, v)
            Same convention as :meth:`DDPM.predict`.
        priors : sequence of prior fields, optional
            One field per lag (13 h, 25 h), each (44, 94, 2) in m/s.
        sensor_noise : float, default 0.0
            Assumed observation-error level, as a fraction of the field standard
            deviation, in ``[0, CORRDIFF_NOISE_MAX]``. Widens the predictive
            distribution and reduces trust in the observations.
        n_draws : int
            Ensemble members. >1 gives the calibrated per-cell spread;
            1 gives a single field with zero uncertainty (fast path).
        steps : int
            DDIM sampling steps. The default is the evaluated setting; fewer
            steps trade distribution quality for speed (the mean field is far
            less sensitive to this than the spread is).
        seed : int
            Base RNG seed; results are reproducible for a given seed.
        calibrate : bool, default True
            Scale the raw ensemble spread by the pre-fitted conformal factor
            :data:`~ddpm_library.config.CORRDIFF_SIGMA_SCALE` so the reported
            uncertainty is a calibrated 1-sigma. The raw diffusion ensemble is
            under-dispersed (a known property of conditional diffusion models);
            this factor was fitted on held-out data. Set False for the raw spread.

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
        if self.noise_cond:
            if not 0.0 <= sensor_noise <= self.noise_max:
                raise ValueError(
                    f"sensor_noise must be in [0, {self.noise_max}] (fraction of the "
                    f"field standard deviation); got {sensor_noise}. Values outside "
                    f"the trained range are not supported.")
        elif sensor_noise:
            raise ValueError(
                "this checkpoint has no sensor-noise conditioning; "
                "sensor_noise must be 0.")

        # --- front-end: same rasterizer as the other predictors ---
        sparse_u, sparse_v, missing_mask = observations_to_channels(obs_list)
        obs_field_lib = np.stack([sparse_u, sparse_v], axis=-1)        # (44, 94, 2)
        obs_field_model = _lib2model_field(obs_field_lib)              # (2, 94, 44)
        obs_field_std = self._standardize(obs_field_model).astype(np.float32)
        obs_field_std[:, self.land_np] = 0.0
        path_mask = (_lib2model_2d(missing_mask) < 0.5) & self.ocean_np

        if not path_mask.any():
            raise ValueError(
                "no observation fell on an ocean cell of the model grid; "
                "check the observation coordinates.")

        # --- conditioning stack, in the trained channel order ---
        obs_ch = observation_channels(obs_field_std, path_mask, self.land_np)
        priors_std = self._build_priors(priors)
        cond = assemble_cond(obs_ch, priors_std, self.geom)
        if cond.shape[0] != self.cond_ch:
            raise RuntimeError(
                f"assembled cond has {cond.shape[0]} channels but the model expects "
                f"{self.cond_ch}; check the priors/geometry configuration.")

        anchor = self._anchor(obs_list)                                # (2, 94, 44) std units
        parts = [cond, anchor]
        if self.use_dist:
            parts.append(dist_channel(path_mask))
        if self.noise_cond:
            parts.append(sigma_channel(sensor_noise, self.noise_max, path_mask.shape))
        cond_aug = np.concatenate(parts, axis=0).astype(np.float32)
        if cond_aug.shape[0] != self.total_cond:
            raise RuntimeError(
                f"assembled input has {cond_aug.shape[0]} conditioning channels but "
                f"the model expects {self.total_cond}.")

        # --- sample residuals, add to the anchor ---
        cond_t = torch.from_numpy(cond_aug).unsqueeze(0)
        res = ddim_sample_residual(
            self.model, cond_t, self._ocean_t, self.diffusion, self.device,
            n_draws=n_draws, steps=steps, seed=seed,
        ).cpu().numpy()                                                # (K, 2, 94, 44)
        draws = (anchor[None] + res) * self.ocean_np[None, None]       # standardised

        mean_model = draws.mean(axis=0) * self.data_std + self.data_mean
        if n_draws > 1:
            unc_model = draws.std(axis=0) * self.data_std              # scale-only: no offset
            if calibrate:
                unc_model = unc_model * C.CORRDIFF_SIGMA_SCALE
        else:
            unc_model = np.zeros_like(mean_model)
        mean_model[:, self.land_np] = 0.0
        unc_model[:, self.land_np] = 0.0

        mean = _model2lib_field(mean_model.astype(np.float32))
        uncertainty = _model2lib_field(unc_model.astype(np.float32))
        return mean, uncertainty


# -- Module-level convenience function ---------------------------------------

_default_instance: Optional[CorrDiff] = None


def predict_corrdiff(
    observations: Iterable[Sequence[float]],
    priors=None,
    *,
    device: str = "auto",
    sensor_noise: float = 0.0,
    n_draws: int = C.CORRDIFF_DEFAULT_N_DRAWS,
    steps: int = C.CORRDIFF_STEPS,
    seed: int = 0,
    calibrate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateless wrapper around :meth:`CorrDiff.predict` (lazy singleton)."""
    global _default_instance
    if _default_instance is None or str(_default_instance.device) != str(
        resolve_device(device)
    ):
        _default_instance = CorrDiff(device=device)
    return _default_instance.predict(
        observations, priors, sensor_noise=sensor_noise, n_draws=n_draws,
        steps=steps, seed=seed, calibrate=calibrate)
