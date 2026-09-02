"""High-level API: GP.predict(observations) -> (mean, uncertainty).

The classical baseline: Gaussian-process regression (Matern kriging) over the
grid, fitted independently for the u and v components. Packaged with the SAME
contract as the learned predictors so it can be compared like for like::

    from ddpm_library import GP

    mean, uncertainty = GP().predict(observations)

Why this model is here
----------------------
Kriging is the standard classical method for reconstructing a field from
scattered measurements, so it is the baseline a reader will ask about. More
importantly it is the reference point for any CALIBRATION claim: a GP returns a
posterior standard deviation by construction, with no conformal correction
fitted afterwards. "Calibrated" on its own is cheap -- widening intervals until
coverage reaches the nominal level always works -- so the meaningful comparison
is sharpness AT matched calibration, and that needs a calibrated competitor.

No weights
----------
Unlike every other predictor here, this one has no checkpoint. It fits its
hyperparameters (length scale, noise level) to the observations on every call by
maximising the log marginal likelihood, so there is nothing to download and
``device`` is accepted but ignored (scikit-learn is CPU-only).

Cost
----
Exact GP inference, no sparse approximation: the observation count (order 100) is
far below the prediction count (~3800), so the O(n^3) factorisation is trivial.
A few seconds per field (measured 7.0 s on one test box -- the per-call
hyperparameter refit dominates), and it scales cubically in the NUMBER OF
OBSERVATIONS -- a track of several thousand cells would need a sparse
approximation instead.

Vendored from the research pipeline (``GP Baseline/gp_infer.py``) with the kernel,
its bounds and the normalisation unchanged so results match the published run.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import numpy as np

from . import config as C
from .rasterize import observations_to_channels

# Kernel configuration, unchanged from the research implementation.
_LENGTH_SCALE = 0.15            # initial, as a fraction of the normalised domain
_LENGTH_SCALE_BOUNDS = (1e-3, 5.0)
_NOISE_LEVEL = 1e-4             # initial white-noise variance
_NOISE_LEVEL_BOUNDS = (1e-7, 1e-1)
_MATERN_NU = 2.5                # C^2-smooth: between RBF (oversmooth) and nu=0.5
_N_RESTARTS = 2


def _lib2model_2d(a: np.ndarray) -> np.ndarray:
    """(44, 94) lat x lon -> (94, 44) model grid."""
    return np.ascontiguousarray(a.T)


def _model2lib_field(a: np.ndarray) -> np.ndarray:
    """(2, 94, 44) -> (44, 94, 2)."""
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))


class GP:
    """Matern-kriging reconstruction with a native posterior standard deviation.

    Parameters
    ----------
    device : str
        Accepted for interface compatibility and ignored -- scikit-learn is
        CPU-only.
    length_scale, noise_level : float
        Initial hyperparameter values; both are then fitted per call by log
        marginal likelihood, so these only set the optimiser's starting point.
    n_restarts : int
        Random restarts for the hyperparameter optimiser.
    """

    #: This model takes no temporal priors.
    takes_priors = False

    def __init__(self, device: str = "auto", *,
                 length_scale: float = _LENGTH_SCALE,
                 noise_level: float = _NOISE_LEVEL,
                 n_restarts: int = _N_RESTARTS):
        try:
            import sklearn  # noqa: F401
        except ImportError as exc:                              # pragma: no cover
            raise ImportError(
                "GP needs scikit-learn: pip install scikit-learn") from exc

        self.device = "cpu"          # kept for API symmetry with the other models
        self.length_scale = float(length_scale)
        self.noise_level = float(noise_level)
        self.n_restarts = int(n_restarts)

        # This pipeline ships no grid asset; borrow the shared land mask, as the
        # RePaint wrapper does. The domain is identical across models.
        if not C.CORRDIFF_GRID_PATH.exists():
            raise FileNotFoundError(
                f"grid asset not found at {C.CORRDIFF_GRID_PATH}; "
                f"needed for the land mask.")
        self.land_np = np.asarray(
            np.load(C.CORRDIFF_GRID_PATH)["land_mask"]).astype(bool)   # (94, 44)
        self.ocean_np = ~self.land_np

    # ------------------------------------------------------------------

    @property
    def ocean_mask(self) -> np.ndarray:
        """(44, 94) ocean mask (1 = ocean), library orientation."""
        return self.ocean_np.T.astype(np.float32).copy()

    def predict(
        self,
        observations: Iterable[Sequence[float]],
        *,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the field by fitting a GP to the observations.

        Parameters
        ----------
        observations : iterable of (lat, lon, unix_t, u, v)
            The timestamp is ignored -- this model has no temporal component.
        seed : int or None
            Seeds the hyperparameter optimiser's restarts. The fit is otherwise
            deterministic, so results barely move with it.

        Returns
        -------
        mean, uncertainty : np.ndarray, each (44, 94, 2), float32, m/s.
            ``uncertainty`` is the GP POSTERIOR STANDARD DEVIATION -- a genuine
            predictive sigma, not an ensemble spread, and not rescaled by any
            fitted factor.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        obs_list = list(observations)
        if not obs_list:
            raise ValueError(
                "At least one observation is required; got an empty sequence.")

        # Front-end: the same rasterizer as every other predictor. Values are in
        # physical m/s; a GP needs no standardization.
        sparse_u, sparse_v, missing_mask = observations_to_channels(obs_list)
        u_grid = _lib2model_2d(sparse_u)
        v_grid = _lib2model_2d(sparse_v)
        observed = (_lib2model_2d(missing_mask) < 0.5) & self.ocean_np
        if not observed.any():
            raise ValueError(
                "no observation fell on an ocean cell of the model grid; "
                "check the observation coordinates.")

        H, W = self.land_np.shape                       # (94, 44)
        rows, cols = np.mgrid[0:H, 0:W]
        # Normalising each axis to [0, 1] matches the research implementation.
        # Note this makes the domain a unit square rather than preserving the
        # 94:44 aspect ratio, so an isotropic kernel is effectively anisotropic
        # in physical space -- kept as-is so numbers match the published run.
        row_n = rows / (H - 1)
        col_n = cols / (W - 1)

        X_obs = np.stack([row_n[observed], col_n[observed]], axis=1)
        y_u = u_grid[observed].astype(np.float64)
        y_v = v_grid[observed].astype(np.float64)
        X_pred = np.stack([row_n[self.ocean_np], col_n[self.ocean_np]], axis=1)

        kernel = (
            Matern(length_scale=self.length_scale,
                   length_scale_bounds=_LENGTH_SCALE_BOUNDS,
                   nu=_MATERN_NU)
            + WhiteKernel(noise_level=self.noise_level,
                          noise_level_bounds=_NOISE_LEVEL_BOUNDS)
        )

        mean_model = np.zeros((2, H, W), np.float32)
        std_model = np.zeros((2, H, W), np.float32)
        for ch, y in ((0, y_u), (1, y_v)):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                normalize_y=True,          # prior mean = sample mean, not zero
                random_state=seed,
            )
            gp.fit(X_obs, y)
            mu, sd = gp.predict(X_pred, return_std=True)
            mean_model[ch][self.ocean_np] = mu.astype(np.float32)
            std_model[ch][self.ocean_np] = sd.astype(np.float32)

        return _model2lib_field(mean_model), _model2lib_field(std_model)


# -- Module-level convenience function ---------------------------------------

_default_instance: Optional[GP] = None


def predict_gp(
    observations: Iterable[Sequence[float]],
    *,
    device: str = "auto",
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateless wrapper around :meth:`GP.predict` (lazy singleton)."""
    global _default_instance
    if _default_instance is None:
        _default_instance = GP()
    return _default_instance.predict(observations, seed=seed)
