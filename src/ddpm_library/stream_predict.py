"""High-level API: StreamDDPM.predict(observations, priors) -> (mean, uncertainty).

The best research pipeline, packaged with the SAME contract as :class:`DDPM`
and :class:`VCNN`:

    from ddpm_library import StreamDDPM

    model = StreamDDPM(device="auto")
    mean, uncertainty = model.predict(observations, priors=priors)
    # mean.shape        == (44, 94, 2)   # (lat, lon, [u, v]) in m/s
    # uncertainty.shape == (44, 94, 2)

Under the hood: a conditional stream-function DDPM predicts the flow DIRECTION
(divergence-free by construction), and a heteroscedastic magnitude UNet supplies
per-cell speed; they are fused (coupled) and Helmholtz-reprojected. With
``n_draws > 1`` the per-cell ensemble spread is returned as ``uncertainty``;
with ``n_draws == 1`` (default) ``uncertainty`` is zeros, exactly like the
other two predictors.

Temporal priors
---------------
This model is conditioned on the ocean state ~13 h and ~25 h earlier. Supply
them via ``priors`` as two fields shaped like the output ((44, 94, 2), m/s).
If omitted, the priors are zeroed (degraded quality) and a warning is emitted.
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
from .stream import (
    DDPM, StreamFunctionUNet, ensemble_infer, dpmpp_ensemble,
    geometry_channels, build_conditioning, load_hetero_magnitude_model,
    fuse_coupled, helmholtz_project,
)


# --- grid orientation helpers (library lat×lon 44×94  <->  model 94×44) ------

def _lib2model_2d(a: np.ndarray) -> np.ndarray:
    """(44, 94) lat×lon  ->  (94, 44) model grid."""
    return np.ascontiguousarray(a.T)


def _lib2model_field(a: np.ndarray) -> np.ndarray:
    """(44, 94, 2) lat×lon×uv  ->  (2, 94, 44) model field."""
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))


def _model2lib_field(a: np.ndarray) -> np.ndarray:
    """(2, 94, 44) model field  ->  (44, 94, 2) lat×lon×uv."""
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))


def _smooth_ocean(field: np.ndarray, ocean: np.ndarray, sigma: float) -> np.ndarray:
    """Nan-aware Gaussian smooth of a (2, H, W) field over ocean cells only.

    Normalizing by the smoothed ocean mask keeps land/coastlines from bleeding
    in. Removes the grid-scale (checkerboard) mode from the ensemble spread.
    """
    from scipy.ndimage import gaussian_filter
    om = ocean.astype(np.float64)
    denom = np.clip(gaussian_filter(om, sigma=sigma), 1e-6, None)
    out = np.empty_like(field)
    for c in range(field.shape[0]):
        out[c] = gaussian_filter(np.nan_to_num(field[c]) * om, sigma=sigma) / denom
    out[:, ~ocean] = 0.0
    return out.astype(np.float32)


class StreamDDPM:
    """Conditional stream-function DDPM + heteroscedastic magnitude predictor.

    Parameters
    ----------
    device : str
        ``"auto"`` (CUDA > MPS > CPU) or an explicit torch device string.
    dir_weights_path, mag_weights_path : str or Path, optional
        Override the bundled diffusion / magnitude checkpoints.
    """

    def __init__(
        self,
        device: str = "auto",
        dir_weights_path: Optional[str | Path] = None,
        mag_weights_path: Optional[str | Path] = None,
    ):
        self.device = resolve_device(device)

        # --- static grid + normalization stats ---
        grid_path = C.STREAM_GRID_PATH
        if not grid_path.exists():
            raise FileNotFoundError(
                f"Stream grid asset not found at {grid_path}.")
        g = np.load(grid_path)
        self.land_np = np.asarray(g["land_mask"]).astype(bool)   # (94,44) True=land
        self.ocean_np = ~self.land_np
        self.data_mean = float(g["data_mean"])
        self.data_std = float(g["data_std"])
        self.lags = tuple(int(x) for x in g["lags"])
        # geometry is static — precompute once (3, 94, 44)
        self.geom = geometry_channels(self.land_np)

        # --- diffusion direction model ---
        dir_path = Path(dir_weights_path) if dir_weights_path else C.STREAM_DIR_WEIGHTS_PATH
        if not dir_path.exists():
            raise FileNotFoundError(
                f"Diffusion weights not found at {dir_path} (git-lfs pull?).")
        dckpt = torch.load(dir_path, map_location="cpu", weights_only=False)
        self.pred_type = dckpt.get("pred_type", C.STREAM_PRED_TYPE)
        self.cond_ch = int(dckpt.get("cond_ch", C.STREAM_COND_CH))
        ca = dckpt.get("args", {})
        self.model = StreamFunctionUNet(
            in_ch=2, base_ch=ca.get("base_ch", C.STREAM_BASE_CH),
            time_dim=ca.get("time_dim", C.STREAM_TIME_DIM),
            cond_ch=self.cond_ch,
        ).to(self.device)
        self.model.load_state_dict(dckpt["model"])
        self.model.eval()
        self.diffusion = DDPM(
            T=ca.get("T", C.STREAM_T),
            beta_schedule=ca.get("schedule", C.STREAM_SCHEDULE),
            device=self.device,
            noise_type=ca.get("noise_type", C.STREAM_NOISE_TYPE),
            spectral_filter=dckpt.get("spectral_filter", None),
        )
        # legacy obs layout (3 obs channels) when cond_ch <= 10
        self._legacy_obs = self.cond_ch <= 10

        # --- heteroscedastic magnitude model ---
        mag_path = Path(mag_weights_path) if mag_weights_path else C.STREAM_MAG_WEIGHTS_PATH
        if not mag_path.exists():
            raise FileNotFoundError(
                f"Magnitude weights not found at {mag_path} (git-lfs pull?).")
        (self._het_net, self._hsm, self._hss,
         self._het_clip) = load_hetero_magnitude_model(mag_path, self.device)

        self._vcnn = None   # lazy — only built if full_field=True is used

    # ------------------------------------------------------------------

    def _divergent_from_vcnn(self, observations) -> np.ndarray:
        """Curl-free (divergent) component of a VCNN prediction, model grid.

        Returns (2, 94, 44) m/s: VCNN field minus its divergence-free part.
        """
        if self._vcnn is None:
            from .vcnn_predict import VCNN
            self._vcnn = VCNN(device=str(self.device))
        v_lib, _ = self._vcnn.predict(observations)          # (44,94,2) m/s
        v_model = _lib2model_field(v_lib)                    # (2,94,44)
        v_divfree = helmholtz_project(v_model, self.ocean_np, max_iters=30, tol=1e-7)
        v_div = (v_model - v_divfree).astype(np.float32)
        v_div[:, self.land_np] = 0.0
        return v_div

    @property
    def ocean_mask(self) -> np.ndarray:
        """A copy of the (44, 94) ocean mask (1 = ocean), library orientation.

        Matches :attr:`VCNN.ocean_mask` so the three predictors expose the same
        mask API.
        """
        return (~self.land_np).T.astype(np.float32).copy()

    def _standardize(self, field_phys: np.ndarray) -> np.ndarray:
        return (field_phys - self.data_mean) / max(self.data_std, 1e-8)

    def _build_priors(self, priors, project=True) -> np.ndarray:
        """Return standardized priors (2*n_lags, 94, 44) in model orientation.

        If ``project`` (default), each prior field is Helmholtz-projected to be
        divergence-free — matching how the training data was built, so real
        (not-quite-div-free) priors from the field are made in-distribution.
        """
        n_lags = len(self.lags)
        if priors is None:
            warnings.warn(
                "StreamDDPM.predict called without `priors`; the model is "
                "conditioned on the ocean state ~13h and ~25h earlier. Zeroing "
                "the priors degrades quality — pass `priors` for best results.",
                stacklevel=3,
            )
            return np.zeros((2 * n_lags, C.STREAM_H, C.STREAM_W), dtype=np.float32)

        priors = list(priors)
        if len(priors) != n_lags:
            raise ValueError(
                f"priors must contain {n_lags} fields (one per lag {self.lags}); "
                f"got {len(priors)}.")
        chans = []
        for p in priors:
            p = np.asarray(p, dtype=np.float32)
            if p.shape == (C.OCEAN_H, C.OCEAN_W, 2):          # library (44,94,2)
                p_model = _lib2model_field(p)                 # -> (2,94,44)
            elif p.shape == (2, C.STREAM_H, C.STREAM_W):      # already model-oriented
                p_model = p
            else:
                raise ValueError(
                    f"each prior must be (44,94,2) or (2,94,44); got {p.shape}.")
            p_std = self._standardize(p_model).astype(np.float32)
            p_std[:, self.land_np] = 0.0
            if project:
                # Match the training data's Leray/Helmholtz projection so the
                # divergent component of a real prior doesn't shift the input
                # out of distribution. (Linear + land-zeroing → scale-safe.)
                p_std = helmholtz_project(p_std, self.ocean_np)
            chans.append(p_std)
        return np.concatenate(chans, axis=0).astype(np.float32)

    @torch.no_grad()
    def predict(
        self,
        observations: Iterable[Sequence[float]],
        priors=None,
        *,
        n_draws: int = C.STREAM_DEFAULT_N_DRAWS,
        sampler: str = C.STREAM_SAMPLER,
        inference_steps: Optional[int] = None,
        seed: int = 0,
        project_priors: bool = True,
        full_field: bool = False,
        smooth_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the full velocity field from scattered observations + priors.

        Parameters
        ----------
        observations : iterable of (lat, lon, unix_t, u, v)
            Same convention as :meth:`DDPM.predict`.
        priors : sequence of prior fields, optional
            One field per lag (default lags 13 h, 25 h), each shaped (44, 94, 2)
            in m/s (like the output). Required for full quality.
        n_draws : int
            Ensemble members. 1 (default) → single field, uncertainty zeros.
            >1 → mean field + real per-cell ensemble std as uncertainty.
        sampler : {"dpmpp", "ddpm"}
            "dpmpp" (default) = DPM-Solver++(2M): better calibration/accuracy and
            ~24x faster on this model. "ddpm" = classic ancestral sampler
            (bit-exact to the research pipeline).
        inference_steps : int, optional
            Reverse steps. Default depends on sampler: 6 for dpmpp, 100 for ddpm.
        seed : int
            Base RNG seed for reproducibility.
        project_priors : bool, default True
            Helmholtz-project the priors to divergence-free before conditioning,
            matching how the training data was built. Keep on for real-world
            (not-quite-div-free) priors; the field the model outputs is
            divergence-free regardless of this flag.
        full_field : bool, default False
            If True, add back the irrotational (divergent) component the
            stream-function model cannot represent, via Helmholtz recombination:
            the divergence-free field from this model + the curl-free part
            extracted from a VCNN prediction. Improves accuracy against the raw
            (non-div-free) ROMS field while keeping this model's ensemble
            uncertainty. Off by default (the pure model is exactly divergence-free).
        smooth_uncertainty : bool, default True
            Apply a light nan-aware Gaussian smooth to the uncertainty field.
            Removes the grid-scale (checkerboard) numerical artifact from the
            ensemble spread — an odd-even decoupling of the central-difference
            scheme, present only in the spread, not the mean field. Improves
            calibration (r_angle/mag/overall) by ~0.025. No effect when n_draws==1.

        Returns
        -------
        mean, uncertainty : np.ndarray, each (44, 94, 2), float32, m/s.
        """
        obs_list = list(observations)
        if not obs_list:
            raise ValueError(
                "At least one observation is required; got an empty sequence.")

        # --- front-end: same rasterizer as DDPM/VCNN, then to model grid ---
        sparse_u, sparse_v, missing_mask = observations_to_channels(obs_list)
        obs_field_lib = np.stack([sparse_u, sparse_v], axis=-1)     # (44,94,2)
        obs_field_model = _lib2model_field(obs_field_lib)           # (2,94,44)
        obs_field_std = self._standardize(obs_field_model).astype(np.float32)
        obs_field_std[:, self.land_np] = 0.0
        path_mask = (_lib2model_2d(missing_mask) < 0.5) & self.ocean_np  # observed ∩ ocean

        priors_std = self._build_priors(priors, project=project_priors)

        cond = build_conditioning(
            obs_field_std, path_mask, priors_std, self.land_np, self.geom,
            legacy_obs=self._legacy_obs,
        )
        if cond.shape[0] != self.cond_ch:
            raise RuntimeError(
                f"assembled cond has {cond.shape[0]} channels but model expects "
                f"{self.cond_ch}; check priors/geometry configuration.")

        # --- diffusion ensemble (direction) + coupled magnitude fuse ---
        if sampler == "dpmpp":
            steps = inference_steps if inference_steps is not None else C.STREAM_DPMPP_STEPS
            members = dpmpp_ensemble(
                self.model, self.diffusion, cond, self.land_np,
                n_members=n_draws, inference_steps=steps,
                device=self.device, seed=seed, pred_type=self.pred_type,
            )
        elif sampler == "ddpm":
            steps = inference_steps if inference_steps is not None else C.STREAM_DDPM_STEPS
            members = ensemble_infer(
                self.model, self.diffusion, cond, self.land_np,
                n_members=n_draws, inference_steps=steps,
                device=self.device, base_seed=seed, pred_type=self.pred_type,
            )
        else:
            raise ValueError(f"sampler must be 'dpmpp' or 'ddpm'; got {sampler!r}")
        fused = fuse_coupled(
            members, cond, self.land_np,
            self._het_net, self._hsm, self._hss, self._het_clip,
            self.data_std, self.device,
        )

        arr = np.stack(fused, axis=0)                 # (K, 2, 94, 44) normalized
        mean_model = arr.mean(axis=0) * self.data_std  # (2,94,44) m/s
        if len(fused) > 1:
            unc_model = arr.std(axis=0) * self.data_std
            if smooth_uncertainty:
                unc_model = _smooth_ocean(unc_model, self.ocean_np,
                                          C.STREAM_UNC_SMOOTH_SIGMA)
        else:
            unc_model = np.zeros_like(mean_model)
        mean_model[:, self.land_np] = 0.0
        unc_model[:, self.land_np] = 0.0

        if full_field:
            # Helmholtz recombination: our (divergence-free) field + the
            # curl-free/divergent component the stream function cannot produce,
            # taken from a VCNN prediction. Deterministic addition, so it shifts
            # the mean but leaves the ensemble uncertainty unchanged.
            mean_model = mean_model + self._divergent_from_vcnn(obs_list)
            mean_model[:, self.land_np] = 0.0

        mean = _model2lib_field(mean_model).astype(np.float32)      # (44,94,2)
        uncertainty = _model2lib_field(unc_model).astype(np.float32)
        return mean, uncertainty


# ── Module-level convenience function ─────────────────────────────────

_default_instance: Optional[StreamDDPM] = None


def predict_stream(
    observations: Iterable[Sequence[float]],
    priors=None,
    *,
    device: str = "auto",
    n_draws: int = C.STREAM_DEFAULT_N_DRAWS,
    sampler: str = C.STREAM_SAMPLER,
    inference_steps: Optional[int] = None,
    seed: int = 0,
    project_priors: bool = True,
    full_field: bool = False,
    smooth_uncertainty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateless wrapper around :meth:`StreamDDPM.predict` (lazy singleton)."""
    global _default_instance
    if _default_instance is None or str(_default_instance.device) != str(
        resolve_device(device)
    ):
        _default_instance = StreamDDPM(device=device)
    return _default_instance.predict(
        observations, priors, n_draws=n_draws, sampler=sampler,
        inference_steps=inference_steps, seed=seed,
        project_priors=project_priors, full_field=full_field,
        smooth_uncertainty=smooth_uncertainty)
