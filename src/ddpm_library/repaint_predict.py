"""High-level API for the two collaborator RePaint models.

Both are packaged with the SAME contract as every other predictor here, so they
can be compared like for like::

    from ddpm_library import RePaint, RePaintUncond

    mean, unc = RePaint(device="auto").predict(observations, priors, n_draws=10)
    mean, unc = RePaintUncond(device="auto").predict(observations, n_draws=10)

The two models
--------------
They share one architecture and differ only in what the network is allowed to
see, so they get one class each rather than one class and a weights argument:

* :class:`RePaint` -- time-conditioned. Takes the ocean state ~13 h and ~25 h
  earlier as four extra input channels. This is the collaborator's primary
  model and the one with published reconstruction numbers.
* :class:`RePaintUncond` -- unconditional. No temporal priors; its ``predict``
  takes no ``priors`` argument at all.

Each class pins its own checkpoint and refuses one that does not match, so a
wrong path fails loudly instead of silently becoming the other model.

How these differ from :class:`CorrDiff`
---------------------------------------
Neither is trained on the observations. The sparse measurements are imposed at
SAMPLING time by a guidance term (DPS, or MCG which additionally replaces the
observed cells each step). CorrDiff instead feeds the observations to the
network as conditioning and generates a residual around a deterministic mean.
Comparing them is comparing guided sampling against trained conditioning.

They also work in PHYSICAL m/s -- they are not z-scored -- so no normalization
is applied here. Getting that wrong silently produces plausible but wrong fields.

Uncertainty
-----------
The sampler is stochastic, so ``n_draws > 1`` gives a real per-cell ensemble
spread. Unlike :class:`CorrDiff` there is no fitted calibration factor for
either model, so the returned ``uncertainty`` is the RAW ensemble standard
deviation and is expected to be under-dispersed. Do not read it as a calibrated
1-sigma; fit a scale factor on held-out data first if you need one.

Cost
----
Guided sampling runs the full reverse chain WITH a gradient at every step, so it
is substantially slower than CorrDiff's DDIM sampler. Use ``stride`` to
subsample the chain when you need speed.
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


# --------------------------------------------------------------------------- #
# Shared machinery
# --------------------------------------------------------------------------- #
class _RePaintBase:
    """Checkpoint loading and guided sampling shared by both RePaint variants.

    Subclasses declare which checkpoint they own and whether it is conditioned;
    everything below is identical between them. Not part of the public API --
    use :class:`RePaint` or :class:`RePaintUncond`.
    """

    #: Bundled checkpoint for this variant. Set by each subclass.
    _DEFAULT_WEIGHTS: Path
    #: Whether this variant's checkpoint must carry conditioning channels.
    _EXPECT_CONDITIONED: bool
    #: True if :meth:`predict` accepts a ``priors`` argument.
    takes_priors: bool

    def __init__(self, device: str = "auto", weights_path: Optional[str | Path] = None):
        self.device = resolve_device(device)

        wpath = Path(weights_path) if weights_path else self._DEFAULT_WEIGHTS
        if not wpath.exists():
            raise FileNotFoundError(
                f"{type(self).__name__} weights not found at {wpath} (git-lfs pull?).")
        ck = torch.load(wpath, map_location="cpu", weights_only=False)
        a = ck.get("args", {})

        # The time-conditioned checkpoint stores cond_ch=4 / lags=(13, 25); the
        # unconditional one stores an explicit None for both, so `or` (not a .get
        # default) is what handles it.
        self.cond_ch = int(ck.get("cond_ch") or 0)
        self.lags = tuple(int(x) for x in (ck.get("lags") or ()))
        if self.cond_ch != 2 * len(self.lags):
            raise RuntimeError(
                f"checkpoint cond_ch={self.cond_ch} is inconsistent with lags="
                f"{self.lags} (expected {2 * len(self.lags)} channels).")
        self.conditional = self.cond_ch > 0

        # Fail loudly rather than silently running the sibling model.
        if self.conditional != self._EXPECT_CONDITIONED:
            want = "time-conditioned" if self._EXPECT_CONDITIONED else "unconditional"
            got = "time-conditioned" if self.conditional else "unconditional"
            other = "RePaint" if self.conditional else "RePaintUncond"
            raise ValueError(
                f"{type(self).__name__} only loads {want} checkpoints, but "
                f"{wpath.name} is {got} (cond_ch={self.cond_ch}). "
                f"Use {other} for this checkpoint.")

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

    def _sample(
        self,
        observations: Iterable[Sequence[float]],
        cond_t: Optional[torch.Tensor],
        *,
        n_draws: int,
        sampler: str,
        step_size: float,
        stride: int,
        seed,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize the observations, run the guided chain, reduce to mean/spread."""
        obs_list = list(observations)
        if not obs_list:
            raise ValueError(
                "At least one observation is required; got an empty sequence.")
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1; got {n_draws}.")
        if sampler not in SAMPLERS:
            raise ValueError(
                f"sampler must be one of {sorted(SAMPLERS)}; got {sampler!r}")

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
        x0_known[:, ~path_mask] = 0.0      # observed cells only, as the sampler expects
        x0_known_t = torch.from_numpy(x0_known)

        infer = SAMPLERS[sampler]
        draws = []
        for k in range(n_draws):
            if seed is not None:               # seed=None -> non-reproducible draws,
                torch.manual_seed(seed + k)    # matching the rest of the library
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


# --------------------------------------------------------------------------- #
# Time-conditioned variant
# --------------------------------------------------------------------------- #
class RePaint(_RePaintBase):
    """Linear-schedule RePaint UNet with temporal priors, sampled with DPS/MCG guidance.

    The collaborator's primary model: the ocean state ~13 h and ~25 h earlier is
    supplied as four extra input channels. For the sibling model that uses no
    priors, see :class:`RePaintUncond`.

    Parameters
    ----------
    device : str
        ``"auto"`` (CUDA > MPS > CPU) or an explicit torch device string.
    weights_path : str or Path, optional
        Override the bundled checkpoint. Must itself be time-conditioned.
    """

    _DEFAULT_WEIGHTS = C.REPAINT_WEIGHTS_PATH
    _EXPECT_CONDITIONED = True
    takes_priors = True

    def _build_priors(self, priors) -> np.ndarray:
        """(cond_ch, 94, 44) temporal priors in PHYSICAL m/s (no standardization)."""
        if priors is None:
            warnings.warn(
                "RePaint.predict called without `priors`; this model is conditioned on "
                "the ocean state ~13h and ~25h earlier. Zeroing them degrades quality "
                "-- pass `priors` for best results, or use RePaintUncond if you have "
                "no prior fields.",
                stacklevel=3,
            )
            return np.zeros((self.cond_ch, C.REPAINT_H, C.REPAINT_W), dtype=np.float32)

        priors = list(priors)
        if len(priors) != len(self.lags):
            raise ValueError(
                f"priors must contain {len(self.lags)} fields (one per lag "
                f"{self.lags}); got {len(priors)}.")
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
        seed : int or None
            Base RNG seed; draw ``k`` uses ``seed + k``. ``None`` (the default)
            leaves the global RNG untouched, so draws are not reproducible.

        Returns
        -------
        mean, uncertainty : np.ndarray, each (44, 94, 2), float32, m/s.
        """
        cond_np = self._build_priors(priors)
        cond_t = torch.from_numpy(cond_np).unsqueeze(0).to(self.device)
        return self._sample(observations, cond_t, n_draws=n_draws, sampler=sampler,
                            step_size=step_size, stride=stride, seed=seed)


# --------------------------------------------------------------------------- #
# Unconditional variant
# --------------------------------------------------------------------------- #
class RePaintUncond(_RePaintBase):
    """Linear-schedule RePaint UNet with NO temporal priors, sampled with DPS/MCG.

    Architecturally identical to :class:`RePaint` -- ``cond_ch=0`` reproduces the
    same network without the four prior channels -- but trained without them, so
    the sparse observations are its only information about the target frame.

    Parameters
    ----------
    device : str
        ``"auto"`` (CUDA > MPS > CPU) or an explicit torch device string.
    weights_path : str or Path, optional
        Override the bundled checkpoint. Must itself be unconditional.
    """

    _DEFAULT_WEIGHTS = C.REPAINT_UNCOND_WEIGHTS_PATH
    _EXPECT_CONDITIONED = False
    takes_priors = False

    def predict(
        self,
        observations: Iterable[Sequence[float]],
        *,
        n_draws: int = C.REPAINT_DEFAULT_N_DRAWS,
        sampler: str = C.REPAINT_SAMPLER,
        step_size: float = C.REPAINT_STEP_SIZE,
        stride: int = C.REPAINT_STRIDE,
        seed=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the field from scattered observations alone.

        Takes no ``priors``: this model was trained without them. See
        :meth:`RePaint.predict` for the shared parameters and return contract.
        """
        return self._sample(observations, None, n_draws=n_draws, sampler=sampler,
                            step_size=step_size, stride=stride, seed=seed)


# --------------------------------------------------------------------------- #
# Module-level convenience functions
# --------------------------------------------------------------------------- #
_default_instances: dict[type, _RePaintBase] = {}


def _cached(cls: type, device: str) -> _RePaintBase:
    """Lazy per-class singleton, rebuilt when the requested device changes."""
    inst = _default_instances.get(cls)
    if inst is None or str(inst.device) != str(resolve_device(device)):
        inst = cls(device=device)
        _default_instances[cls] = inst
    return inst


def predict_repaint(
    observations: Iterable[Sequence[float]],
    priors=None,
    *,
    device: str = "auto",
    n_draws: int = C.REPAINT_DEFAULT_N_DRAWS,
    sampler: str = C.REPAINT_SAMPLER,
    step_size: float = C.REPAINT_STEP_SIZE,
    stride: int = C.REPAINT_STRIDE,
    seed=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateless wrapper around :meth:`RePaint.predict` (lazy singleton)."""
    return _cached(RePaint, device).predict(
        observations, priors, n_draws=n_draws, sampler=sampler,
        step_size=step_size, stride=stride, seed=seed)


def predict_repaint_uncond(
    observations: Iterable[Sequence[float]],
    *,
    device: str = "auto",
    n_draws: int = C.REPAINT_DEFAULT_N_DRAWS,
    sampler: str = C.REPAINT_SAMPLER,
    step_size: float = C.REPAINT_STEP_SIZE,
    stride: int = C.REPAINT_STRIDE,
    seed=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateless wrapper around :meth:`RePaintUncond.predict` (lazy singleton)."""
    return _cached(RePaintUncond, device).predict(
        observations, n_draws=n_draws, sampler=sampler,
        step_size=step_size, stride=stride, seed=seed)
