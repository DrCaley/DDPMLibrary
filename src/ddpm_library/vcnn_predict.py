"""High-level API: VCNN.predict(observations) -> (mean, uncertainty).

The Voronoi-CNN baseline (Fukami et al. 2021) reconstructs a full velocity
field from sparse observations by first building a Voronoi tessellation of
the observed values, then refining it with a small U-Net. It is purely
deterministic — one forward pass, no diffusion.

Example
-------
    from ddpm_library import VCNN

    obs = [(lat, lon, unix_t, u, v), ...]
    model = VCNN(device="auto")
    mean, uncertainty = model.predict(obs)
    # mean.shape == (44, 94, 2)   # uncertainty is all zeros
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import OCEAN_H, OCEAN_W, VCNN_WEIGHTS_PATH
from .inference import resolve_device
from .model.vcnn import VoronoiCNN, build_voronoi_input
from .rasterize import observations_to_channels


def _load_vcnn(
    weights_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> tuple[VoronoiCNN, np.ndarray, np.ndarray, np.ndarray]:
    """Load V-CNN weights + per-component normalization stats + ocean mask.

    Returns (model, norm_mean (2,), norm_std (2,), ocean_mask (44, 94)).
    """
    path = Path(weights_path) if weights_path is not None else VCNN_WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"V-CNN weights not found at {path}. The library ships a default "
            f"checkpoint; if it is missing, ensure git-lfs is installed and "
            f"`git lfs pull` has been run."
        )
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ck["model_config"]
    model = VoronoiCNN(**cfg)
    model.load_state_dict(ck["model_state"])
    model.eval()
    if device is not None:
        model = model.to(device)

    norm_mean = np.asarray(ck["norm_mean"], dtype=np.float32).reshape(2)
    norm_std = np.asarray(ck["norm_std"], dtype=np.float32).reshape(2)
    ocean_mask = np.asarray(ck["ocean_mask"], dtype=np.float32)
    if ocean_mask.shape != (OCEAN_H, OCEAN_W):
        raise ValueError(
            f"V-CNN checkpoint ocean_mask has shape {ocean_mask.shape}, "
            f"expected ({OCEAN_H}, {OCEAN_W})"
        )
    return model, norm_mean, norm_std, ocean_mask


class VCNN:
    """Voronoi-CNN predictor — a fast, deterministic alternative to the DDPM.

    Same input/output contract as :class:`DDPM`. ``predict`` returns
    ``(mean, uncertainty)`` where ``uncertainty`` is currently zeros.

    Parameters
    ----------
    device : str
        ``"auto"`` (default) picks CUDA > MPS > CPU. Pass an explicit
        torch device string (e.g. ``"cpu"``, ``"cuda"``) to override.
    weights_path : str or pathlib.Path, optional
        Path to a V-CNN checkpoint. Defaults to the bundled
        ``vcnn_weights.pt``.
    """

    def __init__(
        self,
        device: str = "auto",
        weights_path: Optional[str | Path] = None,
    ) -> None:
        self.device = resolve_device(device)
        net, norm_mean, norm_std, ocean_mask = _load_vcnn(
            weights_path=weights_path, device=self.device
        )
        self.net = net
        self._norm_mean = norm_mean
        self._norm_std = norm_std
        self._ocean_mask = ocean_mask

    @property
    def ocean_mask(self) -> np.ndarray:
        """A copy of the (44, 94) ocean mask baked into the checkpoint."""
        return self._ocean_mask.copy()

    def predict(
        self,
        observations: Iterable[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the full velocity field from scattered observations.

        Parameters
        ----------
        observations : iterable of (lat, lon, unix_t, u, v)
            See :meth:`DDPM.predict` for the coordinate convention.

        Returns
        -------
        mean : np.ndarray, shape (44, 94, 2), dtype float32
            Predicted velocity field on the model grid (lat, lon, [u, v]).
            Land cells (per the bundled ocean mask) are zero.
        uncertainty : np.ndarray, shape (44, 94, 2), dtype float32
            Currently all zeros — V-CNN is a deterministic regressor.

        Raises
        ------
        ValueError
            If observations is empty or any (lat, lon) is outside the
            covered region.
        """
        obs_list = list(observations)
        if not obs_list:
            raise ValueError(
                "At least one observation is required; got an empty sequence."
            )

        sparse_u, sparse_v, missing_mask = observations_to_channels(obs_list)
        # `obs_mask` is the V-CNN convention: 1 = observed.
        obs_mask = (1.0 - missing_mask).astype(np.float32)

        # Stack into (2, H, W) and z-score per-component.
        vel = np.stack([sparse_u, sparse_v], axis=0)
        nm = self._norm_mean.reshape(2, 1, 1)
        ns = self._norm_std.reshape(2, 1, 1)
        vel_n = ((vel - nm) / ns) * self._ocean_mask[None]

        vi = build_voronoi_input(vel_n, obs_mask, self._ocean_mask)
        with torch.no_grad():
            x = torch.from_numpy(vi).unsqueeze(0).to(self.device)
            pred = self.net(x).squeeze(0).cpu().numpy()  # (2, H, W)

        # Inverse standardise + apply ocean mask.
        pred_phys = (pred * ns + nm) * self._ocean_mask[None]
        # Reorder to (H, W, 2) to match DDPM.predict's convention.
        mean = np.transpose(pred_phys, (1, 2, 0)).astype(np.float32)
        uncertainty = np.zeros_like(mean)
        return mean, uncertainty


# ── Module-level convenience function ─────────────────────────────────

_default_instance: Optional[VCNN] = None


def predict_vcnn(
    observations: Iterable[Sequence[float]],
    *,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Stateless wrapper around :meth:`VCNN.predict`.

    Lazy-loads the V-CNN model on first call and caches it. Subsequent
    calls reuse the cached model unless ``device`` changes.
    """
    global _default_instance
    if _default_instance is None or str(_default_instance.device) != str(
        resolve_device(device)
    ):
        _default_instance = VCNN(device=device)
    return _default_instance.predict(observations)
