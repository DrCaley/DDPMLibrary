"""Conditioning-tensor assembly for the CorrDiff residual model.

Vendored from the research pipeline (``Utils/cond_dataset.py`` and
``Conditional DDPM/train_corrdiff_v2.py``). The channel ORDER and each channel's
normalisation must match training exactly -- a permuted or differently-scaled
channel silently produces a plausible-looking but wrong field -- so the layout is
spelled out here and asserted at load time by the predictor.

Model-grid orientation throughout: (C, 94, 44), land already zeroed.

Full input to the network is ``[x_t | cond | mean | dist | sigma?]``:

    x_t     2   the noisy residual being denoised
    ---- cond (11) --------------------------------------------------------
    obs_u   1   observed u on the path, 0 elsewhere      (standardised units)
    obs_v   1   observed v on the path, 0 elsewhere      (standardised units)
    mask    1   1.0 on observed cells, 0.0 elsewhere
    dpath   1   distance to nearest observation, /max over ocean -> [0, 1]
    priors  4   two lag fields (13 h, 25 h) x (u, v)     (standardised units)
    geom    3   coord_x, coord_y in [-1, 1]; distance-to-coast /max -> [0, 1]
    ---- augmentation -----------------------------------------------------
    mean    2   the deterministic anchor (V-CNN field)   (standardised units)
    dist    1   distance to nearest observation / 20.0   (UNSCALED by max --
                deliberately a different normalisation from `dpath` above)
    sigma   1   sensor-noise level / noise_max, constant map (dial model only)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Channel-group sizes, for assertions and documentation.
N_OBS_CH = 4
N_GEOM_CH = 3
DIST_SCALE = 20.0          # train_corrdiff_v2.dist_channel


def geometry_channels(land_np: np.ndarray) -> np.ndarray:
    """Static geometry: (3, H, W) = [coord_x, coord_y, distance-to-coast].

    coord_x/coord_y are linspace(-1, 1) along width/height; distance-to-coast is
    the EDT of the ocean mask normalised by its own maximum, land zeroed.
    """
    land = np.asarray(land_np, dtype=bool)
    ocean = ~land
    H, W = land.shape
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)
    dist = ndimage.distance_transform_edt(ocean).astype(np.float32)
    dmax = float(dist.max())
    if dmax > 0:
        dist = dist / dmax
    dist[land] = 0.0
    return np.stack([xs, ys, dist], axis=0).astype(np.float32)


def observation_channels(obs_field_std: np.ndarray, path_mask: np.ndarray,
                         land_np: np.ndarray) -> np.ndarray:
    """(4, H, W) = [obs_u, obs_v, path_mask, dist_to_path].

    ``obs_field_std`` is the already-standardised (2, H, W) field carrying the
    measured values on the path (zeros elsewhere). ``dist_to_path`` is normalised
    by its maximum over ocean cells -- note this is a DIFFERENT normalisation from
    :func:`dist_channel`, and both appear in the input; keep them distinct.
    """
    land = np.asarray(land_np, dtype=bool)
    ocean = ~land
    pm = np.asarray(path_mask, dtype=bool)

    obs = np.zeros_like(obs_field_std, dtype=np.float32)
    obs[:, pm] = obs_field_std[:, pm]
    mask = pm.astype(np.float32)[None]

    dist = ndimage.distance_transform_edt(~pm).astype(np.float32)
    dist[land] = 0.0
    dmax = float(dist[ocean].max()) if ocean.any() and dist[ocean].max() > 0 else 1.0
    dist = dist / dmax
    dist[land] = 0.0
    return np.concatenate([obs, mask, dist[None]], axis=0).astype(np.float32)


def dist_channel(path_mask: np.ndarray, scale: float = DIST_SCALE) -> np.ndarray:
    """(1, H, W) distance-to-observation map divided by a FIXED scale.

    Deliberately not normalised by the per-frame maximum (unlike the
    ``dist_to_path`` channel inside :func:`observation_channels`), so the model
    sees an absolute measure of how far a cell is from any measurement.
    """
    pm = np.asarray(path_mask, dtype=bool)
    return (ndimage.distance_transform_edt(~pm) / scale).astype(np.float32)[None]


def assemble_cond(obs: np.ndarray, priors: np.ndarray, geom: np.ndarray) -> np.ndarray:
    """Concatenate the three conditioning groups in the trained order."""
    return np.concatenate([obs, priors, geom], axis=0).astype(np.float32)


def sigma_channel(sensor_noise: float, noise_max: float,
                  shape: tuple[int, int]) -> np.ndarray:
    """(1, H, W) constant map encoding the sensor-noise level ("the dial").

    The model was trained with sigma ~ U(0, noise_max) as a fraction of the field
    standard deviation, presented as ``sigma / noise_max``. Values outside
    [0, noise_max] are outside the trained range and are rejected by the caller.
    """
    return np.full((1, *shape), sensor_noise / max(noise_max, 1e-8), dtype=np.float32)
