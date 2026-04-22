"""Rasterize scattered (lat, lon, unix_t, u, v) observations onto the model grid.

Cells with multiple observations get their u and v values averaged.
The unix timestamp is accepted but ignored (the model is not
time-conditioned).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Tuple

import numpy as np

from .config import OCEAN_H, OCEAN_W
from .geo import lat_lon_to_index


Observation = Tuple[float, float, float, float, float]  # (lat, lon, unix_t, u, v)


def observations_to_channels(
    observations: Iterable[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter sparse observations onto (44, 94) arrays.

    Returns
    -------
    sparse_u : np.ndarray (44, 94)  — observed u values (0 where unobserved)
    sparse_v : np.ndarray (44, 94)  — observed v values (0 where unobserved)
    missing_mask : np.ndarray (44, 94)  — 1 where unobserved, 0 where observed
    """
    sum_u = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    sum_v = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    counts = np.zeros((OCEAN_H, OCEAN_W), dtype=np.int32)

    for obs in observations:
        if len(obs) != 5:
            raise ValueError(
                f"Each observation must be (lat, lon, unix_t, u, v); got length {len(obs)}"
            )
        lat, lon, _unix_t, u, v = obs
        i_lat, j_lon = lat_lon_to_index(float(lat), float(lon))
        sum_u[i_lat, j_lon] += float(u)
        sum_v[i_lat, j_lon] += float(v)
        counts[i_lat, j_lon] += 1

    observed = counts > 0
    safe = np.where(observed, counts, 1).astype(np.float32)
    sparse_u = np.where(observed, sum_u / safe, 0.0).astype(np.float32)
    sparse_v = np.where(observed, sum_v / safe, 0.0).astype(np.float32)
    missing_mask = np.where(observed, 0.0, 1.0).astype(np.float32)

    return sparse_u, sparse_v, missing_mask
