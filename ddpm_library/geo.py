"""Map (lat, lon) to the model's grid indices.

The split-head model operates on a fixed 44×94 grid (lat × lon) covering
the St. John Rams Head "northwest" region. This module loads the exact
lat/lon arrays extracted from the source ROMS .mat file (shipped in
assets/lat_lon_grid.npz) and maps arbitrary query coordinates to grid
indices via nearest-neighbour lookup. Out-of-bounds queries raise
ValueError.
"""

from functools import lru_cache

import numpy as np

from .config import (
    LAT_LON_GRID_PATH, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, OCEAN_H, OCEAN_W,
)


@lru_cache(maxsize=1)
def _load_grid() -> tuple[np.ndarray, np.ndarray]:
    """Load (lat[44], lon[94]) arrays from the bundled npz."""
    if not LAT_LON_GRID_PATH.exists():
        raise FileNotFoundError(
            f"Grid file missing: {LAT_LON_GRID_PATH}. Re-fetch the library "
            f"assets."
        )
    data = np.load(LAT_LON_GRID_PATH)
    lat = np.asarray(data["lat"], dtype=np.float64)
    lon = np.asarray(data["lon"], dtype=np.float64)
    if lat.shape != (OCEAN_H,):
        raise ValueError(
            f"Expected lat array of shape ({OCEAN_H},), got {lat.shape}"
        )
    if lon.shape != (OCEAN_W,):
        raise ValueError(
            f"Expected lon array of shape ({OCEAN_W},), got {lon.shape}"
        )
    return lat, lon


def grid_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return (lat[44], lon[94]) arrays corresponding to the model grid rows/cols."""
    lat, lon = _load_grid()
    return lat.copy(), lon.copy()


def in_bounds(lat: float, lon: float, tol: float = 1e-6) -> bool:
    """True if (lat, lon) lies inside the model's covered region."""
    return (
        LAT_MIN - tol <= lat <= LAT_MAX + tol
        and LON_MIN - tol <= lon <= LON_MAX + tol
    )


def lat_lon_to_index(lat: float, lon: float) -> tuple[int, int]:
    """Nearest-neighbour grid index (i_lat, j_lon) for a query (lat, lon).

    Raises ValueError if the query is outside the model's bounding box.
    """
    if not in_bounds(lat, lon):
        raise ValueError(
            f"Observation ({lat}, {lon}) is outside the model's covered "
            f"region: lat ∈ [{LAT_MIN}, {LAT_MAX}], "
            f"lon ∈ [{LON_MIN}, {LON_MAX}]"
        )
    lat_arr, lon_arr = _load_grid()
    i_lat = int(np.argmin(np.abs(lat_arr - lat)))
    j_lon = int(np.argmin(np.abs(lon_arr - lon)))
    return i_lat, j_lon
