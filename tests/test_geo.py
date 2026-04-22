"""Unit tests for the geo module."""

import numpy as np
import pytest

from ddpm_library.geo import grid_arrays, in_bounds, lat_lon_to_index
from ddpm_library.config import (
    LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, OCEAN_H, OCEAN_W,
)


def test_grid_shape():
    lat, lon = grid_arrays()
    assert lat.shape == (OCEAN_H,)
    assert lon.shape == (OCEAN_W,)


def test_grid_ascending():
    lat, lon = grid_arrays()
    assert np.all(np.diff(lat) > 0)
    assert np.all(np.diff(lon) > 0)


def test_corner_indices():
    lat, lon = grid_arrays()
    assert lat_lon_to_index(lat[0], lon[0]) == (0, 0)
    assert lat_lon_to_index(lat[-1], lon[-1]) == (OCEAN_H - 1, OCEAN_W - 1)


def test_nearest_neighbour():
    lat, lon = grid_arrays()
    # A point slightly off the exact grid node should snap back
    assert lat_lon_to_index(lat[10] + 1e-8, lon[20] + 1e-8) == (10, 20)


def test_out_of_bounds_raises():
    # Far outside
    with pytest.raises(ValueError):
        lat_lon_to_index(0.0, 0.0)
    # Just past north edge
    with pytest.raises(ValueError):
        lat_lon_to_index(LAT_MAX + 1e-3, (LON_MIN + LON_MAX) / 2)
    # Just past west edge
    with pytest.raises(ValueError):
        lat_lon_to_index((LAT_MIN + LAT_MAX) / 2, LON_MIN - 1e-3)


def test_in_bounds_tolerance():
    # Exact boundaries should be accepted
    assert in_bounds(LAT_MIN, LON_MIN)
    assert in_bounds(LAT_MAX, LON_MAX)
    # Slightly outside: out of bounds
    assert not in_bounds(LAT_MAX + 1e-3, LON_MAX)
