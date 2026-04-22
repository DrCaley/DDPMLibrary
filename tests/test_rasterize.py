"""Unit tests for observations_to_channels rasterizer."""

import numpy as np

from ddpm_library.rasterize import observations_to_channels
from ddpm_library.geo import grid_arrays


def test_shapes():
    lat, lon = grid_arrays()
    obs = [(lat[5], lon[10], 0.0, 0.1, -0.05)]
    u, v, m = observations_to_channels(obs)
    assert u.shape == (44, 94)
    assert v.shape == (44, 94)
    assert m.shape == (44, 94)


def test_single_obs_placement():
    lat, lon = grid_arrays()
    obs = [(lat[3], lon[7], 0.0, 0.25, -0.75)]
    u, v, m = observations_to_channels(obs)
    assert u[3, 7] == np.float32(0.25)
    assert v[3, 7] == np.float32(-0.75)
    # missing_mask: 0.0 at observed cell, 1.0 everywhere else
    assert m[3, 7] == 0.0
    assert m.sum() == (44 * 94 - 1)


def test_colocated_averages():
    lat, lon = grid_arrays()
    obs = [
        (lat[2], lon[4], 0.0, 1.0, 2.0),
        (lat[2], lon[4], 1.0, 3.0, 4.0),
    ]
    u, v, m = observations_to_channels(obs)
    assert np.isclose(u[2, 4], 2.0)  # (1+3)/2
    assert np.isclose(v[2, 4], 3.0)  # (2+4)/2
    assert m[2, 4] == 0.0
    assert m.sum() == (44 * 94 - 1)


def test_empty():
    u, v, m = observations_to_channels([])
    assert (u == 0).all()
    assert (v == 0).all()
    # Nothing observed → every cell is "missing"
    assert (m == 1).all()
