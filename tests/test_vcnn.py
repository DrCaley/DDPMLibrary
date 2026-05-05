"""Smoke tests for the V-CNN baseline predictor."""

import numpy as np
import pytest

from ddpm_library import VCNN
from ddpm_library.geo import grid_arrays


@pytest.fixture(scope="module")
def vcnn():
    return VCNN(device="cpu")


def test_vcnn_predict_shape_and_finite(vcnn):
    lat, lon = grid_arrays()
    rng = np.random.default_rng(0)
    obs = [
        (float(lat[rng.integers(0, 44)]),
         float(lon[rng.integers(0, 94)]),
         1_700_000_000.0,
         float(rng.normal(0, 0.1)),
         float(rng.normal(0, 0.1)))
        for _ in range(5)
    ]
    mean, unc = vcnn.predict(obs)
    assert mean.shape == (44, 94, 2)
    assert unc.shape == (44, 94, 2)
    assert mean.dtype == np.float32
    assert np.isfinite(mean).all()
    assert (unc == 0).all()


def test_vcnn_deterministic(vcnn):
    lat, lon = grid_arrays()
    obs = [(float(lat[10]), float(lon[20]), 0.0, 0.05, -0.03)]
    m1, _ = vcnn.predict(obs)
    m2, _ = vcnn.predict(obs)
    np.testing.assert_array_equal(m1, m2)


def test_vcnn_land_cells_zero(vcnn):
    """Output should be zero on every land cell of the bundled ocean mask."""
    lat, lon = grid_arrays()
    obs = [(float(lat[22]), float(lon[47]), 0.0, 0.05, -0.03)]
    mean, _ = vcnn.predict(obs)
    om = vcnn.ocean_mask  # (44, 94)
    assert mean[om < 0.5].sum() == 0.0


def test_vcnn_empty_observations_raises(vcnn):
    with pytest.raises(ValueError):
        vcnn.predict([])
