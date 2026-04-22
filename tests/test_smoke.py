"""End-to-end smoke test: small DDPM.predict run on CPU."""

import numpy as np
import pytest

from ddpm_library import DDPM
from ddpm_library.geo import grid_arrays


@pytest.fixture(scope="module")
def ddpm():
    return DDPM(device="cpu")


def test_predict_shape_and_finite(ddpm):
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
    # Very small t_start for speed
    mean, unc = ddpm.predict(obs, t_start=5, resample_steps=1, seed=42)
    assert mean.shape == (44, 94, 2)
    assert unc.shape == (44, 94, 2)
    assert mean.dtype == np.float32
    assert np.isfinite(mean).all()
    assert (unc == 0).all()  # uncertainty stub


def test_predict_single_obs(ddpm):
    lat, lon = grid_arrays()
    obs = [(float(lat[22]), float(lon[47]), 0.0, 0.05, -0.03)]
    mean, _ = ddpm.predict(obs, t_start=5, resample_steps=1, seed=0)
    assert mean.shape == (44, 94, 2)
    assert np.isfinite(mean).all()


def test_predict_deterministic(ddpm):
    lat, lon = grid_arrays()
    obs = [(float(lat[10]), float(lon[20]), 0.0, 0.05, -0.03)]
    m1, _ = ddpm.predict(obs, t_start=5, resample_steps=1, seed=123)
    m2, _ = ddpm.predict(obs, t_start=5, resample_steps=1, seed=123)
    np.testing.assert_allclose(m1, m2)


def test_iterative_mode_runs(ddpm):
    """Full RePaint chain still works when single_step=False."""
    lat, lon = grid_arrays()
    obs = [(float(lat[22]), float(lon[47]), 0.0, 0.05, -0.03)]
    mean, _ = ddpm.predict(
        obs, single_step=False, t_start=3, resample_steps=1, seed=0,
    )
    assert mean.shape == (44, 94, 2)
    assert np.isfinite(mean).all()


def test_single_step_preserves_observations(ddpm):
    """In single-step mode, observed cells must equal the input exactly."""
    lat, lon = grid_arrays()
    obs = [
        (float(lat[5]),  float(lon[10]), 0.0,  0.12, -0.03),
        (float(lat[30]), float(lon[70]), 0.0, -0.05,  0.08),
    ]
    mean, _ = ddpm.predict(obs, single_step=True, seed=0)
    for la, lo, _, u, v in obs:
        i = int(np.argmin(np.abs(lat - la)))
        j = int(np.argmin(np.abs(lon - lo)))
        np.testing.assert_allclose(mean[i, j, 0], u, atol=1e-5)
        np.testing.assert_allclose(mean[i, j, 1], v, atol=1e-5)
