"""Tests for the classical GP (Matern kriging) baseline.

No checkpoint is needed -- the GP fits per call -- so unlike the other model
tests these always run, provided the shared grid asset is present.
"""

import numpy as np
import pytest

from ddpm_library.config import (CORRDIFF_GRID_PATH, LAT_MAX, LAT_MIN, LON_MAX,
                                 LON_MIN, OCEAN_H, OCEAN_W)

_needs_grid = pytest.mark.skipif(
    not CORRDIFF_GRID_PATH.exists(), reason="grid asset not present (git-lfs pull)")

pytest.importorskip("sklearn", reason="scikit-learn not installed")


@pytest.fixture(scope="module")
def model():
    from ddpm_library import GP
    return GP()


@pytest.fixture(scope="module")
def smooth_obs(model):
    """Observations sampled from a SMOOTH field, so the GP has real structure to
    fit. Random values would drive the length scale to its lower bound."""
    ocean = model.ocean_mask > 0.5
    lats = np.linspace(LAT_MIN, LAT_MAX, OCEAN_H)
    lons = np.linspace(LON_MIN, LON_MAX, OCEAN_W)
    yy, xx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
    u = np.sin(2 * np.pi * yy / 30) * np.cos(2 * np.pi * xx / 40)
    v = np.cos(2 * np.pi * yy / 25) * np.sin(2 * np.pi * xx / 35)
    rng = np.random.default_rng(0)
    cells = np.argwhere(ocean)
    pick = cells[rng.choice(len(cells), 80, replace=False)]
    obs = [(float(lats[i]), float(lons[j]), 1.7e9, float(u[i, j]), float(v[i, j]))
           for i, j in pick]
    return obs, pick, (u, v)


@_needs_grid
def test_output_contract(model, smooth_obs):
    obs, _, _ = smooth_obs
    mean, unc = model.predict(obs, seed=0)
    ocean = model.ocean_mask > 0.5
    for a in (mean, unc):
        assert a.shape == (OCEAN_H, OCEAN_W, 2) and a.dtype == np.float32
        assert np.all(np.isfinite(a))
        assert np.all(a[~ocean] == 0)
    assert model.takes_priors is False


@_needs_grid
def test_uncertainty_is_small_at_observations_and_large_away(model, smooth_obs):
    """The defining property of a GP posterior, and why it is the calibration
    reference: sigma collapses where you measured and grows where you did not."""
    obs, pick, _ = smooth_obs
    _, unc = model.predict(obs, seed=0)
    om = np.zeros((OCEAN_H, OCEAN_W), bool)
    for i, j in pick:
        om[i, j] = True
    ocean = model.ocean_mask > 0.5
    assert unc[om].mean() < 0.2 * unc[ocean & ~om].mean()


@_needs_grid
def test_reproduces_observed_values(model, smooth_obs):
    """With a small noise floor the posterior mean should nearly interpolate."""
    obs, pick, _ = smooth_obs
    mean, _ = model.predict(obs, seed=0)
    err = [abs(mean[i, j, 0] - o[3]) for (i, j), o in zip(pick, obs)]
    assert np.mean(err) < 0.05


@_needs_grid
def test_predict_takes_no_priors(model, smooth_obs):
    obs, _, _ = smooth_obs
    with pytest.raises(TypeError):
        model.predict(obs, [np.zeros((OCEAN_H, OCEAN_W, 2), np.float32)] * 2)


@_needs_grid
def test_empty_observations_rejected(model):
    with pytest.raises(ValueError, match="(?i)at least one observation"):
        model.predict([])


@_needs_grid
def test_out_of_bounds_rejected(model):
    with pytest.raises(ValueError, match="outside the model"):
        model.predict([(0.0, 0.0, 1.7e9, 0.1, 0.0)])


@_needs_grid
def test_no_checkpoint_needed(model):
    """Documents the property that makes this baseline free to distribute."""
    assert not hasattr(model, "model")
    import ddpm_library.config as C
    assert not any("gp" in p.name.lower()
                   for p in C._ASSETS_DIR.glob("*.pt"))
