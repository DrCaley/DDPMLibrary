"""Smoke tests for the StreamDDPM (conditional stream-function) predictor.

Uses a small inference_steps for speed. Requires the bundled stream weights
(git-lfs), like test_vcnn requires the V-CNN weights.
"""

import numpy as np
import pytest

from ddpm_library import StreamDDPM
from ddpm_library.geo import grid_arrays


@pytest.fixture(scope="module")
def stream():
    return StreamDDPM(device="cpu")


def _obs(n=8, seed=0):
    lat, lon = grid_arrays()
    rng = np.random.default_rng(seed)
    return [
        (float(lat[rng.integers(0, 44)]),
         float(lon[rng.integers(0, 94)]),
         1_700_000_000.0,
         float(rng.normal(0, 0.1)),
         float(rng.normal(0, 0.1)))
        for _ in range(n)
    ]


def _priors():
    # two prior fields (lags 13h, 25h), library orientation (44, 94, 2)
    return [np.zeros((44, 94, 2), dtype=np.float32) for _ in range(2)]


def test_stream_predict_shape_and_finite(stream):
    mean, unc = stream.predict(_obs(), _priors(), n_draws=1,
                               inference_steps=5, seed=0)
    assert mean.shape == (44, 94, 2)
    assert unc.shape == (44, 94, 2)
    assert mean.dtype == np.float32
    assert np.isfinite(mean).all()
    assert (unc == 0).all()          # single draw → uncertainty stub is zeros


def test_stream_ensemble_uncertainty(stream):
    """With n_draws > 1 the uncertainty is a real (non-zero) ensemble spread."""
    mean, unc = stream.predict(_obs(), _priors(), n_draws=3,
                               inference_steps=5, seed=0)
    assert mean.shape == (44, 94, 2)
    assert np.isfinite(unc).all()
    assert float(np.abs(unc).sum()) > 0.0


def test_stream_deterministic(stream):
    obs, pri = _obs(), _priors()
    m1, u1 = stream.predict(obs, pri, n_draws=2, inference_steps=5, seed=7)
    m2, u2 = stream.predict(obs, pri, n_draws=2, inference_steps=5, seed=7)
    np.testing.assert_allclose(m1, m2)
    np.testing.assert_allclose(u1, u2)


def test_stream_land_cells_zero(stream):
    mean, _ = stream.predict(_obs(), _priors(), n_draws=1,
                             inference_steps=5, seed=0)
    om = stream.ocean_mask               # (44, 94), 1 = ocean — same API as VCNN
    assert float(np.abs(mean[om < 0.5]).sum()) == 0.0


def test_stream_empty_observations_raises(stream):
    with pytest.raises(ValueError):
        stream.predict([], _priors())


def test_stream_wrong_prior_count_raises(stream):
    with pytest.raises(ValueError):
        stream.predict(_obs(), [np.zeros((44, 94, 2), dtype=np.float32)])  # need 2


def test_stream_missing_priors_warns(stream):
    with pytest.warns(UserWarning):
        stream.predict(_obs(), None, n_draws=1, inference_steps=5, seed=0)


def test_stream_ddpm_sampler_runs(stream):
    """The classic ancestral sampler is still selectable and valid."""
    mean, unc = stream.predict(_obs(), _priors(), sampler="ddpm",
                               n_draws=1, inference_steps=10, seed=0)
    assert mean.shape == (44, 94, 2)
    assert np.isfinite(mean).all()


def test_stream_default_sampler_is_dpmpp_and_runs(stream):
    """Default sampler runs and returns a valid field (fast path)."""
    mean, _ = stream.predict(_obs(), _priors(), n_draws=1, seed=0)  # default dpmpp
    assert mean.shape == (44, 94, 2)
    assert np.isfinite(mean).all()


def test_stream_invalid_sampler_raises(stream):
    with pytest.raises(ValueError):
        stream.predict(_obs(), _priors(), sampler="nope")


def test_stream_project_priors_flag_changes_input(stream):
    """project_priors should change the result for a non-div-free prior."""
    rng = np.random.default_rng(2)
    raw_prior = [rng.normal(0, 0.15, size=(44, 94, 2)).astype(np.float32)
                 for _ in range(2)]
    m_on, _ = stream.predict(_obs(), raw_prior, sampler="ddpm",
                             inference_steps=10, seed=0, project_priors=True)
    m_off, _ = stream.predict(_obs(), raw_prior, sampler="ddpm",
                              inference_steps=10, seed=0, project_priors=False)
    assert not np.allclose(m_on, m_off)   # projection actually did something
