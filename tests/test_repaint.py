"""Tests for the two collaborator RePaint predictors.

These need the bundled checkpoints (git-lfs); they skip cleanly if absent.
Sampling is deliberately coarse (large ``stride``, 2 draws) -- these check the
wiring and the guardrails, not reconstruction accuracy. For accuracy see
``scripts/validate_repaint.py``, which reproduces the published number.
"""

import warnings

import numpy as np
import pytest

from ddpm_library.config import (
    CORRDIFF_GRID_PATH, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, OCEAN_H, OCEAN_W,
    REPAINT_LAGS, REPAINT_UNCOND_WEIGHTS_PATH, REPAINT_WEIGHTS_PATH,
)

_HAVE = (REPAINT_WEIGHTS_PATH.exists() and REPAINT_UNCOND_WEIGHTS_PATH.exists()
         and CORRDIFF_GRID_PATH.exists())
_needs_assets = pytest.mark.skipif(
    not _HAVE, reason="RePaint assets not present (git-lfs pull)")

# Coarse settings: the reverse chain is 1000 steps and guided, so a full-stride
# run here would dominate the suite's runtime.
_STRIDE, _DRAWS = 500, 2


@pytest.fixture(scope="module")
def timecond():
    from ddpm_library import RePaint
    return RePaint(device="cpu")


@pytest.fixture(scope="module")
def uncond():
    from ddpm_library import RePaintUncond
    return RePaintUncond(device="cpu")


@pytest.fixture(scope="module")
def obs():
    """Observations spread over the covered region (which is small: ~0.02 deg)."""
    rng = np.random.default_rng(0)
    return [(rng.uniform(LAT_MIN, LAT_MAX), rng.uniform(LON_MIN, LON_MAX),
             1_700_000_000.0, rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2))
            for _ in range(40)]


@pytest.fixture
def priors():
    return [np.zeros((OCEAN_H, OCEAN_W, 2), np.float32) for _ in REPAINT_LAGS]


# --------------------------------------------------------------------------- #
# Checkpoint identity -- the point of having two classes
# --------------------------------------------------------------------------- #
@_needs_assets
def test_each_class_loads_its_own_checkpoint(timecond, uncond):
    assert timecond.conditional and timecond.cond_ch == 2 * len(timecond.lags)
    assert timecond.lags == REPAINT_LAGS
    assert not uncond.conditional and uncond.cond_ch == 0 and uncond.lags == ()
    assert timecond.takes_priors and not uncond.takes_priors


@_needs_assets
def test_sibling_checkpoint_is_rejected():
    """A swapped path must fail loudly, not silently run the other model."""
    from ddpm_library import RePaint, RePaintUncond
    with pytest.raises(ValueError, match="only loads time-conditioned"):
        RePaint(device="cpu", weights_path=REPAINT_UNCOND_WEIGHTS_PATH)
    with pytest.raises(ValueError, match="only loads unconditional"):
        RePaintUncond(device="cpu", weights_path=REPAINT_WEIGHTS_PATH)


@_needs_assets
def test_missing_weights_raise():
    from ddpm_library import RePaint
    with pytest.raises(FileNotFoundError):
        RePaint(device="cpu", weights_path="/nonexistent/repaint.pt")


# --------------------------------------------------------------------------- #
# Output contract -- shared with every other predictor
# --------------------------------------------------------------------------- #
@_needs_assets
@pytest.mark.parametrize("which", ["timecond", "uncond"])
def test_output_contract(which, timecond, uncond, obs, priors):
    mdl = timecond if which == "timecond" else uncond
    args = (obs, priors) if mdl.takes_priors else (obs,)
    mean, unc = mdl.predict(*args, n_draws=_DRAWS, stride=_STRIDE, seed=0)
    ocean = mdl.ocean_mask > 0.5
    for a in (mean, unc):
        assert a.shape == (OCEAN_H, OCEAN_W, 2) and a.dtype == np.float32
        assert np.all(np.isfinite(a))
        assert np.all(a[~ocean] == 0)          # land must be exactly zero
    assert unc[ocean].max() > 0                # >1 draw gives real spread


@_needs_assets
def test_single_draw_has_zero_uncertainty(uncond, obs):
    _, unc = uncond.predict(obs, n_draws=1, stride=_STRIDE, seed=0)
    assert np.all(unc == 0)


@_needs_assets
def test_seeding(uncond, obs):
    a, _ = uncond.predict(obs, n_draws=_DRAWS, stride=_STRIDE, seed=3)
    b, _ = uncond.predict(obs, n_draws=_DRAWS, stride=_STRIDE, seed=3)
    c, _ = uncond.predict(obs, n_draws=_DRAWS, stride=_STRIDE, seed=4)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


# --------------------------------------------------------------------------- #
# Priors: accepted, validated, and genuinely used by the conditioned model only
# --------------------------------------------------------------------------- #
@_needs_assets
def test_uncond_predict_takes_no_priors(uncond, obs, priors):
    """The signature itself encodes that this model has no prior channels."""
    with pytest.raises(TypeError):
        uncond.predict(obs, priors)


@_needs_assets
def test_missing_priors_warns_but_works(timecond, obs):
    with pytest.warns(UserWarning, match="without `priors`"):
        mean, _ = timecond.predict(obs, n_draws=1, stride=_STRIDE, seed=0)
    assert mean.shape == (OCEAN_H, OCEAN_W, 2)


@_needs_assets
def test_priors_change_the_prediction(timecond, obs, priors):
    """Guards the conditioning path: priors must actually reach the network."""
    zero, _ = timecond.predict(obs, priors, n_draws=1, stride=_STRIDE, seed=0)
    rng = np.random.default_rng(1)
    strong = [rng.normal(0, 0.2, (OCEAN_H, OCEAN_W, 2)).astype(np.float32)
              for _ in REPAINT_LAGS]
    other, _ = timecond.predict(obs, strong, n_draws=1, stride=_STRIDE, seed=0)
    assert not np.allclose(zero, other)


@_needs_assets
@pytest.mark.parametrize("bad", [
    [np.zeros((OCEAN_H, OCEAN_W, 2), np.float32)],           # too few
    [np.zeros((5, 5, 2), np.float32)] * 2,                   # wrong shape
])
def test_bad_priors_rejected(timecond, obs, bad):
    with pytest.raises(ValueError):
        timecond.predict(obs, bad, n_draws=1, stride=_STRIDE)


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #
@_needs_assets
@pytest.mark.parametrize("kwargs", [{"n_draws": 0}, {"sampler": "nope"}])
def test_invalid_arguments(uncond, obs, kwargs):
    with pytest.raises(ValueError):
        uncond.predict(obs, stride=_STRIDE, **kwargs)


@_needs_assets
def test_empty_observations_rejected(uncond):
    with pytest.raises(ValueError, match="(?i)at least one observation"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uncond.predict([], stride=_STRIDE)


@_needs_assets
def test_both_samplers_run(timecond, obs, priors):
    for sampler in ("dps", "mcg"):
        mean, _ = timecond.predict(obs, priors, n_draws=1, stride=_STRIDE,
                                   sampler=sampler, seed=0)
        assert np.all(np.isfinite(mean))
