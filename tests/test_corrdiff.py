"""Tests for the CorrDiff predictor and the shared metrics suite.

The model tests need the bundled weights (git-lfs); they skip cleanly if the
assets are absent. The metrics tests are pure-numpy and always run.
"""

import numpy as np
import pytest

from ddpm_library import metrics
from ddpm_library.config import (
    CORRDIFF_GRID_PATH, CORRDIFF_NOISE_MAX, CORRDIFF_SIGMA_SCALE,
    CORRDIFF_WEIGHTS_PATH, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, OCEAN_H, OCEAN_W,
)

_HAVE_ASSETS = CORRDIFF_WEIGHTS_PATH.exists() and CORRDIFF_GRID_PATH.exists()
_needs_assets = pytest.mark.skipif(
    not _HAVE_ASSETS, reason="CorrDiff assets not present (git-lfs pull)")

# Cheap sampling settings — these tests check plumbing, not accuracy.
_STEPS, _DRAWS = 6, 4


@pytest.fixture(scope="module")
def model():
    from ddpm_library import CorrDiff
    return CorrDiff(device="cpu")


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(0)
    obs = [(rng.uniform(LAT_MIN, LAT_MAX), rng.uniform(LON_MIN, LON_MAX),
            1_700_000_000.0, rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2))
           for _ in range(60)]
    priors = [rng.normal(0, 0.1, (OCEAN_H, OCEAN_W, 2)).astype(np.float32)
              for _ in range(2)]
    return obs, priors


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
@_needs_assets
def test_channel_bookkeeping(model):
    """cond + mean + dist + sigma must add up to what the UNet expects."""
    expected = model.cond_ch + 2 + int(model.use_dist) + int(model.noise_cond)
    assert expected == model.total_cond


@_needs_assets
def test_output_contract(model, inputs):
    obs, priors = inputs
    mean, unc = model.predict(obs, priors, n_draws=1, steps=_STEPS)
    assert mean.shape == (OCEAN_H, OCEAN_W, 2)
    assert unc.shape == (OCEAN_H, OCEAN_W, 2)
    assert mean.dtype == np.float32 and unc.dtype == np.float32
    assert np.all(np.isfinite(mean))
    # n_draws=1 is the fast path: a single field, no uncertainty
    assert np.all(unc == 0)


@_needs_assets
def test_ensemble_has_uncertainty_and_land_is_zero(model, inputs):
    obs, priors = inputs
    mean, unc = model.predict(obs, priors, n_draws=_DRAWS, steps=_STEPS)
    ocean = model.ocean_mask > 0.5
    assert unc[ocean].max() > 0
    assert np.all(mean[~ocean] == 0) and np.all(unc[~ocean] == 0)


@_needs_assets
def test_reproducible(model, inputs):
    obs, priors = inputs
    a, _ = model.predict(obs, priors, n_draws=_DRAWS, steps=_STEPS, seed=3)
    b, _ = model.predict(obs, priors, n_draws=_DRAWS, steps=_STEPS, seed=3)
    c, _ = model.predict(obs, priors, n_draws=_DRAWS, steps=_STEPS, seed=4)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


@_needs_assets
def test_calibration_scaling(model, inputs):
    """calibrate=True must scale the raw spread by exactly the fitted factor."""
    obs, priors = inputs
    _, raw = model.predict(obs, priors, n_draws=_DRAWS, steps=_STEPS, calibrate=False)
    _, cal = model.predict(obs, priors, n_draws=_DRAWS, steps=_STEPS, calibrate=True)
    ocean = model.ocean_mask > 0.5
    assert np.allclose(cal[ocean], raw[ocean] * CORRDIFF_SIGMA_SCALE, rtol=1e-5)


@_needs_assets
def test_sensor_noise_dial_widens_spread(model, inputs):
    """The dial must monotonically widen the predictive distribution."""
    obs, priors = inputs
    ocean = model.ocean_mask > 0.5
    spreads = []
    for s in (0.0, CORRDIFF_NOISE_MAX / 2, CORRDIFF_NOISE_MAX):
        _, unc = model.predict(obs, priors, sensor_noise=s, n_draws=_DRAWS,
                               steps=_STEPS, seed=0)
        spreads.append(float(unc[ocean].mean()))
    assert spreads[0] < spreads[1] < spreads[2]


@_needs_assets
@pytest.mark.parametrize("kwargs,exc", [
    ({"sensor_noise": 10.0}, ValueError),        # outside the trained range
    ({"sensor_noise": -0.1}, ValueError),
    ({"n_draws": 0}, ValueError),
])
def test_invalid_arguments(model, inputs, kwargs, exc):
    obs, priors = inputs
    with pytest.raises(exc):
        model.predict(obs, priors, steps=_STEPS, **kwargs)


@_needs_assets
def test_empty_observations_rejected(model, inputs):
    _, priors = inputs
    with pytest.raises(ValueError):
        model.predict([], priors, steps=_STEPS)


@_needs_assets
def test_bad_priors_rejected(model, inputs):
    obs, priors = inputs
    with pytest.raises(ValueError):
        model.predict(obs, priors[:1], steps=_STEPS)              # wrong count
    with pytest.raises(ValueError):
        model.predict(obs, [np.zeros((5, 5, 2))] * 2, steps=_STEPS)  # wrong shape


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
    truth = np.stack([np.sin(2 * np.pi * yy / 15) * np.cos(2 * np.pi * xx / 12),
                      np.cos(2 * np.pi * yy / 18) * np.sin(2 * np.pi * xx / 9)],
                     axis=-1).astype(np.float32)
    ocean = np.ones((OCEAN_H, OCEAN_W), bool)
    ocean[:3, :3] = False
    return truth, ocean, rng


def test_perfect_prediction_scores_perfectly(synthetic):
    truth, ocean, _ = synthetic
    r = metrics.evaluate(truth, np.zeros_like(truth), truth, ocean_mask=ocean)
    assert r["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert r["angle_error"] == pytest.approx(0.0, abs=1e-5)
    assert r["crps"] == pytest.approx(0.0, abs=1e-9)
    assert r["ke_ratio_small"] == pytest.approx(1.0, abs=1e-6)
    assert r["fss_skilful_scale"] == 1.0


def test_crps_reduces_to_mae_when_deterministic(synthetic):
    """The property that lets deterministic and probabilistic models share an axis."""
    truth, ocean, rng = synthetic
    pred = truth + rng.normal(0, 0.05, truth.shape)
    mae = float(np.abs((pred - truth)[ocean]).mean())
    got = metrics.crps_gaussian(pred, np.zeros_like(pred), truth, ocean)
    assert got == pytest.approx(mae, rel=1e-9)


def test_crps_rewards_honest_uncertainty(synthetic):
    """CRPS must be minimised near the true error scale, not at zero or infinity."""
    truth, ocean, rng = synthetic
    sd_true = 0.05
    pred = truth + rng.normal(0, sd_true, truth.shape)
    scores = {s: metrics.crps_gaussian(pred, np.full_like(pred, s), truth, ocean)
              for s in (0.0, sd_true, 1.0)}
    assert scores[sd_true] < scores[0.0]        # honest beats over-confident
    assert scores[sd_true] < scores[1.0]        # honest beats vague


def test_blur_is_detected_by_small_scale_energy(synthetic):
    from scipy.ndimage import gaussian_filter
    truth, ocean, _ = synthetic
    blur = np.stack([gaussian_filter(truth[..., 0], 2.0),
                     gaussian_filter(truth[..., 1], 2.0)], axis=-1)
    assert metrics.ke_ratio_small(blur, truth, ocean) < 0.5


def test_sal_structure_flags_blur_on_multiscale_field():
    """S > 0 == too flat. Uses a MULTI-SCALE field on purpose: on a single
    wavenumber, blurring is only an amplitude change, and S is amplitude-invariant
    by design (see the companion test), so it correctly reports ~0 there."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    truth = np.stack([gaussian_filter(rng.normal(0, 1, (OCEAN_H, OCEAN_W)), 1.2),
                      gaussian_filter(rng.normal(0, 1, (OCEAN_H, OCEAN_W)), 1.2)],
                     axis=-1).astype(np.float32)
    ocean = np.ones((OCEAN_H, OCEAN_W), bool)
    blur = np.stack([gaussian_filter(truth[..., 0], 2.0),
                     gaussian_filter(truth[..., 1], 2.0)], axis=-1)
    assert metrics.sal_structure(blur, truth, ocean) > 0.05


def test_sal_structure_is_amplitude_invariant(synthetic):
    """Scaling every vector must not move S -- magnitude bias is ke_ratio_small's job."""
    truth, ocean, _ = synthetic
    assert metrics.sal_structure(truth * 0.5, truth, ocean) == pytest.approx(0.0, abs=1e-6)


def test_fss_tolerates_displacement(synthetic):
    """A displaced feature must score better at larger neighbourhoods."""
    from scipy.ndimage import shift
    truth, ocean, _ = synthetic
    moved = np.stack([shift(truth[..., 0], (5, 0), order=1),
                      shift(truth[..., 1], (5, 0), order=1)], axis=-1)
    f = metrics.fss(moved, truth, ocean)
    assert f["fss_41"] > f["fss_1"]


def test_spread_skill_and_coverage_are_nan_for_deterministic(synthetic):
    truth, ocean, rng = synthetic
    pred = truth + rng.normal(0, 0.05, truth.shape)
    z = np.zeros_like(pred)
    assert np.isnan(metrics.spread_skill_ratio(pred, z, truth, ocean))
    assert np.isnan(metrics.coverage(pred, z, truth, 0.9, ocean))
    assert metrics.pit_values(pred, z, truth, ocean).size == 0


def test_calibrated_gaussian_gives_nominal_coverage(synthetic):
    """Coverage must recover the nominal level when the noise really is Gaussian."""
    truth, ocean, rng = synthetic
    sd = 0.05
    pred = truth + rng.normal(0, sd, truth.shape)
    cov = metrics.coverage(pred, np.full_like(pred, sd), truth, 0.90, ocean)
    assert cov == pytest.approx(0.90, abs=0.02)
    ssr = metrics.spread_skill_ratio(pred, np.full_like(pred, sd), truth, ocean)
    assert ssr == pytest.approx(1.0, abs=0.05)


def test_evaluate_shape_mismatch_raises(synthetic):
    truth, ocean, _ = synthetic
    with pytest.raises(ValueError):
        metrics.evaluate(truth[:10], np.zeros_like(truth), truth, ocean_mask=ocean)
