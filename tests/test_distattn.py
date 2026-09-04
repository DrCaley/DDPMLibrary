"""Tests for the distance/time-aware attention predictor.

These need the bundled checkpoint (git-lfs); they skip cleanly if absent.
Sampling uses a large ``stride`` so the suite stays fast -- these check the
wiring, the token contract and the age conditioning, not accuracy. For accuracy
see ``scripts/validate_distattn.py``.
"""

import numpy as np
import pytest

from ddpm_library.config import (
    DISTATTN_H, DISTATTN_OCEAN_MASK_PATH, DISTATTN_SIGMA_SCALE_TIMED,
    DISTATTN_T, DISTATTN_W,
    DISTATTN_WEIGHTS_PATH, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, OCEAN_H, OCEAN_W,
)

_HAVE = DISTATTN_WEIGHTS_PATH.exists() and DISTATTN_OCEAN_MASK_PATH.exists()
_needs_assets = pytest.mark.skipif(
    not _HAVE, reason="DistAttn assets not present (git-lfs pull)")

_STRIDE, _DRAWS = 250, 2          # 4 network calls per draw


@pytest.fixture(scope="module")
def model():
    from ddpm_library import DistAttn
    return DistAttn(device="cpu")


@pytest.fixture(scope="module")
def obs(model):
    """A 90-minute transect over cells this model calls ocean."""
    ocean = model.ocean_mask > 0.5                       # (44, 94) library
    lats = np.linspace(LAT_MIN, LAT_MAX, OCEAN_H)
    lons = np.linspace(LON_MIN, LON_MAX, OCEAN_W)
    rng = np.random.default_rng(0)
    cells = np.argwhere(ocean)
    picked = cells[rng.choice(len(cells), 60, replace=False)]
    return [(float(lats[i]), float(lons[j]), 1.7e9 + k * 60.0,
             float(rng.uniform(-0.2, 0.2)), float(rng.uniform(-0.2, 0.2)))
            for k, (i, j) in enumerate(picked)]


# --------------------------------------------------------------------------- #
# Diffusion schedule -- must reproduce the collaborator's exactly
# --------------------------------------------------------------------------- #
def test_schedule_matches_reference():
    """betas = linspace(1e-4, 0.02, T); alpha_bar = cumprod(1 - betas)."""
    import torch
    from ddpm_library.distattn import DDPM
    d = DDPM(T=DISTATTN_T, noise_std=0.5)
    betas = torch.linspace(1e-4, 0.02, DISTATTN_T)
    assert torch.equal(d.betas, betas)
    assert torch.equal(d.alpha_bar, torch.cumprod(1.0 - betas, dim=0))
    assert d.noise_std == 0.5          # NOT 1.0: this pipeline is in physical m/s


# --------------------------------------------------------------------------- #
# Token construction -- the part unique to this model
# --------------------------------------------------------------------------- #
@_needs_assets
def test_token_layout_and_ranges(model, obs):
    tok = model._build_tokens(obs)[0].cpu().numpy()
    assert tok.shape == (len(obs), 5)
    x, y, u, v, age = tok.T
    assert np.all((x >= 0) & (x <= 1)) and np.all((y >= 0) & (y <= 1))
    # u, v are passed through in physical m/s, unscaled
    assert np.allclose(sorted(u), sorted(o[3] for o in obs), atol=1e-6)
    # ages are hours relative to the NEWEST observation, so min is exactly 0
    assert age.min() == pytest.approx(0.0)
    assert age.max() == pytest.approx((max(o[2] for o in obs)
                                       - min(o[2] for o in obs)) / 3600.0, rel=1e-5)


@_needs_assets
def test_token_coordinates_match_the_training_convention(model):
    """x_norm indexes the 44-wide axis, y_norm the 94-wide axis.

    Getting this backwards still produces a plausible field, so it is pinned
    against the grid corners rather than left to inspection.
    """
    lats = np.linspace(LAT_MIN, LAT_MAX, OCEAN_H)
    lons = np.linspace(LON_MIN, LON_MAX, OCEAN_W)
    ocean = model.ocean_mask > 0.5
    i0, j0 = map(int, np.argwhere(ocean)[0])
    tok = model._build_tokens([(float(lats[i0]), float(lons[j0]), 1.7e9, 0.1, 0.0)])
    x, y = float(tok[0, 0, 0]), float(tok[0, 0, 1])
    assert x == pytest.approx(i0 / (DISTATTN_W - 1))     # lat index over 43
    assert y == pytest.approx(j0 / (DISTATTN_H - 1))     # lon index over 93


@_needs_assets
def test_absolute_epoch_is_irrelevant(model, obs):
    """Only the SPACING of timestamps matters, since ages are relative."""
    shifted = [(la, lo, t + 86_400.0, u, v) for (la, lo, t, u, v) in obs]
    a = model._build_tokens(obs)[0].cpu().numpy()
    b = model._build_tokens(shifted)[0].cpu().numpy()
    assert np.allclose(a, b)


@_needs_assets
def test_stale_observations_change_the_prediction(model, obs):
    """Guards the age channel: identical readings, different ages, different field."""
    fresh = [(la, lo, 1.7e9, u, v) for (la, lo, _, u, v) in obs]        # all age 0
    stale = [(la, lo, 1.7e9 - k * 120.0, u, v)                          # spread over 2 h
             for k, (la, lo, _, u, v) in enumerate(obs)]
    a, _ = model.predict(fresh, n_draws=1, stride=_STRIDE, seed=0)
    b, _ = model.predict(stale, n_draws=1, stride=_STRIDE, seed=0)
    assert not np.allclose(a, b)


@_needs_assets
def test_out_of_range_age_warns(model, obs):
    old = [(la, lo, 1.7e9 - k * 900.0, u, v)            # spans ~15 h, trained on <= 3 h
           for k, (la, lo, _, u, v) in enumerate(obs)]
    with pytest.warns(UserWarning, match="beyond the"):
        model._build_tokens(old)


@_needs_assets
def test_land_observations_are_dropped_with_a_warning(model):
    lats = np.linspace(LAT_MIN, LAT_MAX, OCEAN_H)
    lons = np.linspace(LON_MIN, LON_MAX, OCEAN_W)
    land = np.argwhere(model.ocean_mask <= 0.5)
    ocean = np.argwhere(model.ocean_mask > 0.5)
    mixed = ([(float(lats[i]), float(lons[j]), 1.7e9, 0.1, 0.0) for i, j in land[:3]]
             + [(float(lats[i]), float(lons[j]), 1.7e9, 0.1, 0.0) for i, j in ocean[:5]])
    with pytest.warns(UserWarning, match="snapped onto land"):
        tok = model._build_tokens(mixed)
    assert tok.shape[1] == 5          # the 3 land observations were dropped


@_needs_assets
def test_all_land_observations_rejected(model):
    lats = np.linspace(LAT_MIN, LAT_MAX, OCEAN_H)
    lons = np.linspace(LON_MIN, LON_MAX, OCEAN_W)
    land = np.argwhere(model.ocean_mask <= 0.5)[:4]
    allland = [(float(lats[i]), float(lons[j]), 1.7e9, 0.1, 0.0) for i, j in land]
    with pytest.raises(ValueError, match="ocean cell"):
        with pytest.warns(UserWarning):
            model._build_tokens(allland)


# --------------------------------------------------------------------------- #
# Output contract -- shared with every other predictor
# --------------------------------------------------------------------------- #
@_needs_assets
def test_output_contract(model, obs):
    mean, unc = model.predict(obs, n_draws=_DRAWS, stride=_STRIDE, seed=0)
    ocean = model.ocean_mask > 0.5
    for a in (mean, unc):
        assert a.shape == (OCEAN_H, OCEAN_W, 2) and a.dtype == np.float32
        assert np.all(np.isfinite(a))
        assert np.all(a[~ocean] == 0)
    assert unc[ocean].max() > 0
    assert model.takes_priors is False


@_needs_assets
def test_single_draw_has_zero_uncertainty(model, obs):
    _, unc = model.predict(obs, n_draws=1, stride=_STRIDE, seed=0)
    assert np.all(unc == 0)


@_needs_assets
def test_seeding(model, obs):
    a, _ = model.predict(obs, n_draws=_DRAWS, stride=_STRIDE, seed=3)
    b, _ = model.predict(obs, n_draws=_DRAWS, stride=_STRIDE, seed=3)
    c, _ = model.predict(obs, n_draws=_DRAWS, stride=_STRIDE, seed=4)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


@_needs_assets
def test_predict_takes_no_priors(model, obs):
    """The signature encodes that this model is conditioned only by tokens."""
    with pytest.raises(TypeError):
        model.predict(obs, [np.zeros((OCEAN_H, OCEAN_W, 2), np.float32)] * 2)


@_needs_assets
@pytest.mark.parametrize("kwargs", [{"n_draws": 0}, {"stride": 0},
                                    {"stride": DISTATTN_T + 1}])
def test_invalid_arguments(model, obs, kwargs):
    base = {"n_draws": 1, "stride": _STRIDE}
    base.update(kwargs)
    with pytest.raises(ValueError):
        model.predict(obs, **base)


@_needs_assets
def test_empty_observations_rejected(model):
    with pytest.raises(ValueError, match="(?i)at least one observation"):
        model.predict([], stride=_STRIDE)


@_needs_assets
def test_out_of_bounds_observation_rejected(model):
    with pytest.raises(ValueError, match="outside the model"):
        model.predict([(0.0, 0.0, 1.7e9, 0.1, 0.0)], stride=_STRIDE)


@_needs_assets
def test_mask_is_a_subset_of_the_shared_grid(model):
    """This model's mask is stricter than the shared one; it must not claim
    ocean where the shared grid says land, or the common-mask intersection in
    scripts/compare_models.py would be meaningless."""
    from ddpm_library.config import CORRDIFF_GRID_PATH
    shared = ~np.asarray(np.load(CORRDIFF_GRID_PATH)["land_mask"]).astype(bool)
    assert model.ocean_np.shape == (DISTATTN_H, DISTATTN_W) == shared.shape
    assert not (model.ocean_np & ~shared).any()


@_needs_assets
def test_distattn_calibration_scales_raw_spread_by_the_fitted_factor(model, obs):
    """calibrate=True must be exactly the raw spread times the fitted factor.

    DistAttn flipped from returning RAW spread to calibrated on 2026-09-04, and its
    old docstring instructed callers to multiply by 1.621 themselves -- so anyone
    following it would now apply the factor twice.
    """
    _, raw = model.predict(obs, n_draws=_DRAWS, stride=_STRIDE, seed=11,
                           calibrate=False)
    _, cal = model.predict(obs, n_draws=_DRAWS, stride=_STRIDE, seed=11,
                           calibrate=True)
    om = model.ocean_mask > 0.5
    np.testing.assert_allclose(cal[om], raw[om] * DISTATTN_SIGMA_SCALE_TIMED, rtol=1e-5)


@_needs_assets
def test_distattn_sigma_scale_override_beats_the_fitted_factor(model, obs):
    _, raw = model.predict(obs, n_draws=_DRAWS, stride=_STRIDE, seed=11,
                           calibrate=False)
    _, cal = model.predict(obs, n_draws=_DRAWS, stride=_STRIDE, seed=11,
                           calibrate=True, sigma_scale=3.0)
    om = model.ocean_mask > 0.5
    np.testing.assert_allclose(cal[om], raw[om] * 3.0, rtol=1e-5)
