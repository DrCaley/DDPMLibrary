"""Metric properties that are easy to get wrong and expensive to get wrong.

Not smoke tests -- these pin behaviour that a plausible-looking refactor would
silently break.
"""

import numpy as np

from ddpm_library import metrics


def _vortex_field(H=44, W=94, amp=0.05):
    """A field with real rotational structure: three Gaussian vortices."""
    yy, xx = np.mgrid[0:H, 0:W]
    f = np.zeros((H, W, 2), np.float32)
    for cy, cx, s in ((12, 20, 1.0), (30, 55, -1.0), (20, 78, 1.0)):
        env = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 40.0)
        f[..., 0] += (-(xx - cx) * env * s * amp).astype(np.float32)
        f[..., 1] += ((yy - cy) * env * s * amp).astype(np.float32)
    return f


def _curl(f):
    return np.gradient(f[..., 1], axis=1) - np.gradient(f[..., 0], axis=0)


def _gradient_field(H=44, W=94, amp=0.02):
    """A pure gradient (curl-free) field."""
    yy, xx = np.mgrid[0:H, 0:W]
    return np.stack([(xx - W / 2) * amp, (yy - H / 2) * amp], -1).astype(np.float32)


def test_curl_free_field_does_not_change_vorticity():
    """The premise behind the eddy-metric bias, as a standalone check.

    The curl of a gradient is identically zero, so adding a pure gradient field
    leaves vorticity untouched. Okubo-Weiss is strain^2 - vorticity^2, so it still
    reacts -- it reads the added strain as "less rotation". That is the bias
    documented in docs/EDDY_METRIC_BIAS.md.
    """
    f = _vortex_field()
    interior = (slice(2, -2), slice(2, -2))
    np.testing.assert_allclose(_curl(f)[interior],
                               _curl(f + _gradient_field())[interior], atol=1e-6)


def test_eddy_hit_rate_rotational_is_invariant_to_divergence():
    """rotational=True must not react to a curl-free field.

    It Helmholtz-projects both fields before detecting, so the added divergence is
    removed from each side and cannot shift the strain term. The raw metric has no
    such protection -- on real benchmark fields a gradient of amplitude 0.02 takes
    it from 0.458 to 0.042 while the corrected metric holds near 0.41 (see
    docs/EDDY_METRIC_BIAS.md). That effect size is data-dependent, so it is
    measured there rather than asserted here; this pins the invariant, which is
    mathematical and must always hold.
    """
    rng = np.random.default_rng(0)
    ocean = np.ones((44, 94), bool)
    truth = _vortex_field()
    pred = truth + rng.normal(0, 0.01, truth.shape).astype(np.float32)

    before = metrics.eddy_hit_rate(pred, truth, ocean, rotational=True)
    after = metrics.eddy_hit_rate(pred + _gradient_field(), truth, ocean,
                                  rotational=True)
    assert abs(after - before) < 0.05, (
        f"rotational=True must be invariant to a curl-free field "
        f"({before:.3f} -> {after:.3f})")


def test_eddy_hit_rate_nan_when_truth_has_no_eddies():
    """NaN, not zero -- so frames with nothing to find can be excluded rather than
    silently dragging a mean toward zero. A uniform flow has no rotation."""
    ocean = np.ones((44, 94), bool)
    uniform = np.zeros((44, 94, 2), np.float32)
    uniform[..., 0] = 0.1
    assert np.isnan(metrics.eddy_hit_rate(uniform, uniform, ocean))


def test_rmse_is_per_component_not_vector_magnitude():
    """metrics.rmse averages over cells AND components.

    The vector-magnitude convention is larger by exactly sqrt(2), and confusing
    the two cost this project a round of irreconcilable numbers with a
    collaborator. Pinned so a refactor cannot quietly switch conventions.
    """
    ocean = np.ones((44, 94), bool)
    truth = np.zeros((44, 94, 2), np.float32)
    pred = np.ones((44, 94, 2), np.float32)          # error of 1.0 in each component
    per_component = metrics.rmse(pred, truth, ocean)
    vector_magnitude = float(np.sqrt(((pred - truth) ** 2).sum(-1)[ocean].mean()))
    assert np.isclose(per_component, 1.0)
    assert np.isclose(vector_magnitude / per_component, np.sqrt(2))
