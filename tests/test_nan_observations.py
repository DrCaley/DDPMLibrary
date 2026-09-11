"""Non-finite readings must be dropped loudly, never silently propagated.

Real mission data drops samples: the first AUV track a collaborator sent had NaN
in 22 of 200 readings. With no guard, a single NaN scatters into the conditioning
and every model returns an all-NaN field with no error at all.
"""
import warnings

import numpy as np
import pytest

from ddpm_library.rasterize import observations_to_channels

LAT, LON = 18.30, -64.71
GOOD = [(LAT, LON, 0.0, 0.1, 0.05),
        (LAT + 0.002, LON + 0.002, 30.0, 0.2, -0.05),
        (LAT + 0.004, LON + 0.004, 60.0, -0.1, 0.15)]


def _channels(obs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = observations_to_channels(obs)
    return out, w


def test_clean_observations_are_untouched():
    (u, v, miss), w = _channels(GOOD)
    assert not any(issubclass(x.category, RuntimeWarning) for x in w)
    assert np.isfinite(u).all() and np.isfinite(v).all()
    assert int((miss < 0.5).sum()) == 3


@pytest.mark.parametrize("bad", [
    (LAT, LON, 90.0, float("nan"), 0.1),          # NaN u
    (LAT, LON, 90.0, 0.1, float("nan")),          # NaN v
    (float("nan"), LON, 90.0, 0.1, 0.1),          # NaN lat
    (LAT, LON, 90.0, float("inf"), 0.1),          # inf u
])
def test_one_bad_reading_is_dropped_and_warned(bad):
    (u, v, miss), w = _channels(GOOD + [bad])
    assert np.isfinite(u).all() and np.isfinite(v).all(), "NaN reached the channels"
    assert int((miss < 0.5).sum()) == 3, "a good observation was lost"
    assert any("non-finite" in str(x.message) for x in w), "dropped without warning"


def test_a_nan_timestamp_is_kept_here_because_it_is_unused():
    # the gridded rasterizer ignores the timestamp; DistAttn checks it separately
    (u, v, miss), _ = _channels(GOOD + [(LAT, LON, float("nan"), 0.1, 0.1)])
    assert np.isfinite(u).all() and int((miss < 0.5).sum()) == 3


def test_all_bad_raises_rather_than_returning_an_empty_field():
    with pytest.raises(ValueError, match="nothing to condition on"):
        observations_to_channels([(LAT, LON, 0.0, float("nan"), float("nan"))])


def test_empty_input_behaviour_is_unchanged():
    u, v, miss = observations_to_channels([])
    assert (miss > 0.5).all() and not u.any() and not v.any()
