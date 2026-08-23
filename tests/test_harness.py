"""Pins the evaluation harness helpers in scripts/_harness.py.

The track generator consumes its RNG in a fixed order, and every published number
in this project depends on that order. If a refactor changes it, every track
re-rolls and past results stop reproducing -- silently, because the new numbers
still look plausible. These tests fail loudly instead.

Deliberately data-free: they use a synthetic mask so they run anywhere, without
the 540 MB dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _harness import (LAGS, SEC_PER_CELL, bootstrap_ci, common_ocean_mask,  # noqa: E402
                      interp_at, lookback_ok, make_track, track_ages)


def _mask(h=44, w=94, land_frac=0.1):
    m = np.ones((h, w), bool)
    m[:4, :4] = False                      # a land corner, deterministic
    m[20:24, 40:50] = False                # an island the walk must route around
    return m


# --------------------------------------------------------------------------- #
# Reproducibility -- the whole point of this file
# --------------------------------------------------------------------------- #
def test_make_track_is_pinned():
    """A fixed seed must give a fixed track, forever."""
    cells = make_track(_mask(), 90, np.random.default_rng(0))
    assert len(cells) == 90
    assert len(set(cells)) == 90                     # distinct cells
    # pinned values: changing these means past results no longer reproduce
    assert cells[0] == (37, 48)
    assert cells[-1] == (43, 44)
    assert sum(r + c for r, c in cells) == 8189


def test_make_track_respects_the_mask():
    m = _mask()
    for _ in range(5):
        for r, c in make_track(m, 60, np.random.default_rng(_)):
            assert m[r, c], f"track entered a masked cell at {(r, c)}"


def test_first_visit_sequence_is_not_the_path():
    """Consecutive FIRST VISITS need not be adjacent, because the walk keeps
    moving over cells it has already seen. This is the trap behind the age
    accounting: cell k was reached at step s_k > k, so ages taken from list
    position understate staleness."""
    cells, steps = make_track(_mask(), 90, np.random.default_rng(3),
                              return_steps=True)
    assert steps[0] == 0 and steps == sorted(steps)
    assert steps[-1] > len(cells), (
        "this walk should take more steps than it collects cells; if it does "
        "not, the positional age shortcut would be safe and this test is stale")
    jumps = sum(1 for a, b in zip(cells, cells[1:])
                if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1)
    assert jumps > 0                       # documents the behaviour, not a bug


def test_track_ages_from_steps_exceed_positional_ages():
    """The correct ages are strictly longer than the positional shortcut."""
    cells, steps = make_track(_mask(), 90, np.random.default_rng(1),
                              return_steps=True)
    assert track_ages(len(cells), steps).max() > track_ages(len(cells)).max()


def test_make_track_rejects_impossible_request():
    tiny = np.zeros((44, 94), bool)
    tiny[:2, :2] = True
    with pytest.raises(ValueError, match="usable cells"):
        make_track(tiny, 90, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def test_track_ages():
    """Last cell fresh, first cell oldest; a 90-cell run spans about 1.2 h."""
    a = track_ages(90)
    assert a[-1] == 0.0
    assert a[0] == pytest.approx(89 * SEC_PER_CELL / 3600.0)
    assert 1.1 < a[0] < 1.3
    assert np.all(np.diff(a) < 0)                     # monotone, freshest last


def test_interp_at_never_reads_the_future():
    """An integer index must return that frame, not blend with the next one --
    blending would pull a later frame into the observations."""
    arr = np.stack([np.full((4, 4), float(i)) for i in range(5)])
    assert np.all(interp_at(arr, 3.0) == 3.0)
    assert np.all(interp_at(arr, 2.5) == 2.5)         # genuine midpoint
    assert np.all(interp_at(arr, 2.25) == 2.25)


def test_lookback_ok_blocks_segment_crossing():
    assert lookback_ok(t=100, oldest_hours=2.0, block=336)
    assert not lookback_ok(t=337, oldest_hours=2.0, block=336)   # crosses a boundary
    assert not lookback_ok(t=1, oldest_hours=2.0, block=336)     # before the start


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def test_common_ocean_mask_is_an_intersection():
    class M:
        def __init__(self, mask): self.ocean_mask = mask.astype(np.float32)
    a = np.ones((44, 94), bool); a[0, 0] = False
    b = np.ones((44, 94), bool); b[1, 1] = False
    got = common_ocean_mask([M(a), M(b)])
    assert not got[0, 0] and not got[1, 1]
    assert got.sum() == 44 * 94 - 2


def test_common_ocean_mask_ignores_models_without_one():
    class NoMask: pass
    assert common_ocean_mask([NoMask()]).all()


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    x = np.random.default_rng(0).normal(0.5, 0.1, 40)
    mu, lo, hi = bootstrap_ci(x)
    assert lo < mu < hi
    assert (mu, lo, hi) == bootstrap_ci(x)            # seeded, so repeatable


def test_bootstrap_ci_handles_empty_and_nan():
    assert all(np.isnan(v) for v in bootstrap_ci([]))
    mu, lo, hi = bootstrap_ci([1.0, np.nan, 1.0])
    assert mu == pytest.approx(1.0)


def test_lags_match_the_models():
    from ddpm_library.config import CORRDIFF_LAGS, REPAINT_LAGS
    assert LAGS == CORRDIFF_LAGS == REPAINT_LAGS
