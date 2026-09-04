"""The conformal factor is fitted at one sampling configuration and does not transfer.

Measured on CorrDiff over the 40 benchmark cases, the factor needed for 90%
coverage runs 5.053 / 4.116 / 3.805 / 3.688 at n_draws = 5 / 10 / 20 / 40, so a
factor fitted at 20 and used at 5 under-covers by about a third. These tests lock
in the warning that says so -- see docs/DEFAULTS_AND_DIALS.md.
"""
import warnings

import pytest

from ddpm_library.calibration import resolve_sigma_scale

OBS = [(46.0, -124.0, 0.0, 0.1, 0.2), (46.1, -124.1, 10.0, 0.1, 0.2)]


def _resolve(**kw):
    return resolve_sigma_scale(OBS, timed=2.0, simultaneous=1.5,
                               n_draws=kw.pop("n_draws", None),
                               fitted_n_draws=kw.pop("fitted_n_draws", None),
                               stride=kw.pop("stride", None),
                               fitted_stride=kw.pop("fitted_stride", None),
                               model="TestModel", **kw)


def test_warns_when_ensemble_size_differs_from_the_fitted_one():
    with pytest.warns(RuntimeWarning, match=r"n_draws=5 but the factor was fitted at n_draws=20"):
        _resolve(n_draws=5, fitted_n_draws=20)


def test_silent_at_the_fitted_ensemble_size():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _resolve(n_draws=20, fitted_n_draws=20)


def test_silent_when_the_sizes_are_unknown():
    """Callers that pass neither size keep the old behaviour."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _resolve()


def test_explicit_override_skips_the_check():
    """A caller who passes sigma_scale= has taken responsibility for the factor."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factor, why = _resolve(n_draws=5, fitted_n_draws=20, override=3.0)
    assert factor == 3.0 and why == "caller override"


def test_warns_when_stride_differs_from_the_fitted_one():
    """RePaint's factor is fitted at stride 5; stride changes the spread too."""
    with pytest.warns(RuntimeWarning, match=r"stride=1 but the factor was fitted at stride=5"):
        _resolve(stride=1, fitted_stride=5)


def test_one_warning_lists_every_mismatch():
    with pytest.warns(RuntimeWarning) as rec:
        _resolve(n_draws=5, fitted_n_draws=10, stride=1, fitted_stride=5)
    assert len(rec) == 1, "mismatches should be reported together, not one warning each"
    msg = str(rec[0].message)
    assert "n_draws=5" in msg and "stride=1" in msg


def test_silent_when_both_settings_match():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _resolve(n_draws=10, fitted_n_draws=10, stride=5, fitted_stride=5)
