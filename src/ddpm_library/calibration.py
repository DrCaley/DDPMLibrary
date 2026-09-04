"""Shared conformal-calibration helpers.

Added 2026-09-02 after an audit found that every predictor's default
``uncertainty`` under-covered its stated level, while a correctly fitted factor
sat unused in :mod:`ddpm_library.config`:

    CorrDiff  0.834 coverage at the 90% target  (calibrated, but with the
                                                 simultaneous-observation factor)
    DistAttn  0.737                             (raw ensemble spread)
    Stream    0.489                             (raw ensemble spread)

The conformal factor depends on how the observations were collected -- a field
sampled over two hours has moved under the vehicle, so the model's own spread
understates the error by more than it does for a simultaneous snapshot. The
observation tuples already carry timestamps, so the factor can be chosen from
the data rather than left to the caller.
"""

import warnings
from typing import Iterable, Optional, Sequence

import numpy as np

#: The conformal factors are fitted at each model's DEFAULT ensemble size and do
#: not transfer across ensemble sizes. Measured on CorrDiff over the 40 benchmark
#: cases, the factor needed for 90% coverage runs 5.053 / 4.116 / 3.805 / 3.688 at
#: n_draws = 5 / 10 / 20 / 40 -- so the n=20 factor applied at n=5 under-covers by
#: a third. See `benchmark/corrdiff_noise_and_draws.py`.

#: Observation spans below this (hours) are treated as a simultaneous snapshot.
SIMULTANEOUS_SPAN_H = 0.05


def obs_span_hours(observations: Iterable[Sequence[float]]) -> float:
    """Wall-clock span of an observation set, in hours.

    Observations are ``(lat, lon, unix_t, u, v)``; index 2 is the timestamp.
    Returns 0.0 when there are fewer than two observations or no usable times.
    """
    ts = []
    for o in observations:
        try:
            ts.append(float(o[2]))
        except (IndexError, TypeError, ValueError):
            continue
    if len(ts) < 2:
        return 0.0
    span = (max(ts) - min(ts)) / 3600.0
    return float(span) if np.isfinite(span) and span > 0 else 0.0


def resolve_sigma_scale(
    observations: Iterable[Sequence[float]],
    *,
    timed: float,
    simultaneous: Optional[float] = None,
    override: Optional[float] = None,
    n_draws: Optional[int] = None,
    fitted_n_draws: Optional[int] = None,
    stride: Optional[int] = None,
    fitted_stride: Optional[int] = None,
    model: str = "this model",
) -> tuple[float, str]:
    """Pick the conformal factor for this observation set.

    Parameters
    ----------
    timed, simultaneous
        Factors fitted for time-spread and for simultaneous collection.
        ``simultaneous`` may be None when only the timed factor was fitted, in
        which case the timed factor is used throughout and the reason says so.
    override
        Caller-supplied factor; used verbatim when not None.
    n_draws, fitted_n_draws, stride, fitted_stride, model
        Sampling settings of this call, and the settings the factor was fitted
        at. Both change the raw ensemble spread, so when either differs the
        calibrated interval will not hold its stated coverage and a
        ``RuntimeWarning`` names the model and the mismatch. Pass
        ``sigma_scale=`` to silence it once the factor has been refit.

    Returns
    -------
    (factor, reason) -- ``reason`` is a short string for diagnostics and warnings.
    """
    if override is not None:
        if not override > 0:
            raise ValueError(f"sigma_scale must be > 0; got {override}")
        return float(override), "caller override"
    off = [(nm, used, fit) for nm, used, fit in
           (("n_draws", n_draws, fitted_n_draws), ("stride", stride, fitted_stride))
           if used is not None and fit is not None and int(used) != int(fit)]
    if off:
        detail = "; ".join(f"{nm}={used} but the factor was fitted at {nm}={fit}"
                           for nm, used, fit in off)
        warnings.warn(
            f"{model}: {detail}. Both settings change the raw ensemble spread, so "
            f"the calibrated interval will not hold its stated coverage. Refit the "
            f"factor for these settings, or pass sigma_scale= explicitly.",
            RuntimeWarning, stacklevel=3,
        )
    span = obs_span_hours(observations)
    if span <= SIMULTANEOUS_SPAN_H:
        if simultaneous is not None:
            return float(simultaneous), f"simultaneous (span {span:.3f} h)"
        return float(timed), (
            f"span {span:.3f} h looks simultaneous, but only a time-spread factor "
            f"was fitted for this model; using it")
    return float(timed), f"time-spread (span {span:.2f} h)"
