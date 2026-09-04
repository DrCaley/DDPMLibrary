"""DDPMLibrary — standalone inference for ocean velocity reconstruction.

Every predictor exposes the same contract::

    mean, uncertainty = Model(device="auto").predict(observations[, priors])

with ``observations`` an iterable of ``(lat, lon, unix_t, u, v)`` and both
outputs ``(44, 94, 2)`` float32 arrays in m/s. Models that need temporal priors
take them as a second positional argument; the rest omit it entirely.

Predictors
----------
* :class:`CorrDiff` — V-CNN mean + residual diffusion with a sensor-noise dial.
  The most accurate predictor here. Needs priors.
* :class:`VCNN` — Voronoi-CNN baseline (Fukami et al. 2021). One forward pass,
  deterministic, ``uncertainty`` is zeros.
* :class:`StreamDDPM` — stream-function diffusion. Needs priors.
* :class:`DDPM` — split-head diffusion model.
* :class:`RePaint` — collaborator model: guided sampling (DPS/MCG) with 13 h and
  25 h temporal priors. Needs priors.
* :class:`RePaintUncond` — the same architecture trained without priors; its
  ``predict`` takes observations only.
* :class:`GP` — Matern-kriging baseline. No checkpoint (it fits per call) and the
  only model whose ``uncertainty`` is a native posterior sigma rather than a
  calibrated ensemble spread.
* :class:`DistAttn` — collaborator model: observations enter as cross-attention
  tokens penalised by distance and by observation AGE. The only predictor that
  uses the timestamp, so it handles a transect whose readings are not
  simultaneous. Takes observations only.

Uncertainty contract (changed in 0.8.0)
---------------------------------------
``CorrDiff``, ``StreamDDPM`` and ``DistAttn`` return a **conformally calibrated**
1-sigma by default: pass ``calibrate=False`` for the raw ensemble spread, or
``sigma_scale=`` to override the factor. The factor is chosen from the
observation timestamps, since it depends on whether the field was sampled
simultaneously or over a period; ``model._last_sigma_scale`` records the value
used and why. ``RePaint`` calibrates too (factor 2.4513, added 2026-09-03).
``RePaintUncond`` has no fitted factor and returns raw spread; ``VCNN`` returns
zeros; ``GP`` returns its own posterior sigma.

``CorrDiff`` is the recommended default. ``RePaint``/``RePaintUncond`` impose the
observations at sampling time rather than as trained conditioning, so they are
the natural comparison point for guided sampling versus conditioning.

Example
-------
    from ddpm_library import CorrDiff

    obs = [(18.305, -64.710, 1_700_000_000.0, 0.12, -0.03), ...]
    mean, unc = CorrDiff(device="auto").predict(obs, priors, n_draws=20)

See README.md for details.
"""

from .predict import DDPM, predict
from .vcnn_predict import VCNN, predict_vcnn
from .stream_predict import StreamDDPM, predict_stream
from .corrdiff_predict import CorrDiff, predict_corrdiff
from .repaint_predict import (
    RePaint, RePaintUncond, predict_repaint, predict_repaint_uncond,
)
from .distattn_predict import DistAttn, predict_distattn
from .gp_predict import GP, predict_gp
from . import metrics
from .geo import grid_arrays

__all__ = [
    "DDPM", "VCNN", "StreamDDPM", "CorrDiff", "RePaint", "RePaintUncond",
    "DistAttn", "GP",
    "predict", "predict_vcnn", "predict_stream", "predict_corrdiff",
    "predict_repaint", "predict_repaint_uncond", "predict_distattn",
    "predict_gp",
    "metrics", "grid_arrays",
]
# 0.9.0 (2026-09-04), all measured -- see docs/DEFAULTS_AND_DIALS.md:
#   * STREAM_UNC_SMOOTH_SIGMA 0.8 -> 3.2, which re-fitted
#     STREAM_SIGMA_SCALE_TIMED 3.304 -> 3.165 (the factor is coupled to the
#     smoothing sigma as well as the sampler step count).
#   * REPAINT_STRIDE 1 -> 5. Every reported RePaint number, the conformal factor
#     included, was produced at stride 5, but predict() defaulted to 1 and so
#     applied a factor that does not hold there.
#   * The conformal factor does not transfer across sampling settings, so all
#     four predictors now warn when n_draws -- or, for RePaint, stride --
#     differs from the value the factor was fitted at. Pass sigma_scale= to
#     take the factor over yourself.
#   * Ensemble sizes confirmed for CorrDiff and Stream (20 each). DistAttn's
#     shipped 10 is measurably NOT optimal (20 is better on four of six metrics)
#     and is left at 10 deliberately: see docs/DEFAULTS_AND_DIALS.md section 3b.
#   * Removed: repaint/loss_functions.py, two unused generators from
#     stream/paths.py, and the dead STREAM_UNCERTAINTY_N_DRAWS.
#
# 0.8.0 (2026-09-02), all measured -- see docs/AUDIT_2026-09-02.md:
#   * Stream direction weights -> the no-spread predecessor; sampler default
#     6 -> 2 steps; STREAM_SIGMA_SCALE_TIMED refitted to 3.304 (superseded by
#     3.165 in 0.9.0, above).
#   * CorrDiff/Stream/DistAttn now return CALIBRATED uncertainty by default.
#     Previously they delivered 0.834 / 0.489 / 0.737 coverage at a 90% target.
#   * RePaint gains a fitted conformal factor (2.4513); it calibrates as well as
#     the others (0.911 blind on v1b), correcting the earlier claim that it did not.
#   * Removed training-only code from repaint/ (never reachable at inference).
# Stream predictions and all three models' uncertainties differ from 0.7.x.
__version__ = "0.9.0"
