"""DDPMLibrary — standalone inference for ocean velocity reconstruction.

Every predictor exposes the same contract::

    mean, uncertainty = Model(device="auto").predict(observations[, priors])

with ``observations`` an iterable of ``(lat, lon, unix_t, u, v)`` and both
outputs ``(44, 94, 2)`` float32 arrays in m/s. Models that need temporal priors
take them as a second positional argument; the rest omit it entirely.

Predictors
----------
* :class:`CorrDiff` — V-CNN mean + residual diffusion with a sensor-noise dial.
  The most accurate predictor here, and the only one whose ``uncertainty`` is
  calibrated rather than raw spread. Needs priors.
* :class:`VCNN` — Voronoi-CNN baseline (Fukami et al. 2021). One forward pass,
  deterministic, ``uncertainty`` is zeros.
* :class:`StreamDDPM` — stream-function diffusion. Needs priors.
* :class:`DDPM` — split-head diffusion model.
* :class:`RePaint` — collaborator model: guided sampling (DPS/MCG) with 13 h and
  25 h temporal priors. Needs priors.
* :class:`RePaintUncond` — the same architecture trained without priors; its
  ``predict`` takes observations only.

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
from . import metrics
from .geo import grid_arrays

__all__ = [
    "DDPM", "VCNN", "StreamDDPM", "CorrDiff", "RePaint", "RePaintUncond",
    "predict", "predict_vcnn", "predict_stream", "predict_corrdiff",
    "predict_repaint", "predict_repaint_uncond",
    "metrics", "grid_arrays",
]
__version__ = "0.6.0"
