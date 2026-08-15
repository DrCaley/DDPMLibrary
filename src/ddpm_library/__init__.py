"""DDPMLibrary — standalone inference for ocean velocity inpainting.

Two predictors are provided:

* :class:`VCNN` — Voronoi-CNN baseline (Fukami et al. 2021). Single CNN
  forward pass; currently the more accurate of the two on this dataset
  and the recommended default.
* :class:`DDPM` — split-head diffusion model. Generative; can produce
  multiple plausible samples per call but has higher RMSE than V-CNN
  in our benchmarks.
* :class:`CorrDiff` — V-CNN mean + residual diffusion with a sensor-noise
  dial. The most accurate predictor here, and the only one whose
  ``uncertainty`` output is calibrated rather than zeros. Needs temporal
  priors, like :class:`StreamDDPM`.

Both share the same ``predict([(lat, lon, unix_t, u, v), ...])`` API.

Example
-------
    from ddpm_library import VCNN, DDPM

    obs = [(18.305, -64.710, 1_700_000_000.0, 0.12, -0.03), ...]

    mean, _ = VCNN(device="auto").predict(obs)   # baseline (recommended)
    mean, _ = DDPM(device="auto").predict(obs)   # diffusion

See README.md for details.
"""

from .predict import DDPM, predict
from .vcnn_predict import VCNN, predict_vcnn
from .stream_predict import StreamDDPM, predict_stream
from .corrdiff_predict import CorrDiff, predict_corrdiff
from .repaint_predict import RePaint, predict_repaint
from . import metrics

__all__ = [
    "DDPM", "VCNN", "StreamDDPM", "CorrDiff", "RePaint",
    "predict", "predict_vcnn", "predict_stream", "predict_corrdiff",
    "predict_repaint",
    "metrics",
]
__version__ = "0.5.0"
