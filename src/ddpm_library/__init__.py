"""DDPMLibrary — standalone inference for ocean velocity inpainting.

Two predictors are provided:

* :class:`VCNN` — Voronoi-CNN baseline (Fukami et al. 2021). Single CNN
  forward pass; currently the more accurate of the two on this dataset
  and the recommended default.
* :class:`DDPM` — split-head diffusion model. Generative; can produce
  multiple plausible samples per call but has higher RMSE than V-CNN
  in our benchmarks.

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

__all__ = ["DDPM", "VCNN", "predict", "predict_vcnn"]
__version__ = "0.3.0"
