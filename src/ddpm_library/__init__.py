"""DDPMLibrary — standalone inference for ocean velocity inpainting.

Two predictors are provided:

* :class:`DDPM` — split-head diffusion model (more accurate, slower).
* :class:`VCNN` — Voronoi-CNN baseline (Fukami et al. 2021): faster and
  often a better starting point for very sparse observation regimes.

Both share the same ``predict([(lat, lon, unix_t, u, v), ...])`` API.

Example
-------
    from ddpm_library import DDPM, VCNN

    obs = [(18.305, -64.710, 1_700_000_000.0, 0.12, -0.03), ...]

    mean, _ = DDPM(device="auto").predict(obs)   # diffusion
    mean, _ = VCNN(device="auto").predict(obs)   # baseline

See README.md for details.
"""

from .predict import DDPM, predict
from .vcnn_predict import VCNN, predict_vcnn

__all__ = ["DDPM", "VCNN", "predict", "predict_vcnn"]
__version__ = "0.3.0"
