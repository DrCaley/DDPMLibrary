"""DDPMLibrary — standalone inference for the split-head ocean velocity DDPM.

Public API:
    from ddpm_library import DDPM, predict

    mean, uncertainty = predict([(lat, lon, unix_t, u, v), ...])
    # mean: np.ndarray (44, 94, 2)   — (lat, lon, [u, v]) in m/s
    # uncertainty: np.ndarray (44, 94, 2) — zeros (see README limitations)

See README.md for details.
"""

from .predict import DDPM, predict

__all__ = ["DDPM", "predict"]
__version__ = "0.1.0"
