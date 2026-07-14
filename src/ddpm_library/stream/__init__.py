"""Self-contained vendoring of the research stream-function pipeline.

Nothing here depends on the research repo; everything needed to run the best
"old pipeline" (conditional stream-function DDPM direction x heteroscedastic
magnitude UNet, coupled fuse) is copied faithfully so inference matches the
research results.
"""

from .diffusion import DDPM, eps_wrapper_for, x0_from_output
from .stream_model import StreamFunctionUNet
from .mag_model import HeteroMagnitudeUNet
from .sampler import ensemble_infer, dpmpp_ensemble, sample_one
from .conditioning import (
    geometry_channels, observation_channels, assemble_cond, build_conditioning,
    load_hetero_magnitude_model, predict_speed_mean_sigma, coupled_magnitude,
    helmholtz_project, fuse_coupled,
    directional_spread, vector_spread, magnitude_spread, pcorr, EPS,
)

__all__ = [
    "DDPM", "eps_wrapper_for", "x0_from_output",
    "StreamFunctionUNet", "HeteroMagnitudeUNet",
    "ensemble_infer", "dpmpp_ensemble", "sample_one",
    "geometry_channels", "observation_channels", "assemble_cond",
    "build_conditioning", "load_hetero_magnitude_model",
    "predict_speed_mean_sigma", "coupled_magnitude", "helmholtz_project",
    "fuse_coupled", "directional_spread", "vector_spread", "magnitude_spread",
    "pcorr", "EPS",
]
