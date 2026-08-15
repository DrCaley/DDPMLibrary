"""Self-contained vendoring of the research CorrDiff pipeline.

Nothing here depends on the research repo. Everything needed to run the
sensor-noise-conditioned residual diffusion model ("CorrDiff-sigma") is copied
faithfully so inference matches the published evaluation.

Method: a deterministic V-CNN predicts the mean field; a conditional diffusion
model predicts the RESIDUAL (truth - mean) rather than the field itself
(Mardani et al. 2025). Sampling the residual many times gives an ensemble whose
mean is the reconstruction and whose spread is a calibrated uncertainty estimate.
"""

from .conditioning import (
    DIST_SCALE, N_GEOM_CH, N_OBS_CH, assemble_cond, dist_channel,
    geometry_channels, observation_channels, sigma_channel,
)
from .diffusion import VDiffusion, cosine_alpha_bar, ddim_sample_residual
from .unet import ResBlock, UNet, sinusoidal_embedding

__all__ = [
    "UNet", "ResBlock", "sinusoidal_embedding",
    "VDiffusion", "cosine_alpha_bar", "ddim_sample_residual",
    "geometry_channels", "observation_channels", "assemble_cond",
    "dist_channel", "sigma_channel",
    "N_OBS_CH", "N_GEOM_CH", "DIST_SCALE",
]
