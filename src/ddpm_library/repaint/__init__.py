"""Self-contained vendoring of the RePaint (linear, time-conditioned) pipeline.

Joseph's model: an unconditional-style RePaint UNet whose only per-pixel
conditioning is the ocean state 13 h and 25 h earlier; the sparse observations are
imposed at SAMPLING time by guidance (DPS / MCG) rather than by conditioning.
Operates in physical m/s -- this pipeline is not z-scored.
"""

from .diffusion import DDPM
from .sampler import SAMPLERS, dps_infer, mcg_infer
from .unet import Repaint

__all__ = ["Repaint", "DDPM", "dps_infer", "mcg_infer", "SAMPLERS"]
