"""Self-contained vendoring of the collaborator's linear-schedule RePaint pipeline.

One copy of the model code serves BOTH published checkpoints: ``Repaint`` built
with ``cond_ch=4`` is the time-conditioned model (per-pixel conditioning on the
ocean state 13 h and 25 h earlier), and ``cond_ch=0`` reproduces the
unconditional model exactly. In both cases the sparse observations are imposed at
SAMPLING time by guidance (DPS / MCG) rather than by conditioning.

Operates in physical m/s -- this pipeline is not z-scored.

Vendored from the collaborator's training repo; the only edit is the import path
in ``diffusion.py`` (``from loss_functions`` -> ``from .loss_functions``). The
user-facing wrappers are in ``ddpm_library.repaint_predict``.
"""

from .diffusion import DDPM
from .sampler import SAMPLERS, dps_infer, mcg_infer
from .unet import Repaint

__all__ = ["Repaint", "DDPM", "dps_infer", "mcg_infer", "SAMPLERS"]
