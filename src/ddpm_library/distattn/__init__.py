"""Self-contained vendoring of the collaborator's distance/time-aware attention model.

Unlike the other pipelines here, this one does not receive the observations as
input channels or impose them during sampling. They arrive as a SET OF TOKENS
``[x_norm, y_norm, u, v, age_norm]`` that every resolution level cross-attends
to, with the raw attention score penalised by two learned scalars::

    attn = (q @ k^T) / sqrt(d)  -  alpha * distance(query_xy, obs_xy)
                                -  beta  * age(obs)

``alpha`` and ``beta`` are zero-initialised, so training decides how much
physical distance and observation staleness matter. The age term is what lets
this model handle a vehicle transect whose readings were taken minutes to hours
apart, rather than treating them as simultaneous.

Operates in physical m/s -- not z-scored. ``unet.py`` and ``sampler.py`` are
vendored verbatim from the collaborator's repo; ``diffusion.py`` is the
inference subset of their ``ddpm.py``. The user-facing wrapper is in
``ddpm_library.distattn_predict``.
"""

from .diffusion import DDPM
from .sampler import ddpm_sample
from .unet import AGE_SCALE_SEC, OBS_DIM, TimeCondUNet

__all__ = ["TimeCondUNet", "DDPM", "ddpm_sample", "OBS_DIM", "AGE_SCALE_SEC"]
