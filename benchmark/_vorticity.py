"""One vorticity operator for the whole benchmark suite.

Four scripts had grown their own ``curl``: three used an explicit central
difference returning 0 on the outer ring, one used ``np.gradient``, which falls
back to a one-sided stencil there. They agree exactly on the interior and differ
by up to 0.043 on the boundary -- against a field vorticity RMS of 0.019 -- so
any comparison mixing the two would have been quietly wrong.

Use ``curl`` with ``interior_ocean_mask`` so the outer ring, where a centred
stencil is undefined, is excluded rather than approximated.
"""

import numpy as np

__all__ = ["curl", "interior_ocean_mask"]


def curl(field: np.ndarray) -> np.ndarray:
    """Relative vorticity dv/dx - du/dy of an (H, W, 2) field.

    Central differences with the [-1, 0, 1]/2 stencil, matching the operator the
    stream-function head, the curl/divergence loss and the divergence metric all
    use. The outer ring is returned as 0 and should be masked out, not scored.
    """
    u, v = field[..., 0], field[..., 1]
    dvdx = np.zeros_like(v); dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / 2
    dudy = np.zeros_like(u); dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / 2
    return dvdx - dudy


def interior_ocean_mask(ocean_mask: np.ndarray) -> np.ndarray:
    """``ocean_mask`` minus its outer ring, where the centred stencil is undefined."""
    ocean = np.asarray(ocean_mask, bool)
    interior = np.zeros_like(ocean); interior[1:-1, 1:-1] = True
    return interior & ocean
