"""Unified z-score standardization helpers.

The split-head model was trained with UnifiedZScoreStandardizer: both u and v
are z-scored by the same (shared_mean, shared_std) so the divergence-free
structure is preserved. See config.SHARED_MEAN / SHARED_STD.
"""

import numpy as np

from .config import SHARED_MEAN, SHARED_STD


def standardize(x: np.ndarray) -> np.ndarray:
    """(x - shared_mean) / shared_std, same for u and v."""
    return (x - SHARED_MEAN) / SHARED_STD


def inverse_standardize(x: np.ndarray) -> np.ndarray:
    """x * shared_std + shared_mean."""
    return x * SHARED_STD + SHARED_MEAN
