"""Baked-in constants and asset paths for the split-head DDPM model.

Values correspond to the EMA checkpoint from
    experiments/12_helmholtz_dual_head/multires_splitnoise/
trained on the St. John Rams Head "northwest" single-location dataset
(5-channel input, no bathymetry).
"""

from pathlib import Path

# ── Grid ──────────────────────────────────────────────────────────────
# Native ocean grid (lat × lon) — output shape of predict()
OCEAN_H, OCEAN_W = 44, 94
# Internal UNet grid (zero-padded, upper-left corner is the ocean)
FULL_H, FULL_W = 64, 128

# ── Diffusion schedule (split Helmholtz) ──────────────────────────────
N_STEPS = 250
MIN_BETA = 1e-4
MAX_BETA = 0.02
IRR_SPEED = 2.0

# ── Data standardization (unified z-score) ────────────────────────────
SHARED_MEAN = -0.05084468695562498
SHARED_STD = 0.11479844598042026

# ── Geographic bounding box (St. John Rams Head ROMS grid) ────────────
# Extracted directly from data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat
# This is the native 44×94 ocean grid the model was trained on.
LAT_MIN = 18.290007194176347
LAT_MAX = 18.309564294149357
LON_MIN = -64.724759611195822
LON_MAX = -64.680048072672093

# ── Asset paths ───────────────────────────────────────────────────────
# Assets live inside the package so they are bundled by pip / wheel.
_ASSETS_DIR = Path(__file__).parent / "assets"
WEIGHTS_PATH = _ASSETS_DIR / "weights.pt"
LAT_LON_GRID_PATH = _ASSETS_DIR / "lat_lon_grid.npz"

# ── Default inference parameters (match scripts/eval_helmholtz_split.py) ─
# Single-step: one UNet call at a moderate noise level (not t=T-1, which
# is near-pure noise and starves the model of signal).
DEFAULT_SINGLE_STEP_T = 50
# Iterative RePaint: partial reverse chain with 3 re-samplings per step.
DEFAULT_T_START = 75
DEFAULT_RESAMPLE_STEPS = 3
