"""Baked-in constants and asset paths for the split-head DDPM model.

Values correspond to the EMA checkpoint from
    experiments/12_helmholtz_dual_head/multires_splitnoise/
trained on the St. John Rams Head "northwest" single-location dataset
(5-channel input, no bathymetry).
"""

import os
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
VCNN_WEIGHTS_PATH = _ASSETS_DIR / "vcnn_weights.pt"
LAT_LON_GRID_PATH = _ASSETS_DIR / "lat_lon_grid.npz"

# ── Default inference parameters (match scripts/eval_helmholtz_split.py) ─
# Single-step: one UNet call at a moderate noise level (not t=T-1, which
# is near-pure noise and starves the model of signal).
DEFAULT_SINGLE_STEP_T = 50
# Iterative RePaint: partial reverse chain with 3 re-samplings per step.
DEFAULT_T_START = 75
DEFAULT_RESAMPLE_STEPS = 3


# ===========================================================================
# Stream-function pipeline (the research "best" model — conditional stream-fn
# DDPM direction x heteroscedastic-magnitude UNet, coupled fuse).
# Additive: does NOT affect the DDPM / VCNN predictors above.
# ===========================================================================

# Assets (bundled via git-LFS like the other *.pt weights).
STREAM_DIR_WEIGHTS_PATH = _ASSETS_DIR / "stream_dir_weights.pt"   # diffusion direction
STREAM_MAG_WEIGHTS_PATH = _ASSETS_DIR / "stream_mag_weights.pt"   # hetero magnitude
STREAM_GRID_PATH = _ASSETS_DIR / "stream_grid.npz"               # land_mask + stats

# Full hourly dataset (conditioned-priors chrono pickle) — used ONLY by the
# eval/uncertainty-map scripts (to build temporal priors + empirical neighbours),
# NEVER by predict(). It is large (~540 MB) and is deliberately NOT bundled with
# the library. Point to it with the STREAM_DATASET environment variable, or pass
# --pickle to the script. None here means "not configured".
STREAM_DATASET_PATH = (
    Path(os.environ["STREAM_DATASET"]) if os.environ.get("STREAM_DATASET") else None
)

# Native model grid orientation (transpose of the library's lat x lon grid):
#   library grid is (lat=44, lon=94); the stream model works in (94, 44).
STREAM_H, STREAM_W = 94, 44

# Checkpoint architecture / diffusion config (from StreamFn_Cond_x0_mag_spread.pt).
STREAM_COND_CH = 10          # legacy: 3 obs + 4 priors (lags 13,25) + 3 geom
STREAM_LAGS = (13, 25)       # temporal-prior lags, in hours/frames
STREAM_PRED_TYPE = "x0_streamfn_cond"
STREAM_BASE_CH = 64
STREAM_TIME_DIM = 256
STREAM_T = 1000
STREAM_SCHEDULE = "cosine"
STREAM_NOISE_TYPE = "div_free"

# Default sampler. "dpmpp" = DPM-Solver++(2M): on this model it beats the DDPM
# ancestral sampler on every calibration/accuracy metric AND is ~24x faster
# (validated head-to-head). "ddpm" = the classic ancestral sampler (bit-exact
# to the research pipeline) — kept available for reproducing published numbers.
STREAM_SAMPLER = "dpmpp"
STREAM_DPMPP_STEPS = 6            # sweet spot: peak calibration, diverse draws
STREAM_DDPM_STEPS = 100          # the proven ancestral config
STREAM_DEFAULT_N_DRAWS = 1   # 1 -> single field (uncertainty zeros, like DDPM/VCNN);
                             # >1 -> real ensemble spread in the uncertainty output.
STREAM_UNCERTAINTY_N_DRAWS = 40   # used by the uncertainty-map scripts

# The stream-function + div-free-noise scheme uses central differences, whose
# Fourier symbol vanishes at the Nyquist frequency, so grid-scale (checkerboard)
# modes are unconstrained by the div-free structure and show up as a numerical
# artifact in the ENSEMBLE SPREAD (not the mean field). A light nan-aware
# Gaussian smooth of the uncertainty removes it and improves calibration
# (r_angle/mag/overall all rise ~0.025 on a 40-frame test).
STREAM_UNC_SMOOTH_SIGMA = 0.8


# ===========================================================================
# CorrDiff pipeline (the research group's best model — V-CNN mean + residual
# diffusion with a sensor-noise dial). Additive: does NOT affect the predictors
# above.
# ===========================================================================

# Assets (bundled via git-LFS like the other *.pt weights).
CORRDIFF_WEIGHTS_PATH = _ASSETS_DIR / "corrdiff_weights.pt"   # residual diffusion UNet
CORRDIFF_GRID_PATH = _ASSETS_DIR / "corrdiff_grid.npz"        # land_mask + stats + lags

# Native model grid orientation (transpose of the library's lat x lon grid):
#   library grid is (lat=44, lon=94); the CorrDiff model works in (94, 44).
CORRDIFF_H, CORRDIFF_W = 94, 44

# Checkpoint architecture / diffusion config (from corrdiff_sigma epoch 199).
CORRDIFF_COND_CH = 11        # 4 obs (u, v, mask, dist-to-path) + 4 priors + 3 geom
CORRDIFF_LAGS = (13, 25)     # temporal-prior lags, in hours/frames
CORRDIFF_BASE_CH = 64
CORRDIFF_TIME_DIM = 256
CORRDIFF_T = 1000            # training diffusion steps (cosine, v-prediction)

# Sampling. 50 DDIM steps is the evaluated setting. The ensemble MEAN is far less
# sensitive to step count than the SPREAD is: at 16 steps the mean RMSE is within
# ~2% but the distribution degrades, so do not reduce this when uncertainty matters.
CORRDIFF_STEPS = 50
CORRDIFF_DEFAULT_N_DRAWS = 20   # uncertainty is the point of this model; 1 = fast
                                # single field with zero uncertainty.

# Sensor-noise dial: the model was trained on sigma ~ U(0, CORRDIFF_NOISE_MAX),
# expressed as a fraction of the field standard deviation. predict() rejects
# values outside this range rather than silently extrapolating.
CORRDIFF_NOISE_MAX = 0.10

# Calibration. The raw diffusion ensemble is under-dispersed — a known property of
# conditional diffusion models, reported by the original CorrDiff paper too. This
# factor rescales the raw ensemble std into a calibrated 1-sigma, so the ordinary
# Gaussian reading holds: mu +/- 1.645 * sigma covers ~90% of outcomes. Fitted by
# split conformal on held-out frames at dial = 0 (see scripts/, MEASURED — do not
# edit by hand without re-fitting).
CORRDIFF_SIGMA_SCALE = 1.6787   # MEASURED: split conformal, 60 held-out frames,
                                # dial=0; out-of-sample coverage 0.8999 vs 0.900 target.
