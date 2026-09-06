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

# Checkpoint architecture / diffusion config (from StreamFn_Cond_x0_mag.pt).
# 2026-09-02: the shipped direction weights were changed from
# StreamFn_Cond_x0_mag_spread.pt (48 ep, spread term on) to its own no-spread
# predecessor StreamFn_Cond_x0_mag.pt (78 ep). The spread term -- the 4th
# direction-loss term, 1 - rho(sigma_model, sigma_empirical) -- was measured to
# degrade vorticity fidelity, interval sharpness, and the uncertainty-error
# correlation it was itself designed to raise, at a statistically tied RMSE.
# Verified across a short fine-tune, a 20k-step fine-tune on other hardware, and
# the two historical checkpoints, on both benchmarks. See
# docs/STREAM_LOSS_ABLATIONS.md. Architecture is unchanged: cond_ch 10,
# base_ch 64, time_dim 256, T 1000, cosine, div-free noise.
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
# 2026-09-02: was 6, a value validated against the PREVIOUS direction weights.
# Re-swept on the current weights over {1,2,3,4,5,6,10,16}: 2 is the optimum and
# beats 6 on RMSE (0.0865 vs 0.0919), vorticity RMSE (0.01654 vs 0.01743),
# vorticity correlation (0.571 vs 0.562), calibrated CRPS (0.0347 vs 0.0368) and
# interval width (0.216 vs 0.230) — replicated on ocean_bench_v1b — while costing
# a third of the compute. 1 step lowers RMSE and CRPS further but DEGRADES
# vorticity (corr 0.528), i.e. it buys the metric by blurring; 2 is the structure
# optimum. See docs/STREAM_LOSS_ABLATIONS.md.
STREAM_DPMPP_STEPS = 2
STREAM_DDPM_STEPS = 100          # the proven ancestral config
STREAM_DEFAULT_N_DRAWS = 20  # matches CorrDiff; every other diffusion predictor
                             # here defaults to 10-20. This was 1, which returned a
                             # single noisy draw as the "mean" and zeros for the
                             # uncertainty -- measured at +3.8% RMSE versus 20 draws.
                             # Set 1 explicitly for the fast single-field path.
#: Ensemble size STREAM_SIGMA_SCALE_TIMED was fitted at (see CORRDIFF_FITTED_N_DRAWS).
STREAM_FITTED_N_DRAWS = 20

#: Below this, Stream's coupled-magnitude fuse stops carrying diffusion spread in
#: the magnitude: it standardizes each draw's magnitude across the ensemble, so a
#: one-draw "ensemble" has a z-score of exactly zero and every cell falls back to
#: the heteroscedastic network's mean speed. Not a speed/quality dial.
STREAM_MIN_COUPLED_DRAWS = 5

#: Split-conformal factor for Stream's raw ensemble spread, at the CURRENT
#: defaults (full_field=True, n_draws=20, dpmpp with STREAM_DPMPP_STEPS steps) on
#: 2 h time-varying collection. Fitted on 20 cases of ocean_bench_v1, held-out
#: coverage 0.895 at the 0.90 level; applied blind to all of ocean_bench_v1b it
#: gives 0.910 at width 0.221. Multiply the returned uncertainty by this for
#: calibrated intervals.
#: MEASURED -- do not edit by hand without re-fitting. It is coupled to the sampler
#: step count, to STREAM_UNC_SMOOTH_SIGMA, and to the `helmholtz_project` symbol:
#: 2.909 at 6 steps with sigma 0.8, 3.304 at 2 steps with sigma 0.8, 3.165 at 2
#: steps with sigma 3.2 under the old continuous Fourier symbol, and 3.147 at the
#: current settings under the discrete central-difference symbol. Re-fit if any of
#: those three change.
#: REFIT 2026-09-05 by `benchmark/uncertainty_final.py` after the symbol fix:
#: 40 cases of ocean_bench_v1, 20 fit / 20 verify, held-out coverage 0.8929.
STREAM_SIGMA_SCALE_TIMED = 3.147   # used by the uncertainty-map scripts

# The stream-function + div-free-noise scheme uses central differences, whose
# Fourier symbol vanishes at the Nyquist frequency, so grid-scale (checkerboard)
# modes are unconstrained by the div-free structure and show up as a numerical
# artifact in the ENSEMBLE SPREAD (not the mean field). A light nan-aware
# Gaussian smooth of the uncertainty removes it and improves calibration
# (r_angle/mag/overall all rise ~0.025 on a 40-frame test).
# 2026-09-02: was 0.8, a value never validated against a sweep. Re-swept at the
# current defaults over {0, 0.8, 1.6, 3.2, 6.4, 12.8} with the conformal factor
# re-fitted per value: 3.2 gives significantly narrower intervals (0.2075 vs
# 0.2160, CI [-0.0090, -0.0080]) and better calibrated CRPS (0.0344 vs 0.0347,
# CI [-0.00034, -0.00021]) at matched coverage, with r(sigma, error) tied.
# r peaks here and falls beyond, so this is an interior optimum, not an edge.
STREAM_UNC_SMOOTH_SIGMA = 3.2


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
#: Ensemble size the conformal factors below were fitted at. Kept separate from the
#: default on purpose: the default is a speed/quality choice, this is a fact about
#: the factor, and the predictors warn when a call does not match it. Changing the
#: default without refitting must therefore leave this alone.
CORRDIFF_FITTED_N_DRAWS = 20
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

# The factor above assumes every observation was taken at the SAME instant. A real
# vehicle needs ~1.2 h to cover 90 cells, and over that window the field moves more
# than the model's own error, so the intervals come out too narrow (coverage ~0.82).
# This is the factor refit on time-stamped observations sampled from the field at the
# moment each cell was actually visited. Pass it as `sigma_scale=` when your
# observations were collected over a period rather than simultaneously.
# MEASURED by `scripts/staleness.py recalibrate`: split conformal, 58 frames (29 fit /
# 29 verify), out-of-sample coverage 0.9096 vs 0.900 target; intervals 1.24x wider.
# CROSS-CHECK 2026-09-05: `benchmark/uncertainty_final.py` fits 2.201 on its own
# 40 benchmark cases, 1% from the value below. Different set, different split, and
# CorrDiff's code is unchanged by the audit, so the 58-frame fit above stands.
CORRDIFF_SIGMA_SCALE_TIMED = 2.1801


# ===========================================================================
# RePaint pipeline (collaborator model: linear-schedule RePaint UNet, temporal
# priors as conditioning, observations imposed by DPS/MCG guidance at sampling).
# Additive: does NOT affect the predictors above.
# ===========================================================================

# One checkpoint per variant; each is pinned by its own class (RePaint /
# RePaintUncond), which rejects the other's checkpoint rather than silently
# running the wrong model.
REPAINT_WEIGHTS_PATH = _ASSETS_DIR / "repaint_timecond_weights.pt"
# Same architecture, trained WITHOUT temporal priors (cond_ch = 0).
REPAINT_UNCOND_WEIGHTS_PATH = _ASSETS_DIR / "repaint_uncond_weights.pt"

# Native model grid (same as the other pipelines: library is lat x lon 44 x 94).
REPAINT_H, REPAINT_W = 94, 44

# From the checkpoint (epoch 120, val_loss 2.13e-4).
REPAINT_COND_CH = 4          # prev_13h(u, v), prev_25h(u, v)
REPAINT_LAGS = (13, 25)
REPAINT_BASE_CH = 64
REPAINT_TIME_DIM = 256
REPAINT_T = 1000
REPAINT_SCHEDULE = "linear"

# IMPORTANT: this pipeline is trained on RAW physical values (m/s), NOT z-scored,
# unlike the CorrDiff/Stream pipelines. Do not standardize its inputs.
REPAINT_NOISE_STD = 0.11614292860031128   # from the checkpoint; scales the initial latent

# Sampling. The published evaluation used DPS/MCG with step_size (z) = 0.04 and
# the full 1000-step chain; stride > 1 subsamples the chain for speed.
REPAINT_SAMPLER = "dps"       # DPS marginally beat MCG in the published numbers
REPAINT_STEP_SIZE = 0.04
#: Chain subsampling. Every RePaint number we report -- the conformal factor below,
#: the v1b calibration, the 49.7 s/field cost -- was produced at stride 5, so that is
#: the default: a default no measurement supports is worse than one that matches the
#: evidence. Changed from 1 on 2026-09-04, when the mismatch was found (the factor is
#: fitted at stride 5 and `predict()` was defaulting to stride 1, silently applying a
#: factor that does not hold).
#: OPEN: stride 1 has never been scored against stride 5 head to head -- stride 1 is
#: ~5x the cost, about 7 h for the 40-case benchmark, so it was not affordable. More
#: steps would normally mean better samples, so treat 5 as the calibrated setting
#: rather than as a proven optimum.
REPAINT_STRIDE = 5
#: The stride the shipped factor was fitted at; predictors warn when they differ.
REPAINT_CALIBRATED_STRIDE = 5
REPAINT_DEFAULT_N_DRAWS = 10  # the published evaluation used n = 10 per seed
#: Ensemble size REPAINT_SIGMA_SCALE_TIMED was fitted at.
REPAINT_FITTED_N_DRAWS = 10

#: Split-conformal factor for RePaint's raw ensemble spread (n_draws=10, stride 5,
#: 1 h observation cutoff) on 2 h time-varying collection. Fitted on the first 20
#: cases of ocean_bench_v1, held-out coverage 0.920 at the 0.90 level; applied blind
#: to all of ocean_bench_v1b it gives 0.911 at width 0.195.
#: MEASURED 2026-09-03 -- do not edit by hand without re-fitting.
#: Added because RePaint was the only diffusion predictor still returning raw
#: spread. It also corrects the record: RePaint calibrates as well as the others
#: (0.911 blind, against corrdiff 0.917 / stream 0.910 / distattn 0.894), so the
#: reason to prefer CorrDiff is its 41x lower cost, not interval quality.
#: STALE as of 2026-09-05: fitted before RePaint's per-draw seeding was fixed, and
#: RePaint is not in `benchmark/uncertainty_final.py`, so nothing refit it. RePaint
#: is not one of the three paper models. Refit before quoting this number.
REPAINT_SIGMA_SCALE_TIMED = 2.4513


# ===========================================================================
# Distance/time-aware attention pipeline (collaborator model: observations as
# cross-attention tokens, penalised by physical distance and by observation age).
# Additive: does NOT affect the predictors above.
# ===========================================================================

DISTATTN_WEIGHTS_PATH = _ASSETS_DIR / "distattn_weights.pt"
# The model's OWN ocean mask (a strict subset of the shared grid: 3749 cells
# vs 3787). Sampling zeroes land every step, so using the shared mask instead
# would diverge from how the model was trained.
DISTATTN_OCEAN_MASK_PATH = _ASSETS_DIR / "distattn_ocean_mask.npy"

# Native model grid, shared with the RePaint pipeline (library is lat x lon 44 x 94).
DISTATTN_H, DISTATTN_W = 94, 44

# From the checkpoint (epoch 142, val_loss 7.12e-4).
DISTATTN_BASE_CH = 64
DISTATTN_TIME_DIM = 256
DISTATTN_N_HEADS = 4
DISTATTN_T = 1000
DISTATTN_OBS_DIM = 5          # [x_norm, y_norm, u, v, age_norm]

# IMPORTANT: this pipeline is trained on RAW physical values (m/s), NOT z-scored,
# like the RePaint pipeline and unlike CorrDiff/Stream. Do not standardize.
DISTATTN_NOISE_STD = 0.11618577542245857  # from the checkpoint; scales the latent

# Observation age is tokenised in HOURS: age_norm = (t_end - t_obs) / 3600.
DISTATTN_AGE_SCALE_SEC = 3600.0

# The model was trained on transects spanning 5 min to 3 h, so ages much beyond
# ~3 h are out of distribution. predict() warns past this.
DISTATTN_MAX_AGE_SEC = 10800.0

# Sampling: strided DDPM reverse chain. stride=10 -> 100 network calls.
DISTATTN_STRIDE = 10

#: Split-conformal factor for DistAttn's raw ensemble spread on 2 h time-varying
#: collection, fitted on the first 20 cases of ocean_bench_v1 with coverage verified
#: on the other 20. predict() applies it; pass calibrate=False for the raw spread.
#: MEASURED -- do not edit by hand without re-fitting.
#: History: 1.621 at n_draws=10; 1.3592 when the default went 10 -> 20, because a
#: 20-draw ensemble is less under-dispersed and needs less inflation.
#: REFIT 2026-09-05 to 1.490 by `benchmark/uncertainty_final.py` after the per-draw
#: seeding was fixed. The old value was fitted while neighbouring benchmark cases
#: shared 19 of their 20 noise draws, so the fit and verify halves were not
#: independent and the factor came out too small. 40 cases, 20 fit / 20 verify,
#: held-out coverage 0.9006 against the 0.90 target.
DISTATTN_SIGMA_SCALE_TIMED = 1.490
#: MEASURED 2026-09-04, raised 10 -> 20. The 10 was inherited from the collaborator's
#: evaluation, not swept here. At 20 the model is significantly better on four of six
#: metrics (calibrated CRPS -0.00093, vorticity correlation +0.0135, vorticity RMSE
#: -0.00020, divergence RMSE -0.00012) with the calibrated interval 14% narrower, and
#: 40 is no better than 20 -- an interior optimum, not "more is better".
#: The price is real: inference cost doubles, 38 -> ~76 s per field, taking DistAttn
#: from ~15x CorrDiff to ~29x, and anything produced at n_draws=10 is not comparable.
#: See docs/DEFAULTS_AND_DIALS.md 3b and benchmark/n_draws_sweep.py.
DISTATTN_DEFAULT_N_DRAWS = 20
#: Ensemble size DISTATTN_SIGMA_SCALE_TIMED was fitted at.
DISTATTN_FITTED_N_DRAWS = 20
