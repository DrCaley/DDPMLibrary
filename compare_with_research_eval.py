"""Apples-to-apples comparison: DDPMLibrary vs the research repo's
`scripts/eval_helmholtz_split.py` on the same checkpoint.

Both paths:
  - Load the SAME weights (ddpm_library/assets/weights.pt, which is a copy
    of experiments/12_helmholtz_dual_head/multires_splitnoise/results/
    inpaint_gaussian_t250_best_ema_weights.pt).
  - Build the SAME observation mask from a fixed seed.
  - Use the SAME inference algorithm (single-step at t=50, or iterative
    with t_start=75, resample_steps=3).
  - Use the SAME standardization (shared_mean=-0.0508, shared_std=0.1148).
  - Use the SAME Voronoi-filled base field for q_sample.

Expected: per-pixel difference is bounded by floating-point noise
(torch.manual_seed governs ε sampling). We report max |Δ| and MSE.

USAGE:
  # from DDPMLibrary root, with research-repo on PYTHONPATH:
  RESEARCH_REPO="/Users/caleyjb/Library/Mobile Documents/com~apple~CloudDocs/JeffsStuff/PLU/Research/diffusionInpaintingVectorFields"
  PYTHONPATH="$RESEARCH_REPO" python compare_with_research_eval.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

# ─── research repo import -----------------------------------------------------
RESEARCH_REPO = os.environ.get(
    "RESEARCH_REPO",
    "/Users/caleyjb/Library/Mobile Documents/com~apple~CloudDocs/JeffsStuff/"
    "PLU/Research/diffusionInpaintingVectorFields",
)
if RESEARCH_REPO not in sys.path:
    sys.path.insert(0, RESEARCH_REPO)

# The research repo's eval script has a lot of side effects on import
# (argparse, logging, CLI). Instead, we import only the pieces we need.
from scripts import eval_helmholtz_split as ehs  # type: ignore

# ─── DDPMLibrary import -------------------------------------------------------
from ddpm_library import DDPM
from ddpm_library.config import (
    DEFAULT_SINGLE_STEP_T, DEFAULT_T_START, DEFAULT_RESAMPLE_STEPS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, OCEAN_H, OCEAN_W,
    SHARED_MEAN, SHARED_STD, WEIGHTS_PATH,
)
from ddpm_library.geo import grid_arrays

# ─── constants ────────────────────────────────────────────────────────
MAT_PATH = Path(RESEARCH_REPO) / "data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"
# Research-repo checkpoint path (relative to research repo root, as ehs.load_model
# prepends BASE_DIR and reads the sibling resolved_config.yaml for UNet type).
WEIGHTS_REL_RESEARCH = str(
    Path(RESEARCH_REPO)
    / "experiments/12_helmholtz_dual_head/multires_splitnoise/results/"
      "inpaint_gaussian_t250_best_ema_weights.pt"
)


def load_one_gt_frame(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Load one (2, 44, 94) GT frame from the .mat file + ocean mask.

    The .mat has NaN on land cells. We return the frame with NaNs zeroed
    plus a (44, 94) ocean_mask (1 where valid ocean).
    """
    m = loadmat(str(MAT_PATH), variable_names=["u", "v"])
    u = m["u"][..., frame_idx].T  # (44, 94)
    v = m["v"][..., frame_idx].T
    nan_mask = np.isnan(u) | np.isnan(v)
    u = np.where(nan_mask, 0.0, u).astype(np.float32)
    v = np.where(nan_mask, 0.0, v).astype(np.float32)
    ocean_mask = (~nan_mask).astype(np.float32)
    gt = np.stack([u, v], axis=0)  # (2, 44, 94)
    return gt, ocean_mask


def research_eval_frame(frame_idx: int, pct: float, seed: int, algo: str):
    """Run the research repo's inference on one frame."""
    # Force the research eval's global device to match what DDPMLibrary will
    # use — so the outputs are bit-comparable.
    device = torch.device("cpu")  # CPU for reproducibility across backends
    ehs.device = device

    # Patch out voronoi: the library uses the sparse `known_std` (zero
    # outside observations) as the q_sample base — no voronoi fill. The
    # research eval's build_masks always returns a standardized voronoi
    # field (which is a NON-zero constant even when vor_fill=zeros, because
    # standardization subtracts the mean). Override build_masks here so
    # `vor_std` is zero on both sides. The multi-res model passes sparse
    # obs directly in its conditioning channels anyway.
    _orig_build_masks = ehs.build_masks

    def _no_voronoi_build_masks(gt_ocean, obs_mask, ocean_mask):
        known_std, miss_mask, known_mask, vor_std = _orig_build_masks(
            gt_ocean, obs_mask, ocean_mask)
        return known_std, miss_mask, known_mask, torch.zeros_like(vor_std)

    ehs.build_masks = _no_voronoi_build_masks

    # Load model (research repo path) — uses the SAME weights file.
    ddpm = ehs.load_model(WEIGHTS_REL_RESEARCH)
    # eval_helmholtz_split loads the split schedule lazily — make sure it's set
    # by re-running load_model side effects. We also need _split_schedule on CPU.
    if ehs._split_schedule is not None:
        ehs._split_schedule = ehs._split_schedule  # no-op; already constructed
        # Move schedule buffers to CPU
        for attr in ("betas_sol", "alphas_sol", "alpha_bars_sol",
                     "betas_irr", "alphas_irr", "alpha_bars_irr"):
            setattr(ehs._split_schedule, attr,
                    getattr(ehs._split_schedule, attr).to(device))
        ehs._split_schedule.device = device

    gt, ocean_mask = load_one_gt_frame(frame_idx)
    rng = np.random.default_rng(seed)
    obs_mask = ehs.random_obs_mask(ocean_mask, pct, rng)

    if algo == "single":
        pred = ehs.run_single_step(
            ddpm, gt, obs_mask, ocean_mask,
            seed=seed, t_val=DEFAULT_SINGLE_STEP_T,
        )
    else:
        pred = ehs.run_reverse_chain(
            ddpm, gt, obs_mask, ocean_mask,
            seed=seed, t_start=DEFAULT_T_START,
            resample_steps=DEFAULT_RESAMPLE_STEPS,
        )
    # pred is (2, 44, 94) raw m/s in OCEAN region.
    return gt, ocean_mask, obs_mask, pred


def library_predict_frame(
    gt: np.ndarray, ocean_mask: np.ndarray, obs_mask: np.ndarray,
    seed: int, algo: str, ddpm_model: DDPM,
) -> np.ndarray:
    """Run DDPMLibrary on the same frame/obs with cell-level input."""
    # Build observation tuples from the obs_mask. Use the library's own
    # lat/lon grid so rasterize.py places them in the exact same cells.
    lats, lons = grid_arrays()
    idx = np.argwhere(obs_mask > 0.5)  # (N, 2) [row, col]
    obs = []
    for (i, j) in idx:
        la = float(lats[i])
        lo = float(lons[j])
        u_val = float(gt[0, i, j])
        v_val = float(gt[1, i, j])
        obs.append((la, lo, 0.0, u_val, v_val))

    if algo == "single":
        mean, _ = ddpm_model.predict(
            obs, single_step=True, t_start=DEFAULT_SINGLE_STEP_T, seed=seed,
        )
    else:
        mean, _ = ddpm_model.predict(
            obs, single_step=False,
            t_start=DEFAULT_T_START, resample_steps=DEFAULT_RESAMPLE_STEPS,
            seed=seed,
        )
    # mean is (44, 94, 2) — reorder to (2, 44, 94) and mask to ocean
    pred = np.transpose(mean, (2, 0, 1)) * ocean_mask[None]
    return pred


def score(pred: np.ndarray, gt: np.ndarray, ocean_mask: np.ndarray) -> dict:
    diff = (pred - gt) * ocean_mask[None]
    n = ocean_mask.sum() * 2  # both channels
    mse = float((diff ** 2).sum() / n)
    rmse = float(np.sqrt(mse))
    return dict(mse=mse, rmse=rmse)


def compare(algo: str, frame_idx: int, pct: float, seed: int,
            ddpm_model: DDPM) -> None:
    print(f"\n=== algo={algo}  frame={frame_idx}  coverage={pct}%  seed={seed} ===")
    t0 = time.time()
    gt, ocean_mask, obs_mask, pred_research = research_eval_frame(
        frame_idx, pct, seed, algo)
    t_research = time.time() - t0

    t0 = time.time()
    pred_library = library_predict_frame(
        gt, ocean_mask, obs_mask, seed, algo, ddpm_model)
    t_library = time.time() - t0

    # pixel-level diff (on ocean cells only)
    diff = (pred_research - pred_library) * ocean_mask[None]
    mask_sum = ocean_mask.sum() * 2
    mean_abs = float(np.abs(diff).sum() / mask_sum)
    max_abs = float(np.max(np.abs(diff)))

    s_research = score(pred_research, gt, ocean_mask)
    s_library = score(pred_library, gt, ocean_mask)

    n_obs = int(obs_mask.sum())
    n_ocean = int(ocean_mask.sum())
    print(f"  ocean cells: {n_ocean}    observed: {n_obs}  ({100*n_obs/n_ocean:.2f}%)")
    print(f"  research eval : MSE={s_research['mse']:.6f}  RMSE={s_research['rmse']:.4f} m/s   "
          f"({t_research*1000:.0f} ms)")
    print(f"  DDPMLibrary   : MSE={s_library['mse']:.6f}  RMSE={s_library['rmse']:.4f} m/s   "
          f"({t_library*1000:.0f} ms)")
    print(f"  |research - library|   mean={mean_abs:.2e}  max={max_abs:.2e}  (m/s)")
    # flag equivalence
    if max_abs < 1e-5:
        print("  ✓ NUMERICALLY IDENTICAL (float tol)")
    elif max_abs < 1e-3:
        print("  ✓ near-identical (< 1 mm/s peak diff)")
    else:
        print("  ✗ notable divergence — investigate")


def main() -> None:
    # Use CPU on both sides for byte-level reproducibility.
    print(f"Loading DDPMLibrary model (forcing cpu) ...")
    ddpm_model = DDPM(device="cpu")
    print(f"  device: {ddpm_model.device}")
    print(f"  weights: {WEIGHTS_PATH}")

    # Sweep 3 coverages × 2 algorithms on a single test frame.
    frame_idx = 15000  # arbitrary late-range frame
    seed = 42
    for algo in ("single", "iterative"):
        for pct in (0.5, 1.0, 2.0):
            compare(algo, frame_idx, pct, seed, ddpm_model)


if __name__ == "__main__":
    main()
