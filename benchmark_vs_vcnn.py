"""Multi-frame head-to-head: DDPMLibrary vs V-CNN on the same frames / masks.

Uses the research-repo test-split protocol:
  - rng = np.random.default_rng(7777); pick n_frames from .mat
  - per-frame obs_mask with seed = args_seed + frame_i
  - percentage-based coverage (0.5, 1.0, 2.0)
  - score MSE over ALL ocean cells (observed + unobserved), averaged per frame

Both V-CNN and DDPMLibrary load the same weights they were trained on.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

RESEARCH_REPO = os.environ.get(
    "RESEARCH_REPO",
    "/Users/caleyjb/Library/Mobile Documents/com~apple~CloudDocs/JeffsStuff/"
    "PLU/Research/diffusionInpaintingVectorFields",
)
if RESEARCH_REPO not in sys.path:
    sys.path.insert(0, RESEARCH_REPO)

from scripts import eval_helmholtz_split as ehs  # type: ignore

from ddpm_library import DDPM
from ddpm_library.config import OCEAN_H, OCEAN_W
from ddpm_library.geo import grid_arrays

MAT_PATH = Path(RESEARCH_REPO) / "data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"


def load_all_gt():
    m = loadmat(str(MAT_PATH), variable_names=["u", "v"])
    u = np.transpose(m["u"], (2, 1, 0)).astype(np.float32)  # (T, 44, 94)
    v = np.transpose(m["v"], (2, 1, 0)).astype(np.float32)
    nan = np.isnan(u) | np.isnan(v)
    u = np.where(nan, 0.0, u)
    v = np.where(nan, 0.0, v)
    # conservative ocean mask = cells valid at every frame
    ocean_mask = (~nan.any(axis=0)).astype(np.float32)
    gt = np.stack([u, v], axis=1)  # (T, 2, 44, 94)
    return gt, ocean_mask


def run_library(ddpm_model, gt_frame, obs_mask, ocean_mask, seed,
                voronoi: bool):
    lats, lons = grid_arrays()
    idx = np.argwhere(obs_mask > 0.5)
    obs = [(float(lats[i]), float(lons[j]), 0.0,
            float(gt_frame[0, i, j]), float(gt_frame[1, i, j]))
           for (i, j) in idx]
    mean, _ = ddpm_model.predict(
        obs, single_step=True, voronoi=voronoi, seed=seed)
    return np.transpose(mean, (2, 0, 1)) * ocean_mask[None]


def mse_on_ocean(pred, gt, ocean_mask):
    diff = (pred - gt) * ocean_mask[None]
    return float((diff ** 2).sum() / (ocean_mask.sum() * 2))


def main(n_frames: int = 20, coverages=(0.5, 1.0, 2.0), base_seed: int = 42):
    print(f"Loading {n_frames} frames + ocean mask from .mat ...")
    gt_all, ocean_mask = load_all_gt()
    T = gt_all.shape[0]
    print(f"  total frames: {T}   ocean cells: {int(ocean_mask.sum())} / {44*94}")

    rng_frames = np.random.default_rng(seed=7777)
    frame_indices = rng_frames.choice(T, size=n_frames, replace=False)

    print("Loading models (cpu) ...")
    dev = torch.device("cpu")
    ddpm_model = DDPM(device="cpu")
    vcnn = ehs.load_vcnn(dev)

    print(f"\n{'coverage':>8}  {'method':>12}  {'RMSE (m/s)':>12}  "
          f"{'MSE':>12}  {'t/call (ms)':>12}")
    print("-" * 68)
    for pct in coverages:
        methods = ("V-CNN", "DDPM (no-voronoi)", "DDPM (voronoi)")
        results = {m: [] for m in methods}
        times = {m: [] for m in methods}
        for vi, fi in enumerate(frame_indices):
            gt = gt_all[fi]
            seed = base_seed + vi
            obs_rng = np.random.default_rng(seed)
            obs_mask = ehs.random_obs_mask(ocean_mask, pct, obs_rng)

            t0 = time.time()
            p_vcnn = ehs.predict_vcnn(
                vcnn, gt * obs_mask[None], obs_mask, ocean_mask, dev)
            times["V-CNN"].append(time.time() - t0)
            results["V-CNN"].append(mse_on_ocean(p_vcnn, gt, ocean_mask))

            t0 = time.time()
            p_nv = run_library(ddpm_model, gt, obs_mask, ocean_mask, seed,
                               voronoi=False)
            times["DDPM (no-voronoi)"].append(time.time() - t0)
            results["DDPM (no-voronoi)"].append(
                mse_on_ocean(p_nv, gt, ocean_mask))

            t0 = time.time()
            p_v = run_library(ddpm_model, gt, obs_mask, ocean_mask, seed,
                              voronoi=True)
            times["DDPM (voronoi)"].append(time.time() - t0)
            results["DDPM (voronoi)"].append(
                mse_on_ocean(p_v, gt, ocean_mask))

        for method in methods:
            mse = float(np.mean(results[method]))
            rmse = float(np.sqrt(mse))
            tc = float(np.mean(times[method]) * 1000)
            print(f"{pct:>7.1f}%  {method:>18}  {rmse:>12.4f}  "
                  f"{mse:>12.6f}  {tc:>12.1f}")
        rmse_vcnn = np.sqrt(np.mean(results["V-CNN"]))
        for method in ("DDPM (no-voronoi)", "DDPM (voronoi)"):
            ratio = np.sqrt(np.mean(results[method])) / rmse_vcnn
            print(f"{'':>8}  {method + ' / V-CNN':>18}  {ratio:>11.2f}×  "
                  f"({'better' if ratio < 1 else 'worse'})")
        print()


if __name__ == "__main__":
    main()
