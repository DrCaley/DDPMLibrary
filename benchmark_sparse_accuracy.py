"""Benchmark DDPM.predict accuracy on sparse-observation reconstruction.

Protocol
--------
For each of several observation budgets, and for N frames randomly drawn
from the ground-truth ROMS data:
    1. Pick K random ocean cells (excluding land / NaN).
    2. Build observations = [(lat, lon, t, u_gt, v_gt)] from those cells.
    3. Call DDPM.predict(obs).
    4. Score the prediction on the *unobserved* ocean cells against
       ground truth (MSE, MAE, speed correlation).

A trivial zero-field baseline ("predict no current anywhere") is included
so the DDPM results can be interpreted in context. NOTE: per the research-
repo convention, V-CNN would be the proper comparison baseline, but this
library intentionally has no research-repo dependencies, so we only
include the zero baseline here.

Usage
-----
    python benchmark_sparse_accuracy.py \\
        --mat /path/to/stjohn_hourly_5m_velocity_ramhead_v2.mat \\
        --n-frames 20 \\
        --obs-counts 5 20 60
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from ddpm_library import DDPM
from ddpm_library.geo import grid_arrays


def load_ground_truth(mat_path: Path) -> np.ndarray:
    """Return (T, 44, 94, 2) float32 array; NaN on land cells."""
    m = loadmat(str(mat_path), variable_names=["u", "v"])
    u = m["u"].astype(np.float32)  # (lon=94, lat=44, T)
    v = m["v"].astype(np.float32)
    # Transpose to (T, lat, lon)
    u = np.transpose(u, (2, 1, 0))
    v = np.transpose(v, (2, 1, 0))
    return np.stack([u, v], axis=-1)  # (T, 44, 94, 2)


def score(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    """Per-field metrics on `valid` cells only.

    pred, gt : (44, 94, 2); valid : (44, 94) bool.
    """
    p = pred[valid]   # (N, 2)
    g = gt[valid]     # (N, 2)
    diff = p - g
    mse = float((diff ** 2).mean())
    mae = float(np.abs(diff).mean())
    # Speed metrics
    sp_p = np.sqrt((p ** 2).sum(axis=-1))
    sp_g = np.sqrt((g ** 2).sum(axis=-1))
    speed_mae = float(np.abs(sp_p - sp_g).mean())
    # Cosine alignment between predicted and true velocity vectors
    denom = (np.linalg.norm(p, axis=-1) * np.linalg.norm(g, axis=-1) + 1e-8)
    cos_sim = float(((p * g).sum(axis=-1) / denom).mean())
    return {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "speed_mae": speed_mae,
        "cosine_sim": cos_sim,
    }


def run_benchmark(
    mat_path: Path,
    n_frames: int,
    obs_counts: list[int],
    single_step: bool,
    t_start: int | None,
    device: str,
    seed: int,
) -> None:
    print(f"Loading ground truth from {mat_path} ...")
    gt_all = load_ground_truth(mat_path)
    T = gt_all.shape[0]
    print(f"  Loaded {T} frames, shape {gt_all.shape}")

    lat, lon = grid_arrays()
    # A cell is "ocean" iff both u and v are non-NaN at every time step we'll use.
    # Use a union across all frames as a conservative land mask.
    land_mask = np.isnan(gt_all).any(axis=(0, 3))  # (44, 94) True = land or missing
    ocean_mask = ~land_mask
    n_ocean = ocean_mask.sum()
    print(f"  Ocean cells: {n_ocean} / {44*94} ({100*n_ocean/(44*94):.1f}%)")

    rng = np.random.default_rng(seed)
    frame_indices = rng.choice(T, size=n_frames, replace=False)

    print(f"\nLoading DDPM (device={device}) ...")
    t0 = time.time()
    ddpm = DDPM(device=device)
    print(f"  Loaded on {ddpm.device} in {time.time() - t0:.1f}s")
    eff_t = t_start if t_start is not None else (50 if single_step else 75)
    mode = f"single_step (t={eff_t})" if single_step else f"iterative (t_start={eff_t})"
    print(f"  Inference mode: {mode}\n")

    ocean_ij = np.argwhere(ocean_mask)  # list of (i, j) for valid cells

    results: dict[int, list[dict]] = {k: [] for k in obs_counts}
    baseline: dict[int, list[dict]] = {k: [] for k in obs_counts}
    timings: dict[int, list[float]] = {k: [] for k in obs_counts}

    for fi, frame_idx in enumerate(frame_indices):
        gt_frame = gt_all[frame_idx]  # (44, 94, 2)

        for K in obs_counts:
            # Randomly sample K ocean cells as observations.
            sel = rng.choice(len(ocean_ij), size=K, replace=False)
            obs_cells = ocean_ij[sel]  # (K, 2)
            obs_set = {(int(i), int(j)) for i, j in obs_cells}

            obs_list = [
                (
                    float(lat[i]), float(lon[j]),
                    float(frame_idx * 3600.0),  # arbitrary unix-like time
                    float(gt_frame[i, j, 0]),
                    float(gt_frame[i, j, 1]),
                )
                for i, j in obs_cells
            ]

            # Score only on unobserved ocean cells to avoid trivially
            # counting the pinned observations.
            unobs_mask = ocean_mask.copy()
            for i, j in obs_cells:
                unobs_mask[i, j] = False

            t0 = time.time()
            pred, _ = ddpm.predict(
                obs_list, single_step=single_step, t_start=t_start, seed=seed,
            )
            timings[K].append(time.time() - t0)

            results[K].append(score(pred, gt_frame, unobs_mask))

            zero_pred = np.zeros_like(pred)
            baseline[K].append(score(zero_pred, gt_frame, unobs_mask))

        if (fi + 1) % max(1, n_frames // 10) == 0:
            print(f"  ... {fi+1}/{n_frames} frames")

    # Summary
    print("\n" + "=" * 78)
    print("SPARSE-OBSERVATION RECONSTRUCTION ACCURACY")
    print("=" * 78)
    print(f"Frames: {n_frames}    Mode: {mode}    Device: {ddpm.device}\n")

    header = f"{'K obs':>6} | {'DDPM RMSE':>10} {'MAE':>8} {'speed-MAE':>10} {'cos-sim':>8} | {'Zero RMSE':>10} {'MAE':>8} | {'t/call (ms)':>12}"
    print(header)
    print("-" * len(header))
    for K in obs_counts:
        d = results[K]
        b = baseline[K]
        ddpm_rmse = np.mean([r["rmse"] for r in d])
        ddpm_mae = np.mean([r["mae"] for r in d])
        ddpm_sp  = np.mean([r["speed_mae"] for r in d])
        ddpm_cos = np.mean([r["cosine_sim"] for r in d])
        base_rmse = np.mean([r["rmse"] for r in b])
        base_mae = np.mean([r["mae"] for r in b])
        ms = 1000 * np.mean(timings[K])
        print(f"{K:6d} | {ddpm_rmse:10.4f} {ddpm_mae:8.4f} {ddpm_sp:10.4f} {ddpm_cos:8.3f} | {base_rmse:10.4f} {base_mae:8.4f} | {ms:12.1f}")

    print("\nUnits: m/s. `cos-sim` = mean cosine similarity of (u,v) vectors (1.0 = perfect direction).")
    print(f"Random seed: {seed}")


def main() -> None:
    default_mat = Path(
        "/Users/caleyjb/Library/Mobile Documents/com~apple~CloudDocs/"
        "JeffsStuff/PLU/Research/diffusionInpaintingVectorFields/"
        "data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"
    )
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mat", type=Path, default=default_mat,
                   help="Path to stjohn_hourly_5m_velocity_ramhead_v2.mat")
    p.add_argument("--n-frames", type=int, default=20,
                   help="Number of random ground-truth frames to evaluate on")
    p.add_argument("--obs-counts", type=int, nargs="+", default=[5, 20, 60],
                   help="Number of sparse observations per trial")
    p.add_argument("--iterative", action="store_true",
                   help="Use full RePaint iterative inference instead of single-step")
    p.add_argument("--t-start", type=int, default=None,
                   help="Diffusion step to query (single-step) or start from (iterative). "
                        "Default: 50 for single-step, 75 for iterative.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.mat.exists():
        raise FileNotFoundError(
            f"Ground-truth .mat not found at {args.mat}. Pass --mat with "
            f"the correct path to stjohn_hourly_5m_velocity_ramhead_v2.mat."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_benchmark(
        mat_path=args.mat,
        n_frames=args.n_frames,
        obs_counts=args.obs_counts,
        single_step=not args.iterative,
        t_start=args.t_start,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
