"""Example usage of DDPMLibrary.

Demonstrates both predictors:

* ``VCNN``  — Voronoi-CNN baseline (Fukami et al. 2021), a single CNN
  forward pass. Currently the more accurate of the two on this dataset.
* ``DDPM``  — split-head diffusion model. Generative; can produce
  multiple plausible samples per call but has higher RMSE than V-CNN
  in our benchmarks.

Both share the same ``predict([(lat, lon, unix_t, u, v), ...])`` API.
"""

import numpy as np

from ddpm_library import DDPM, VCNN
from ddpm_library.geo import grid_arrays


def main() -> None:
    # Get the native lat/lon grid so we can place observations on valid nodes
    lat, lon = grid_arrays()
    print(f"Model grid: {lat.shape[0]} x {lon.shape[0]} "
          f"(lat ∈ [{lat.min():.4f}, {lat.max():.4f}], "
          f"lon ∈ [{lon.min():.4f}, {lon.max():.4f}])")

    # Build a handful of (lat, lon, unix_time, u, v) observations.
    # u, v are surface-velocity components in m/s.
    # unix_time is accepted but ignored by this version of the model.
    rng = np.random.default_rng(42)
    observations = []
    for _ in range(20):
        i = rng.integers(0, lat.shape[0])
        j = rng.integers(0, lon.shape[0])
        observations.append((
            float(lat[i]),
            float(lon[j]),
            1_700_000_000.0,
            float(rng.normal(0.0, 0.1)),   # u (m/s)
            float(rng.normal(0.0, 0.1)),   # v (m/s)
        ))

    # ── DDPM (diffusion) ────────────────────────────────────────────
    # Load the model. The first call downloads nothing — weights are bundled.
    # device="auto" picks CUDA > MPS > CPU.
    ddpm = DDPM(device="auto")
    print(f"DDPM loaded on device: {ddpm.device}")

    # Predict the full 44×94 field.
    # Default is single_step=True: one UNet forward pass, ~40 ms on MPS/CUDA.
    # Set single_step=False for the full iterative RePaint chain.
    mean, uncertainty = ddpm.predict(observations, seed=0)
    print(f"[DDPM]  mean shape: {mean.shape}, dtype: {mean.dtype}")
    print(f"[DDPM]  u range: [{mean[..., 0].min():.3f}, "
          f"{mean[..., 0].max():.3f}] m/s")
    print(f"[DDPM]  v range: [{mean[..., 1].min():.3f}, "
          f"{mean[..., 1].max():.3f}] m/s")

    # ── V-CNN (deterministic baseline) ──────────────────────────────
    vcnn = VCNN(device="auto")
    print(f"V-CNN loaded on device: {vcnn.device}")
    mean_v, _ = vcnn.predict(observations)
    print(f"[VCNN]  mean shape: {mean_v.shape}, dtype: {mean_v.dtype}")
    print(f"[VCNN]  u range: [{mean_v[..., 0].min():.3f}, "
          f"{mean_v[..., 0].max():.3f}] m/s")
    print(f"[VCNN]  v range: [{mean_v[..., 1].min():.3f}, "
          f"{mean_v[..., 1].max():.3f}] m/s")


if __name__ == "__main__":
    main()
