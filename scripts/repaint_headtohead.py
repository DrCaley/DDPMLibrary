"""Joseph's two models head-to-head: do the temporal priors actually help?

Identical frames, identical observations, identical sampler settings (the published
DPS config, stride=1). The only difference is whether the network sees the 13h/25h
prior fields. Their training val-losses cannot answer this -- they were computed on
different datasets and measure denoising, not reconstruction.
"""
import pickle
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "/Users/henryw/Documents/DDPMLibrary/src")
from ddpm_library import RePaint, metrics                                  # noqa: E402
from ddpm_library.config import (LAT_MAX, LAT_MIN, LON_MAX, LON_MIN,       # noqa: E402
                                 REPAINT_UNCOND_WEIGHTS_PATH, REPAINT_WEIGHTS_PATH)

PICKLE = "/Users/henryw/Documents/DiffusionSummer2026/Datasets/pickles/data_raw_chrono.pickle"
N_FRAMES, N_DRAWS, STRIDE, N_OBS, LAGS = 4, 3, 1, 90, (13, 25)

with open(PICKLE, "rb") as f:
    data = pickle.load(f)
fields = np.nan_to_num(np.asarray(data["fields"], dtype=np.float32))
land = np.asarray(data["land_mask"], dtype=bool)
ocean = ~land
lats = np.linspace(LAT_MIN, LAT_MAX, 44)
lons = np.linspace(LON_MIN, LON_MAX, 94)

models = {
    "timecond": RePaint(device="cpu", weights_path=REPAINT_WEIGHTS_PATH),
    "uncond":   RePaint(device="cpu", weights_path=REPAINT_UNCOND_WEIGHTS_PATH),
}
common = np.ones((44, 94), bool)
for m in models.values():
    common &= m.ocean_mask > 0.5

rng = np.random.default_rng(0)
picks = sorted(int(x) for x in rng.choice(np.arange(max(LAGS), fields.shape[0]),
                                          size=N_FRAMES, replace=False))
acc = {k: [] for k in models}
for n, t in enumerate(picks):
    truth_m = fields[t]
    priors_lib = [np.transpose(fields[t - L], (2, 1, 0)) for L in LAGS]
    truth_lib = np.transpose(truth_m, (2, 1, 0))

    r, c = 40, 20
    cells, seen = [], set()
    while len(cells) < N_OBS:                       # contiguous track over ocean
        if ocean[r, c] and (r, c) not in seen:
            seen.add((r, c)); cells.append((r, c))
        r = int(np.clip(r + rng.integers(-1, 2), 0, 93))
        c = int(np.clip(c + rng.integers(-1, 2), 0, 43))
    obs = [(float(lats[cc]), float(lons[rr]), 1.7e9 + t * 3600.0,
            float(truth_m[0, rr, cc]), float(truth_m[1, rr, cc])) for rr, cc in cells]
    omask = np.zeros((44, 94), bool)
    for rr, cc in cells:
        omask[cc, rr] = True

    for name, mdl in models.items():
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, unc = mdl.predict(obs, priors_lib, n_draws=N_DRAWS,
                                    stride=STRIDE, seed=t)
        res = metrics.evaluate(mean, unc, truth_lib, ocean_mask=common,
                               observed_mask=omask)
        acc[name].append(res)
        print(f"  frame {t:5d} {name:9s} rmse {res['rmse']:.4f}  crps {res['crps']:.4f}  "
              f"angle {res['angle_error']:.1f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

keys = ["crps", "rmse", "rmse_observed", "rmse_unobserved", "angle_error",
        "spread_skill_ratio", "ke_ratio_small", "eddy_hit_rate"]
print(f"\n{'model':<11}" + "".join(f"{k[:9]:>11}" for k in keys))
for name in models:
    print(f"{name:<11}" + "".join(
        f"{np.nanmean([r[k] for r in acc[name]]):>11.4f}" for k in keys))
print("\nHEADTOHEAD_DONE")
