"""Validate the RePaint adapter by reproducing the published number.

The wrapper is a reconstruction of an inference script that was never pushed, so
it is only trustworthy if it lands near the published result:

    DPS-TimeConditioned  RMSE mean = 0.0410   (100 seeds, 90 px/seed, n=10)

Uses REAL fields, REAL 13h/25h priors and 90 real observations drawn from the
truth -- unlike the synthetic smoke test, which fed physically impossible
observations and could not validate anything. RMSE convention matches theirs
exactly: sqrt(mean((pred - true)^2)) over both components and ocean cells.
"""
import pickle
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "/Users/henryw/Documents/DDPMLibrary/src")
from ddpm_library import RePaint                                     # noqa: E402
from ddpm_library.config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN   # noqa: E402

PICKLE = "/Users/henryw/Documents/DiffusionSummer2026/Datasets/pickles/data_raw_chrono.pickle"
N_FRAMES, N_DRAWS, STRIDE, N_OBS = 5, 3, 1, 90
LAGS = (13, 25)

with open(PICKLE, "rb") as f:
    data = pickle.load(f)
fields = np.nan_to_num(np.asarray(data["fields"], dtype=np.float32))   # (N,2,94,44)
land = np.asarray(data["land_mask"], dtype=bool)                      # (94,44)
ocean = ~land
print(f"data {fields.shape}  land {land.shape}  ocean cells {ocean.sum()}")
print(f"mean true speed {np.linalg.norm(fields[:, :, ocean], axis=1).mean():.4f} m/s")

m = RePaint(device="cpu")
lats = np.linspace(LAT_MIN, LAT_MAX, 44)
lons = np.linspace(LON_MIN, LON_MAX, 94)

rng = np.random.default_rng(0)
lo = max(LAGS)
picks = sorted(int(x) for x in rng.choice(np.arange(lo, fields.shape[0]),
                                          size=N_FRAMES, replace=False))
rmses, obs_errs, speeds = [], [], []
for n, t in enumerate(picks):
    truth_m = fields[t]                                    # (2,94,44) model orientation
    priors_m = [fields[t - L] for L in LAGS]

    # 90 observations on a contiguous walk over ocean cells (their path convention)
    r, c = 0, 0
    while not ocean[r, c]:
        r, c = int(rng.integers(0, 94)), int(rng.integers(0, 44))
    cells, seen = [], set()
    while len(cells) < N_OBS:
        if ocean[r, c] and (r, c) not in seen:
            seen.add((r, c)); cells.append((r, c))
        r = int(np.clip(r + rng.integers(-1, 2), 0, 93))
        c = int(np.clip(c + rng.integers(-1, 2), 0, 43))
    obs = [(float(lats[cc]), float(lons[rr]), 1.7e9 + t * 3600.0,
            float(truth_m[0, rr, cc]), float(truth_m[1, rr, cc])) for rr, cc in cells]

    priors_lib = [np.transpose(p, (2, 1, 0)) for p in priors_m]        # -> (44,94,2)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_lib, _ = m.predict(obs, priors_lib, n_draws=N_DRAWS, stride=STRIDE, seed=t)
    dt = time.perf_counter() - t0
    pred_m = np.transpose(mean_lib, (2, 1, 0))                         # -> (2,94,44)

    rmse = float(np.sqrt(np.mean((pred_m[:, ocean] - truth_m[:, ocean]) ** 2)))
    pm = np.zeros((94, 44), bool)
    for rr, cc in cells:
        pm[rr, cc] = True
    obs_err = float(np.sqrt(np.mean((pred_m[:, pm] - truth_m[:, pm]) ** 2)))
    sp = float(np.linalg.norm(pred_m[:, ocean], axis=0).mean())
    rmses.append(rmse); obs_errs.append(obs_err); speeds.append(sp)
    print(f"  frame {t:5d}: RMSE {rmse:.4f}   err@obs {obs_err:.4f}   "
          f"speed {sp:.4f} m/s   ({dt:.0f}s)", flush=True)

print(f"\nRMSE          mean {np.mean(rmses):.4f}   (published DPS: 0.0410)")
print(f"err @ observed cells {np.mean(obs_errs):.4f}   (guidance should keep this low)")
print(f"predicted speed      {np.mean(speeds):.4f} m/s")
d = abs(np.mean(rmses) - 0.0410) / 0.0410 * 100
print(f"\ndeviation from published: {d:.0f}%  -> "
      f"{'ADAPTER LOOKS CORRECT' if d < 25 else 'MISMATCH - do not trust the adapter yet'}")
print("VALIDATE_DONE")
