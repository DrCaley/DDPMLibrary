"""Second point on CorrDiff's observation-noise curve.

`corrdiff_noise_and_draws.py` tested one noise level, 5% of the field standard
deviation. That is only ~8% of the model's own RMSE, so "noise costs nothing" was
always going to be the answer there. This adds the 10% level -- CORRDIFF_NOISE_MAX,
the largest the channel was trained for -- so the claim rests on a curve rather
than a single point. Arms D and E see byte-identical noisy observations.
"""
import sys, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmark"))
from ddpm_library import CorrDiff, metrics                           # noqa: E402
import _score                                                         # noqa: E402

SEED = 20260830
NOISE_FRAC = 0.10
b = np.load(ROOT / "benchmark/ocean_bench_v1.npz")
obs, pri, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
n = len(truth)
prev = torch.load(ROOT / "benchmark/results_corrdiff_noise_draws.pt", weights_only=False)

KEYS = ("rmse", "angle", "vort_rmse", "vort_corr", "crps_cal")


def score(m, s, label):
    pc, sm = _score.case_scores(m, s, truth, ocean, crps_fn=metrics.crps_gaussian,
                                keys=KEYS)
    print(_score.row(label, pc, sm), flush=True)
    return pc


def boot(A, B, lab):
    return _score.paired_bootstrap(A, B, lab, keys=KEYS)


def run(model, obs_arr, sensor_noise):
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = model.predict([tuple(x) for x in obs_arr[i]], pri[i], n_draws=20,
                                 sensor_noise=sensor_noise, seed=SEED + i, calibrate=False)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    return np.stack(M), np.stack(S)

cd = CorrDiff(device="mps")
sigma = NOISE_FRAC * float(np.mean(cd.data_std))
noisy = obs.copy().astype(np.float64)
for i in range(n):
    r = np.random.default_rng(2000 + i)
    noisy[i][:, 3:5] += r.normal(0.0, sigma, size=noisy[i][:, 3:5].shape)

print(f"=== obs error {NOISE_FRAC:.0%} of field std = {sigma:.4f} m/s "
      f"({sigma/0.06684:.0%} of CorrDiff's RMSE) ===")
print(_score.HEADER)
D = score(*run(cd, noisy, 0.0),        "D  10% noisy obs, noise=0 (not told)")
E = score(*run(cd, noisy, NOISE_FRAC), "E  10% noisy obs, noise=0.10 (told)")
boot(E, D, "E minus D -- does telling the model help at 10%")
boot(D, prev["A"], "D minus A -- what 10% noise costs")

prev["D"], prev["E"] = D, E
prev["meta"]["noise_frac_2"] = NOISE_FRAC; prev["meta"]["sigma_ms_2"] = sigma
torch.save(prev, ROOT / "benchmark/results_corrdiff_noise_draws.pt")
print("\nupdated benchmark/results_corrdiff_noise_draws.pt")
