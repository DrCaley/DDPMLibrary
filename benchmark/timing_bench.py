"""Measured inference cost per field, per model, at the paper configurations.

The speed claims in circulation (~15 s CorrDiff, ~4 min RePaint) are folklore
from different machines and settings. This measures them on one GPU, at exactly
the configurations the paper tables use, with a warmup field excluded.

RePaint is timed at stride 5 -- the setting at which it TIES CorrDiff on
accuracy -- because that is the honest basis for the speed comparison, not the
stride-1 worst case.
"""
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddpm_library import CorrDiff, DistAttn, StreamDDPM, RePaint, VCNN, GP
from ddpm_library import config as C

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--device", default="auto",
                help="torch device for every model except GP, which is CPU-only")
ap.add_argument("--repeats", type=int, default=5, help="timed fields, warmup excluded")
args = ap.parse_args()
DEV = args.device

bench = np.load(ROOT / "benchmark/ocean_bench_v1.npz")
obs_all, priors_all = bench["observations"], bench["priors"]
N = args.repeats


def fresh(rows, hours):
    ages = (rows[:, 2].max() - rows[:, 2]) / 3600.0
    return rows[ages <= hours]


MODELS = {
    "corrdiff (1h, 20 draws)": (CorrDiff(device=DEV), lambda m, r, p, i: m.predict(
        [tuple(x) for x in fresh(r, 1.0)], p, n_draws=20, seed=i,
        sigma_scale=C.CORRDIFF_SIGMA_SCALE_TIMED)),
    "distattn (10 draws)": (DistAttn(device=DEV), lambda m, r, p, i: m.predict(
        [tuple(x) for x in r], n_draws=10, seed=i)),
    "stream (full field, 20 draws)": (StreamDDPM(device=DEV), lambda m, r, p, i: m.predict(
        [tuple(x) for x in r], p, n_draws=20, seed=i, full_field=True)),
    "repaint (1h, 10 draws, stride 5)": (RePaint(device=DEV), lambda m, r, p, i: m.predict(
        [tuple(x) for x in fresh(r, 1.0)], p, n_draws=10, stride=5, seed=i)),
    "vcnn": (VCNN(device=DEV), lambda m, r, p, i: m.predict([tuple(x) for x in r])),
    "gp (cpu)": (GP(), lambda m, r, p, i: m.predict([tuple(x) for x in r])),
}

def _sync():
    """Block until queued device work is done, or timings measure queueing only."""
    if torch.cuda.is_available():
        _sync()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.synchronize()


print(f"{'model':<34}{'s/field':>9}  (mean of {N}, warmup excluded)")
for name, (mdl, fn) in MODELS.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn(mdl, obs_all[0], list(priors_all[0]), 0)         # warmup / compile
        _sync()
        t0 = time.time()
        for i in range(1, N + 1):
            fn(mdl, obs_all[i], list(priors_all[i]), i)
        _sync()
    print(f"{name:<34}{(time.time() - t0) / N:>9.1f}", flush=True)
print("TIMING_DONE")
