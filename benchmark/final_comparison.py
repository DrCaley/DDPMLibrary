"""The comparison table for the paper: every model at its own best configuration.

Two corrections over previous runs.

1. STREAM RUNS WITH full_field=True. Its output is otherwise exactly
   divergence-free, which costs it 12.5% RMSE on a field whose divergent
   component carries 14% of the energy. Every Stream number we have published so
   far used the constrained default and understates it.

2. EDDY RECALL IS REPORTED TWICE. Okubo-Weiss is strain^2 - vorticity^2, so
   adding a curl-free component raises strain and pushes cells out of the
   rotation-dominated class WITHOUT changing vorticity at all (curl of a gradient
   is identically zero). Measured directly: restoring Stream's divergent part
   left vorticity unchanged (-0.5%, numerical) while strain rose 1.7% and eddy
   recall fell 7.8%. So the standard metric penalises a model for representing
   divergence correctly, and rewards divergence-free models with an artefact.
   `eddy_rot` re-runs the same detector after Helmholtz-projecting BOTH the
   prediction and the truth, so neither side carries divergence and the bias
   cancels.

Each model is scored at its own optimum, which is the honest protocol: CorrDiff
gains from discarding readings older than 1 h and the others measurably lose by
it, so forcing one observation policy on all of them would favour whichever
policy was chosen.
"""
import sys, warnings, hashlib
from pathlib import Path
import numpy as np
import torch

import os                                                          # noqa: E402
#: "auto" resolves cuda / mps / cpu, so these run off the GPU box too.
DEV = os.environ.get("DDPM_DEVICE", "auto")

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmark"))
import score
from ddpm_library import CorrDiff, DistAttn, StreamDDPM, GP, VCNN, metrics
from ddpm_library.stream.conditioning import helmholtz_project

BENCH = ROOT / "benchmark/ocean_bench_v1.npz"
OUT = ROOT / "benchmark/results_final_comparison.pt"
SEED = 20260830
CUTOFF_H = 1.0                      # CorrDiff's measured optimum

bench = np.load(BENCH)
obs_all, priors_all, truth = bench["observations"], bench["priors"], bench["truth"]
ocean = np.asarray(bench["ocean_mask"], bool)
n = len(truth)

cd = CorrDiff(device=DEV); da = DistAttn(device=DEV)
st = StreamDDPM(device=DEV); gp = GP(); vc = VCNN(device=DEV)

# name -> (callable(obs_rows, priors, i) -> mean, description)
def fresh(rows, hours):
    ages = (rows[:, 2].max() - rows[:, 2]) / 3600.0
    return rows[ages <= hours]

MODELS = {
    "corrdiff (1h cutoff)": lambda r, p, i: cd.predict(
        [tuple(x) for x in fresh(r, CUTOFF_H)], p, n_draws=20, seed=SEED + i)[0],
    "corrdiff (full track)": lambda r, p, i: cd.predict(
        [tuple(x) for x in r], p, n_draws=20, seed=SEED + i)[0],
    "distattn": lambda r, p, i: da.predict(
        [tuple(x) for x in r], n_draws=10, seed=SEED + i)[0],
    "stream (+divergent)": lambda r, p, i: st.predict(
        [tuple(x) for x in r], p, n_draws=20, seed=SEED + i, full_field=True)[0],
    "stream (divfree only)": lambda r, p, i: st.predict(
        [tuple(x) for x in r], p, n_draws=20, seed=SEED + i, full_field=False)[0],
    "vcnn": lambda r, p, i: vc.predict([tuple(x) for x in r])[0],
    "gp": lambda r, p, i: gp.predict([tuple(x) for x in r])[0],
}


def rot_part(f):
    """Rotational (divergence-free) component, library orientation."""
    m = np.ascontiguousarray(np.transpose(f, (2, 1, 0)))        # (2,94,44)
    r = helmholtz_project(m, ocean.T, max_iters=30, tol=1e-7)
    return np.ascontiguousarray(np.transpose(r, (2, 1, 0)))     # (44,94,2)


truth_rot = np.stack([rot_part(t) for t in truth])

means, eddy_raw, eddy_rot = {}, {}, {}
for name, fn in MODELS.items():
    M, ER, EO = [], [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.asarray(fn(obs_all[i], list(priors_all[i]), i), np.float32)
        M.append(m)
        ER.append(metrics.eddy_hit_rate(m, truth[i], ocean))
        EO.append(metrics.eddy_hit_rate(rot_part(m), truth_rot[i], ocean))
    means[name] = np.stack(M)
    eddy_raw[name] = np.array(ER); eddy_rot[name] = np.array(EO)
    print(f"  scored {name}", flush=True)

S = {k: score.score_model(v, bench) for k, v in means.items()}

print(f"\n{'model':<24}{'RMSE':>9}{'angle_rms':>11}{'eddy':>8}{'eddy_rot':>10}")
for k in MODELS:
    print(f"{k:<24}{S[k]['rmse_vector'].mean():>9.4f}{S[k]['angle_rms_rad'].mean():>11.4f}"
          f"{np.nanmean(eddy_raw[k]):>8.4f}{np.nanmean(eddy_rot[k]):>10.4f}")

def cmp(a, b):
    print(f"\n{a}  vs  {b}:")
    for m in ("rmse_vector", "angle_rms_rad"):
        d, lo, hi = score.bootstrap_ci(S[a][m] - S[b][m])
        t = "significant" if (lo > 0) == (hi > 0) else "TIED"
        print(f"  {m:<16}{d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  {t}")
    for nm, src in (("eddy", eddy_raw), ("eddy_rot", eddy_rot)):
        x, y = src[a], src[b]
        ok = ~(np.isnan(x) | np.isnan(y))
        d, lo, hi = score.bootstrap_ci(x[ok] - y[ok])
        t = "significant" if (lo > 0) == (hi > 0) else "TIED"
        print(f"  {nm:<16}{d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  {t}")

cmp("corrdiff (1h cutoff)", "distattn")
cmp("corrdiff (1h cutoff)", "stream (+divergent)")
cmp("stream (+divergent)", "stream (divfree only)")
cmp("distattn", "stream (+divergent)")

torch.save({"meta": {"seed": SEED, "benchmark": BENCH.name, "cutoff_h": CUTOFF_H,
                     "benchmark_md5": hashlib.md5(BENCH.read_bytes()).hexdigest(),
                     "n_cases": n},
            "means": {k: torch.from_numpy(v) for k, v in means.items()},
            "per_case_metrics": {k: {m: torch.from_numpy(np.asarray(v))
                                     for m, v in s.items()} for k, s in S.items()},
            "eddy_raw": {k: torch.from_numpy(v) for k, v in eddy_raw.items()},
            "eddy_rot": {k: torch.from_numpy(v) for k, v in eddy_rot.items()}}, OUT)
print(f"\nwrote {OUT}")
