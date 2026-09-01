"""Locate the age-cutoff optimum, and refit the conformal factor for THIS regime.

Two things, one pass over the models, because both need the same predictions.

1. AGE CUTOFF. Discarding observations older than ~1 h was worth about 10% RMSE
   to CorrDiff-with-priors, but 1 h only beat 0.5 h -- the optimum was bracketed,
   never located. This sweeps it properly.

   Only CorrDiff+priors is swept. The cutoff was already measured to HURT
   distattn (+15.5%) and gp (+10.9%) significantly: a model with no independent
   temporal prior has nothing to fall back on, so a stale reading still beats no
   reading. Those are scored at the full track, which is their own optimum, so
   the head-to-head compares every model at its best configuration.

2. CONFORMAL REFIT. The shipped CORRDIFF_SIGMA_SCALE was fitted for SIMULTANEOUS
   observations. Everything we now intend to report is 2 h time-varying
   collection with a discard rule -- a different observation process, so the
   intervals are calibrated for the wrong task. Refit by split conformal: fit the
   factor on half the cases, verify coverage on the held-out half.

Runs on ocean_bench_v1 (frozen, SEED 20260829) so it is reproducible, and saves
raw predictions and sigmas to .pt so any of this can be re-scored without
re-running the models.
"""
import sys, warnings, hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, "/workspace/DDPMLibrary/src")
sys.path.insert(0, "/workspace/DDPMLibrary/benchmark")
import score
from ddpm_library import CorrDiff, DistAttn, GP, VCNN

BENCH = Path("/workspace/DDPMLibrary/benchmark/ocean_bench_v1.npz")
OUT = Path("/workspace/DDPMLibrary/benchmark/results_age_calibration.pt")
SEED = 20260830
CUTOFFS_H = (0.5, 0.75, 1.0, 1.25, 1.5)          # plus the full 2 h track
LEVEL = 0.90

bench = np.load(BENCH)
obs_all, priors_all, truth = bench["observations"], bench["priors"], bench["truth"]
ocean = bench["ocean_mask"]
n_cases = len(truth)

cd, da, gp, vc = (CorrDiff(device="cuda"), DistAttn(device="cuda"), GP(),
                  VCNN(device="cuda"))


def subset(rows, cutoff_h):
    """Readings no older than cutoff_h relative to the newest one."""
    ages = (rows[:, 2].max() - rows[:, 2]) / 3600.0
    return rows[ages <= cutoff_h] if cutoff_h is not None else rows


def predict(model, rows, priors, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs = [tuple(r) for r in rows]
        args = (obs, priors) if priors is not None else (obs,)
        return model.predict(*args, **kw)


# --------------------------------------------------------------------------- #
# 1. CorrDiff+priors across cutoffs. calibrate=False keeps the RAW ensemble
#    spread, which is what the conformal factor must be refitted against.
# --------------------------------------------------------------------------- #
configs = {f"corrdiff+priors@{c}h": c for c in CUTOFFS_H}
configs["corrdiff+priors@full"] = None

means, sigmas, kept = {}, {}, {}
for name, cutoff in configs.items():
    M, S, K = [], [], []
    for i in range(n_cases):
        rows = subset(obs_all[i], cutoff)
        K.append(len(rows))
        m, s = predict(cd, rows, list(priors_all[i]), n_draws=20, seed=SEED + i,
                       calibrate=False)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    means[name] = np.stack(M); sigmas[name] = np.stack(S); kept[name] = np.array(K)
    print(f"  {name:<26} {np.mean(K):>6.0f} readings/case", flush=True)

# Comparators, each at the full track (their own optimum).
for name, mdl, pri, kw in (("distattn@full", da, False, {"n_draws": 10}),
                           ("vcnn@full", vc, False, {}),
                           ("gp@full", gp, False, {})):
    M = []
    for i in range(n_cases):
        kw2 = dict(kw)
        if name.startswith("distattn"):
            kw2["seed"] = SEED + i
        m, _ = predict(mdl, obs_all[i], None, **kw2)
        M.append(np.asarray(m, np.float32))
    means[name] = np.stack(M); kept[name] = np.full(n_cases, len(obs_all[0]))
    print(f"  {name:<26} {len(obs_all[0]):>6d} readings/case", flush=True)

scored = {n: score.score_model(m, bench) for n, m in means.items()}

print(f"\n{'config':<26}{'readings':>10}{'RMSE':>10}{'angle_rms':>12}{'eddy-ish':>10}")
for n in means:
    s = scored[n]
    print(f"{n:<26}{np.mean(kept[n]):>10.0f}{s['rmse_vector'].mean():>10.4f}"
          f"{s['angle_rms_rad'].mean():>12.4f}{'':>10}")

base = scored["corrdiff+priors@full"]["rmse_vector"]
print(f"\npaired vs the full 2 h track (negative = discarding helps):")
best_name, best_mean = "corrdiff+priors@full", base.mean()
for n in configs:
    if n.endswith("@full"):
        continue
    d, lo, hi = score.bootstrap_ci(scored[n]["rmse_vector"] - base)
    tag = "significant" if (lo > 0) == (hi > 0) else "tied"
    print(f"  {n:<26}{d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"({100 * d / base.mean():+.1f}%)  {tag}")
    if scored[n]["rmse_vector"].mean() < best_mean:
        best_name, best_mean = n, scored[n]["rmse_vector"].mean()
print(f"\n  best cutoff: {best_name}  (RMSE {best_mean:.4f})")

# --------------------------------------------------------------------------- #
# 2. Split-conformal refit, same procedure as scripts/staleness.py recalibrate.
# --------------------------------------------------------------------------- #
z = float(norm.ppf(0.5 + LEVEL / 2.0))
half = n_cases // 2
o = np.asarray(ocean, bool)
conformal = {}
print(f"\nconformal refit (level {LEVEL}, z={z:.4f}), fit on {half} cases, "
      f"verify on {n_cases - half}:")
print(f"  {'config':<26}{'factor':>9}{'cov fit':>10}{'cov held out':>14}{'width':>10}")
for n in configs:
    err = np.abs(means[n] - truth)[:, o]                 # (cases, cells, 2)
    sig = sigmas[n][:, o]
    fit_e, fit_s = err[:half].ravel(), sig[:half].ravel()
    ver_e, ver_s = err[half:].ravel(), sig[half:].ravel()
    ok = fit_s > 1e-9
    scale = float(np.quantile(fit_e[ok] / (z * fit_s[ok]), LEVEL))
    cov_f = float(np.mean(fit_e <= z * scale * fit_s))
    cov_v = float(np.mean(ver_e <= z * scale * ver_s))
    width = float(np.mean(2 * z * scale * ver_s))
    conformal[n] = {"scale": scale, "coverage_fit": cov_f,
                    "coverage_holdout": cov_v, "mean_width": width}
    print(f"  {n:<26}{scale:>9.4f}{cov_f:>10.4f}{cov_v:>14.4f}{width:>10.4f}")

torch.save({
    "meta": {
        "seed": SEED, "level": LEVEL, "z": z,
        "benchmark": BENCH.name,
        "benchmark_md5": hashlib.md5(BENCH.read_bytes()).hexdigest(),
        "n_cases": n_cases, "cutoffs_h": CUTOFFS_H,
        "ocean_cells": int(o.sum()),
        "note": "CorrDiff run with calibrate=False; sigmas are RAW ensemble spread.",
    },
    "means": {k: torch.from_numpy(v) for k, v in means.items()},
    "sigmas": {k: torch.from_numpy(v) for k, v in sigmas.items()},
    "readings_per_case": {k: torch.from_numpy(v) for k, v in kept.items()},
    "per_case_metrics": {k: {m: torch.from_numpy(np.asarray(v))
                             for m, v in s.items()} for k, s in scored.items()},
    "conformal": conformal,
    "best_cutoff": best_name,
}, OUT)
print(f"\nwrote {OUT}")
