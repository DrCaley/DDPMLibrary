"""Uncertainty quality for the three paper models on the realistic benchmark.

The paper's central claim for CorrDiff is calibrated uncertainty, but CRPS and
coverage were only ever measured on the idealised simultaneous-observation task.
This is the missing column of the final table, on the same frozen benchmark as
everything else.

Fairness: every model also gets its own split-conformal calibration (factor fitted
on 20 cases, coverage verified on the other 20). "Calibrated" alone is cheap --
widening intervals until coverage hits the level always works -- so the meaningful
comparison is SHARPNESS AT MATCHED CALIBRATION: who needs the narrowest intervals
to be honest. CorrDiff uses its shipped timed factor up front; the others' raw
spreads are documented as uncalibrated, so the conformal pass gives each its best
shot rather than penalising them for not shipping a factor.
"""
import sys, warnings, hashlib
from pathlib import Path
import numpy as np, torch
from scipy.stats import norm

import os                                                          # noqa: E402
#: "auto" resolves cuda / mps / cpu, so these run off the GPU box too.
DEV = os.environ.get("DDPM_DEVICE", "auto")

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmark"))
import score
from ddpm_library import CorrDiff, DistAttn, StreamDDPM, metrics
from ddpm_library import config as C

BENCH = ROOT / "benchmark/ocean_bench_v1.npz"
OUT = ROOT / "benchmark/results_uncertainty_final.pt"
SEED, LEVEL = 20260830, 0.90
z = float(norm.ppf(0.5 + LEVEL / 2.0))

bench = np.load(BENCH)
obs_all, priors_all, truth = bench["observations"], bench["priors"], bench["truth"]
ocean = np.asarray(bench["ocean_mask"], bool); n = len(truth)
cd, da, st = CorrDiff(device=DEV), DistAttn(device=DEV), StreamDDPM(device=DEV)


def fresh(rows, hours):
    ages = (rows[:, 2].max() - rows[:, 2]) / 3600.0
    return rows[ages <= hours]


# Each model at the configuration the accuracy table uses. CorrDiff's sigma uses
# the shipped TIMED factor (2.1801; the refit for the 1 h cutoff gave 2.2006).
MODELS = {
    "corrdiff (1h, timed factor)": lambda r, p, i: cd.predict(
        [tuple(x) for x in fresh(r, 1.0)], p, n_draws=20, seed=SEED + i,
        sigma_scale=C.CORRDIFF_SIGMA_SCALE_TIMED, calibrate=False),
    "distattn (raw spread)": lambda r, p, i: da.predict(
        [tuple(x) for x in r], n_draws=C.DISTATTN_DEFAULT_N_DRAWS, seed=SEED + i, calibrate=False),
    "stream (raw spread)": lambda r, p, i: st.predict(
        [tuple(x) for x in r], p, n_draws=20, seed=SEED + i, full_field=True, calibrate=False),
    # Untested prediction of the age-cutoff rule: Stream carries the 13/25 h
    # priors, and every prior-carrying model measured so far gains from the 1 h
    # discard (corrdiff -8.2%, repaint -14%) while every prior-less model loses.
    "stream (1h cutoff)": lambda r, p, i: st.predict(
        [tuple(x) for x in fresh(r, 1.0)], p, n_draws=20, seed=SEED + i,
        full_field=True, calibrate=False),
}

means, sigmas = {}, {}
for name, fn in MODELS.items():
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = fn(obs_all[i], list(priors_all[i]), i)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    means[name] = np.stack(M); sigmas[name] = np.stack(S)
    print(f"  scored {name}", flush=True)

rows, per_case = {}, {}
half = n // 2
for name in MODELS:
    m, s = means[name], sigmas[name]
    crps = np.array([metrics.crps_gaussian(m[i], s[i], truth[i], ocean_mask=ocean) for i in range(n)])
    cov  = np.array([metrics.coverage(m[i], s[i], truth[i], level=LEVEL, ocean_mask=ocean) for i in range(n)])
    ssr  = np.array([metrics.spread_skill_ratio(m[i], s[i], truth[i], ocean_mask=ocean) for i in range(n)])
    # split conformal on this model's own sigma: fit half, verify half
    err = np.abs(m - truth)[:, ocean]; sg = s[:, ocean]
    ok = sg[:half] > 1e-9
    scale = float(np.quantile((err[:half][ok] / (z * sg[:half][ok])), LEVEL))
    cov_v = float(np.mean(err[half:] <= z * scale * sg[half:]))
    width = float(np.mean(2 * z * scale * sg[half:]))
    rows[name] = {"crps": crps.mean(), "coverage_raw": cov.mean(), "ssr": ssr.mean(),
                  "conformal_factor": scale, "coverage_conformal": cov_v,
                  "width_conformal": width}
    per_case[name] = {"crps": crps, "coverage_raw": cov, "ssr": ssr}

print(f"\n{'model':<30}{'CRPS':>9}{'cov raw':>9}{'SSR':>7}"
      f"{'factor':>9}{'cov cal':>9}{'width':>8}")
for k, r in rows.items():
    print(f"{k:<30}{r['crps']:>9.4f}{r['coverage_raw']:>9.4f}{r['ssr']:>7.2f}"
          f"{r['conformal_factor']:>9.3f}{r['coverage_conformal']:>9.4f}"
          f"{r['width_conformal']:>8.4f}")

names = list(MODELS)
print(f"\npaired CRPS differences:")
for i, a in enumerate(names):
    for b in names[i+1:]:
        d, lo, hi = score.bootstrap_ci(per_case[a]["crps"] - per_case[b]["crps"])
        t = "significant" if (lo > 0) == (hi > 0) else "TIED"
        print(f"  {a.split(' ')[0]} - {b.split(' ')[0]}: {d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  {t}")

rmse_pc = {k: np.array([np.sqrt((((means[k][i]-truth[i])**2).sum(-1))[ocean].mean())
                        for i in range(n)]) for k in MODELS}
d, lo, hi = score.bootstrap_ci(rmse_pc["stream (1h cutoff)"] - rmse_pc["stream (raw spread)"])
t = "significant" if (lo > 0) == (hi > 0) else "TIED"
print(f"\nstream 1h cutoff vs full track [rmse_vector]: {d:+.5f}  "
      f"CI [{lo:+.5f}, {hi:+.5f}]  {t}")
print(f"  full track {rmse_pc['stream (raw spread)'].mean():.4f}  "
      f"1h cutoff {rmse_pc['stream (1h cutoff)'].mean():.4f}")

torch.save({"meta": {"seed": SEED, "level": LEVEL, "z": z, "n_cases": n,
                     "benchmark_md5": hashlib.md5(BENCH.read_bytes()).hexdigest(),
                     "corrdiff_sigma_scale": float(C.CORRDIFF_SIGMA_SCALE_TIMED)},
            "means": {k: torch.from_numpy(v) for k, v in means.items()},
            "sigmas": {k: torch.from_numpy(v) for k, v in sigmas.items()},
            "summary": rows,
            "rmse_per_case": {k: torch.from_numpy(v) for k, v in rmse_pc.items()},
            "per_case": {k: {m_: torch.from_numpy(v_) for m_, v_ in d.items()}
                         for k, d in per_case.items()}}, OUT)
print(f"\nwrote {OUT}")
