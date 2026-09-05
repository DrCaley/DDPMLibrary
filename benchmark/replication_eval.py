"""Replication on a fresh, disjoint benchmark -- and RePaint's full assessment.

ocean_bench_v1b: 40 new cases, zero frame overlap with v1, fresh generation seed,
fresh diffusion seeds. One run answers four reviewer questions at once:

  1. Does the headline ranking replicate on independent cases?
  2. Are the results seed-sensitive? (every random choice here is new)
  3. Do the SHIPPED conformal factors hold out-of-sample? (corrdiff 2.1801,
     distattn 1.3592, stream 3.141 -- fitted on v1, applied blind to v1b)
  4. Is RePaint a contender? It ties CorrDiff on accuracy but has never had an
     uncertainty column. Here it gets sigma, CRPS, and its own conformal fit
     (split within v1b, since no v1 fit exists for it).

CRPS is computed on CALIBRATED sigmas for every model (shipped factors), which
fixes an inconsistency in the v1 table where CorrDiff's CRPS used its calibrated
sigma while DistAttn's and Stream's used raw. Raw-sigma CRPS is also saved.
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
from ddpm_library import CorrDiff, DistAttn, StreamDDPM, RePaint, metrics
from ddpm_library import config as C

BENCH = ROOT / "benchmark/ocean_bench_v1b.npz"
OUT = ROOT / "benchmark/results_replication.pt"
SEED, LEVEL = 20260901, 0.90          # fresh diffusion seeds too
z = float(norm.ppf(0.5 + LEVEL / 2.0))

bench = np.load(BENCH)
obs_all, priors_all, truth = bench["observations"], bench["priors"], bench["truth"]
ocean = np.asarray(bench["ocean_mask"], bool); n = len(truth)
cd, da = CorrDiff(device=DEV), DistAttn(device=DEV)
st, rp = StreamDDPM(device=DEV), RePaint(device=DEV)


def fresh(rows, hours):
    ages = (rows[:, 2].max() - rows[:, 2]) / 3600.0
    return rows[ages <= hours]


# name -> (fn, shipped factor applied to the returned sigma; corrdiff's predict
# already applies its factor internally via sigma_scale, so 1.0 here)
MODELS = {
    "corrdiff (1h)": (lambda r, p, i: cd.predict(
        [tuple(x) for x in fresh(r, 1.0)], p, n_draws=20, seed=SEED + i,
        sigma_scale=C.CORRDIFF_SIGMA_SCALE_TIMED, calibrate=False), 1.0),
    "distattn": (lambda r, p, i: da.predict(
        [tuple(x) for x in r], n_draws=C.DISTATTN_DEFAULT_N_DRAWS, seed=SEED + i, calibrate=False),
        float(C.DISTATTN_SIGMA_SCALE_TIMED)),
    "stream (full field)": (lambda r, p, i: st.predict(
        [tuple(x) for x in r], p, n_draws=20, seed=SEED + i, full_field=True, calibrate=False),
        float(C.STREAM_SIGMA_SCALE_TIMED)),
    "repaint (1h)": (lambda r, p, i: rp.predict(
        [tuple(x) for x in fresh(r, 1.0)], p, n_draws=10, stride=5, seed=SEED + i),
        None),   # no shipped factor: fitted within v1b, split-conformal
}

means, sigmas = {}, {}
for name, (fn, _) in MODELS.items():
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = fn(obs_all[i], list(priors_all[i]), i)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
        if (i + 1) % 10 == 0:
            print(f"    {name}: {i+1}/{n}", flush=True)
    means[name] = np.stack(M); sigmas[name] = np.stack(S)
    print(f"  scored {name}", flush=True)

half = n // 2
rows, pc = {}, {}
for name, (_, factor) in MODELS.items():
    m, s_raw = means[name], sigmas[name]
    err = np.abs(m - truth)[:, ocean]
    if factor is None:
        sg = s_raw[:, ocean]; ok = sg[:half] > 1e-9
        factor = float(np.quantile(err[:half][ok] / (z * sg[:half][ok]), LEVEL))
        cov_note = "fit on half of v1b, verified on the other half"
        ver = slice(half, None)
    else:
        cov_note = "SHIPPED factor, applied blind to all of v1b"
        ver = slice(None)
    s_cal = s_raw * factor
    rmse = np.array([np.sqrt((((m[i]-truth[i])**2).sum(-1))[ocean].mean()) for i in range(n)])
    ang  = np.array([score.case_metrics(m[i], truth[i], ocean)["angle_rms_rad"] for i in range(n)])
    crps = np.array([metrics.crps_gaussian(m[i], s_cal[i], truth[i], ocean_mask=ocean) for i in range(n)])
    cov  = float(np.mean(err[ver] <= z * (s_cal[:, ocean])[ver]))
    width = float(np.mean(2 * z * (s_cal[:, ocean])[ver]))
    rows[name] = {"rmse": rmse.mean(), "angle": ang.mean(), "crps_cal": crps.mean(),
                  "factor": factor, "coverage": cov, "width": width, "note": cov_note}
    pc[name] = {"rmse": rmse, "angle": ang, "crps": crps}

print(f"\nREPLICATION SET (v1b, {n} fresh cases, no frame overlap with v1)")
print(f"{'model':<22}{'RMSE':>9}{'angle':>9}{'CRPS*':>9}{'factor':>9}{'cov@90':>9}{'width':>8}")
for k, r in rows.items():
    print(f"{k:<22}{r['rmse']:>9.4f}{r['angle']:>9.4f}{r['crps_cal']:>9.4f}"
          f"{r['factor']:>9.3f}{r['coverage']:>9.4f}{r['width']:>8.4f}")
print("* CRPS on calibrated sigma for every model")
for k, r in rows.items():
    print(f"  {k}: coverage is {r['note']}")

v1 = {"corrdiff (1h)": 0.0618, "distattn": 0.0738, "stream (full field)": 0.0908,
      "repaint (1h)": 0.0595}
print(f"\nv1 -> v1b RMSE:")
for k in MODELS:
    print(f"  {k:<22}{v1[k]:.4f} -> {rows[k]['rmse']:.4f}   ({100*(rows[k]['rmse']/v1[k]-1):+.1f}%)")

print(f"\npaired on v1b:")
for a, b_ in (("corrdiff (1h)", "distattn"), ("corrdiff (1h)", "stream (full field)"),
              ("corrdiff (1h)", "repaint (1h)"), ("repaint (1h)", "distattn")):
    for mtr in ("rmse", "crps"):
        d, lo, hi = score.bootstrap_ci(pc[a][mtr] - pc[b_][mtr])
        t = "significant" if (lo > 0) == (hi > 0) else "TIED"
        print(f"  {a} - {b_} [{mtr}]: {d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  {t}")

torch.save({"meta": {"seed": SEED, "level": LEVEL, "n_cases": n,
                     "benchmark": BENCH.name,
                     "benchmark_md5": hashlib.md5(BENCH.read_bytes()).hexdigest(),
                     "shipped_factors": {"corrdiff": float(C.CORRDIFF_SIGMA_SCALE_TIMED),
                                          "distattn": float(C.DISTATTN_SIGMA_SCALE_TIMED),
                                          "stream": float(C.STREAM_SIGMA_SCALE_TIMED)}},
            "means": {k: torch.from_numpy(v) for k, v in means.items()},
            "sigmas_raw": {k: torch.from_numpy(v) for k, v in sigmas.items()},
            "summary": rows,
            "per_case": {k: {m_: torch.from_numpy(v_) for m_, v_ in d.items()}
                         for k, d in pc.items()}}, OUT)
print(f"\nwrote {OUT}")
