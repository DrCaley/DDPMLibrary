"""What does Stream's spread term actually buy?

The spread term (lambda_spr, the 4th direction-loss term) exists to make the
model's uncertainty map correlate with where it is actually wrong. A matched
ablation showed it also degrades the vorticity field badly: at 1000 fine-tune
steps, turning it OFF improves vorticity RMSE 40% and vorticity correlation
+0.185, and drops single-draw vorticity amplitude from 1.9x to 1.14x truth
(results_stream_spread_ablation.pt).

So the question is whether that is a trade or a pure cost. This scores the
matched pair on the calibration metrics the paper actually reports -- CRPS,
raw coverage, the fitted conformal factor, calibrated coverage and interval
width -- using the same definitions and the same split-conformal protocol as
benchmark/uncertainty_final.py.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from _paths import DEV, MODELS_DIR  # noqa: E402
from ddpm_library import StreamDDPM, metrics                 # noqa: E402

LEVEL, SEED = 0.90, 20260830
z = 1.6448536269514722
MD = MODELS_DIR / "stream_vort"
ARMS = {
    "shipped (spread on)":    ROOT / "src/ddpm_library/assets/stream_dir_weights.pt",
    "1000 steps, spread ON":  MD / "spread_s1000.pt",
    "1000 steps, spread OFF": MD / "lam00_s1000.pt",
}

b = np.load(ROOT / "benchmark" / "ocean_bench_v1.npz")
obs_all, priors_all, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
n = len(truth)
half = n // 2

rows, per_case = {}, {}
for name, path in ARMS.items():
    st = StreamDDPM(device=DEV, dir_weights_path=path)
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = st.predict([tuple(x) for x in obs_all[i]], priors_all[i],
                              n_draws=20, seed=SEED + i, full_field=True, calibrate=False)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    m, s = np.stack(M), np.stack(S)
    del st

    crps = np.array([metrics.crps_gaussian(m[i], s[i], truth[i], ocean_mask=ocean)
                     for i in range(n)])
    cov = np.array([metrics.coverage(m[i], s[i], truth[i], level=LEVEL,
                                     ocean_mask=ocean) for i in range(n)])
    ssr = np.array([metrics.spread_skill_ratio(m[i], s[i], truth[i], ocean_mask=ocean)
                    for i in range(n)])
    # per-case correlation of predicted sigma against realised error -- this is
    # the quantity the spread term is designed to raise.
    r_unc = np.array([np.corrcoef(s[i][ocean].ravel(),
                                  np.abs(m[i] - truth[i])[ocean].ravel())[0, 1]
                      for i in range(n)])
    err, sg = np.abs(m - truth)[:, ocean], s[:, ocean]
    ok = sg[:half] > 1e-9
    scale = float(np.quantile(err[:half][ok] / (z * sg[:half][ok]), LEVEL))
    cov_v = float(np.mean(err[half:] <= z * scale * sg[half:]))
    width = float(np.mean(2 * z * scale * sg[half:]))
    # CRPS on the CALIBRATED sigma -- the apples-to-apples version. On raw
    # sigma the comparison is confounded: both arms are badly under-dispersed
    # (coverage ~0.5 against a 0.90 target), so a wider raw spread scores
    # better on CRPS for the wrong reason.
    crps_cal = np.array([metrics.crps_gaussian(m[i], scale * s[i], truth[i],
                                               ocean_mask=ocean) for i in range(n)])
    rows[name] = dict(crps=crps.mean(), crps_cal=crps_cal.mean(),
                      cov_raw=cov.mean(), ssr=ssr.mean(),
                      r_unc=r_unc.mean(), factor=scale,
                      cov_cal=cov_v, width=width)
    per_case[name] = dict(crps=crps, crps_cal=crps_cal, cov_raw=cov, ssr=ssr,
                          r_unc=r_unc)
    print(f"  scored {name}", flush=True)

print(f"\n{'arm':<26}{'CRPSraw':>9}{'CRPScal':>9}{'cov raw':>9}{'r(sig,err)':>12}"
      f"{'factor':>9}{'cov cal':>9}{'width':>8}")
for k, r in rows.items():
    print(f"{k:<26}{r['crps']:>9.4f}{r['crps_cal']:>9.4f}{r['cov_raw']:>9.4f}"
          f"{r['r_unc']:>12.3f}{r['factor']:>9.3f}{r['cov_cal']:>9.4f}"
          f"{r['width']:>8.4f}")

rng = np.random.default_rng(0)
print("\nspread OFF - spread ON, matched at 1000 steps (paired over cases):")
for k in ("crps", "crps_cal", "cov_raw", "ssr", "r_unc"):
    d = per_case["1000 steps, spread OFF"][k] - per_case["1000 steps, spread ON"][k]
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(6000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"   {k:<10}{d.mean():+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  "
          f"{'SIGNIFICANT' if lo * hi > 0 else 'TIED'}")

torch.save({"rows": rows, "per_case": per_case,
            "meta": {"seed": SEED, "level": LEVEL, "n_cases": n, "n_draws": 20,
                     "benchmark_md5": "44b866540296490c615be13bacb4242e"}},
           ROOT / "benchmark" / "results_stream_spread_calibration.pt")
print("\nsaved benchmark/results_stream_spread_calibration.pt")
