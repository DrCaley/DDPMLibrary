"""Replicate the Stream spread-term ablation on the independent benchmark.

Same matched pair (1000 fine-tune steps, spread ON vs OFF, lambda_vort=0) scored
on ocean_bench_v1b -- 40 fresh cases, zero frame overlap with v1, fresh
generation seed. Nothing about the arms was chosen using v1b.
"""
import sys, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddpm_library import StreamDDPM, metrics                 # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DEV, MODELS_DIR  # noqa: E402
from _vorticity import curl  # noqa: E402

LEVEL, SEED, z = 0.90, 20260901, 1.6448536269514722
MD = MODELS_DIR / "stream_vort"
ARMS = {"1000 steps, spread ON":  MD / "spread_s1000.pt",
        "1000 steps, spread OFF": MD / "lam00_s1000.pt"}

b = np.load(ROOT / "benchmark" / "ocean_bench_v1b.npz")
obs_all, priors_all, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
Mk = np.zeros_like(ocean); Mk[1:-1, 1:-1] = True; Mk &= ocean
n, half = len(truth), len(truth) // 2


rows, per_case = {}, {}
for name, path in ARMS.items():
    st = StreamDDPM(device=DEV, dir_weights_path=path)
    M, S, vr, vc = [], [], [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = st.predict([tuple(x) for x in obs_all[i]], priors_all[i],
                              n_draws=20, seed=SEED + i, full_field=True, calibrate=False)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
        p, t = curl(np.asarray(m))[Mk], curl(truth[i])[Mk]
        vr.append(np.sqrt(((p - t) ** 2).mean())); vc.append(np.corrcoef(p, t)[0, 1])
    m, s = np.stack(M), np.stack(S); del st
    rmse = np.array([np.sqrt(((m[i] - truth[i])[ocean] ** 2).sum(-1).mean())
                     for i in range(n)])
    err, sg = np.abs(m - truth)[:, ocean], s[:, ocean]
    ok = sg[:half] > 1e-9
    scale = float(np.quantile(err[:half][ok] / (z * sg[:half][ok]), LEVEL))
    crps_cal = np.array([metrics.crps_gaussian(m[i], scale * s[i], truth[i],
                                               ocean_mask=ocean) for i in range(n)])
    r_unc = np.array([np.corrcoef(s[i][ocean].ravel(),
                                  np.abs(m[i] - truth[i])[ocean].ravel())[0, 1]
                      for i in range(n)])
    per_case[name] = dict(rmse=rmse, vort_rmse=np.array(vr), vort_corr=np.array(vc),
                          crps_cal=crps_cal, r_unc=r_unc)
    rows[name] = dict(rmse=rmse.mean(), vort_rmse=np.mean(vr), vort_corr=np.mean(vc),
                      crps_cal=crps_cal.mean(), r_unc=r_unc.mean(), factor=scale,
                      cov_cal=float(np.mean(err[half:] <= z * scale * sg[half:])),
                      width=float(np.mean(2 * z * scale * sg[half:])))
    print(f"  scored {name}", flush=True)

print(f"\n{'arm':<26}{'RMSE':>8}{'vortRMSE':>10}{'vortCorr':>10}{'CRPScal':>9}"
      f"{'r(sig,err)':>12}{'factor':>8}{'covcal':>8}{'width':>8}")
for k, r in rows.items():
    print(f"{k:<26}{r['rmse']:>8.5f}{r['vort_rmse']:>10.5f}{r['vort_corr']:>10.3f}"
          f"{r['crps_cal']:>9.4f}{r['r_unc']:>12.3f}{r['factor']:>8.3f}"
          f"{r['cov_cal']:>8.4f}{r['width']:>8.4f}")

rng = np.random.default_rng(0)
print("\nspread OFF - spread ON on v1b (paired over cases):")
for k in ("rmse", "vort_rmse", "vort_corr", "crps_cal", "r_unc"):
    d = per_case["1000 steps, spread OFF"][k] - per_case["1000 steps, spread ON"][k]
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(6000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"   {k:<11}{d.mean():+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  "
          f"{'SIGNIFICANT' if lo * hi > 0 else 'TIED'}")

torch.save({"rows": rows, "per_case": per_case,
            "meta": {"seed": SEED, "benchmark": "ocean_bench_v1b.npz",
                     "benchmark_md5": "08f1a69fbbc6186b4bda90dfe2e79280",
                     "n_cases": n, "n_draws": 20, "level": LEVEL}},
           ROOT / "benchmark" / "results_stream_spread_replication.pt")
print("\nsaved benchmark/results_stream_spread_replication.pt")
