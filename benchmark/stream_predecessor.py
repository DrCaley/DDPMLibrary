"""The natural experiment: the shipped Stream model's own predecessor.

The shipped direction checkpoint is bit-identical to StreamFn_Cond_x0_mag_spread.pt
(epoch 48), whose recorded `init` is
`best_streamfncond_minsnr5_mag0.2_ang1_lags13-25_div_free_cosine.pt`.
Models/StreamFn_Cond_x0_mag.pt carries exactly that config -- same
x0_streamfn_cond architecture, cond_ch 10, lambda_angle 1.0, lambda_mag 0.2,
min_snr_gamma 5.0, lags (13,25), div_free noise, cosine schedule, and NO spread
term -- at epoch 78.

So the shipped model is a fully-trained no-spread model plus 48 epochs with the
spread term added. Comparing the two is the strongest evidence available about
that term, since both are real training runs rather than short fine-tunes.

Caveat: the spread run also changed path_steps (120-200 -> 90) and lr (2e-4 ->
5e-5), so this is the historical comparison, not a single-variable one. Read it
alongside the matched fine-tune ablation, which does isolate the term.
"""
import sys, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddpm_library import StreamDDPM, metrics
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import MODELS_DIR  # noqa: E402
from _vorticity import curl  # noqa: E402

LEVEL, SEED, z = 0.90, 20260830, 1.6448536269514722
DS = MODELS_DIR
ARMS = [("predecessor (78 ep, NO spread)", DS / "StreamFn_Cond_x0_mag.pt"),
        ("shipped (+48 ep, spread on)", ROOT / "src/ddpm_library/assets/stream_dir_weights.pt")]

b = np.load(ROOT / "benchmark/ocean_bench_v1.npz")
obs, pri, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
Mk = np.zeros_like(ocean); Mk[1:-1, 1:-1] = True; Mk &= ocean
n, half = len(truth), len(truth) // 2


rows, per_case = {}, {}
for nd in (20, 1):
    for name, p in ARMS:
        st = StreamDDPM(device="mps", dir_weights_path=p)
        M, S, vr, vc, va = [], [], [], [], []
        for i in range(n):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m, s = st.predict([tuple(x) for x in obs[i]], pri[i], n_draws=nd,
                                  seed=SEED + i, full_field=True, calibrate=False)
            M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
            pc, tc = curl(np.asarray(m))[Mk], curl(truth[i])[Mk]
            vr.append(np.sqrt(((pc - tc) ** 2).mean()))
            vc.append(np.corrcoef(pc, tc)[0, 1])
            va.append(np.sqrt((pc ** 2).mean()) / np.sqrt((tc ** 2).mean()))
        m, s = np.stack(M), np.stack(S); del st
        rmse = np.array([np.sqrt(((m[i] - truth[i])[ocean] ** 2).sum(-1).mean()) for i in range(n)])
        key = f"n{nd}|{name}"
        d = dict(rmse=rmse, vort_rmse=np.array(vr), vort_corr=np.array(vc), vort_amp=np.array(va))
        r = dict(rmse=rmse.mean(), vort_rmse=np.mean(vr), vort_corr=np.mean(vc),
                 vort_amp=np.mean(va))
        if nd == 20:
            err, sg = np.abs(m - truth)[:, ocean], s[:, ocean]
            ok = sg[:half] > 1e-9
            scale = float(np.quantile(err[:half][ok] / (z * sg[:half][ok]), LEVEL))
            crps_cal = np.array([metrics.crps_gaussian(m[i], scale * s[i], truth[i],
                                                       ocean_mask=ocean) for i in range(n)])
            r_unc = np.array([np.corrcoef(s[i][ocean].ravel(),
                                          np.abs(m[i] - truth[i])[ocean].ravel())[0, 1]
                              for i in range(n)])
            d.update(crps_cal=crps_cal, r_unc=r_unc)
            r.update(crps_cal=crps_cal.mean(), r_unc=r_unc.mean(), factor=scale,
                     cov_cal=float(np.mean(err[half:] <= z * scale * sg[half:])),
                     width=float(np.mean(2 * z * scale * sg[half:])))
        per_case[key] = d; rows[key] = r
        extra = (f" CRPScal {r['crps_cal']:.4f}  r(sig,err) {r['r_unc']:.3f}"
                 f"  factor {r['factor']:.3f}  width {r['width']:.4f}") if nd == 20 else ""
        print(f"n_draws={nd:2d}  {name:32s} rmse {r['rmse']:.5f}  "
              f"vortRMSE {r['vort_rmse']:.5f}  corr {r['vort_corr']:.3f}  "
              f"amp {r['vort_amp']:.3f}{extra}", flush=True)

rng = np.random.default_rng(0)
print("\npredecessor - shipped (paired over cases):")
for nd in (20, 1):
    a_k, b_k = f"n{nd}|predecessor (78 ep, NO spread)", f"n{nd}|shipped (+48 ep, spread on)"
    ks = ["rmse", "vort_rmse", "vort_corr"] + (["crps_cal", "r_unc"] if nd == 20 else [])
    print(f"  n_draws={nd}:")
    for k in ks:
        d = per_case[a_k][k] - per_case[b_k][k]
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(6000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"     {k:<10}{d.mean():+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  "
              f"{'SIGNIFICANT' if lo * hi > 0 else 'TIED'}")

torch.save({"rows": rows, "per_case": per_case,
            "meta": {"seed": SEED, "n_cases": n, "level": LEVEL,
                     "benchmark_md5": "44b866540296490c615be13bacb4242e"}},
           ROOT / "benchmark/results_stream_predecessor.pt")
print("\nsaved benchmark/results_stream_predecessor.pt")
