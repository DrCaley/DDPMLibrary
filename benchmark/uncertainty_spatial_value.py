"""Does per-cell uncertainty beat a single scalar? Tested on all three models.

Stream's shipped map does NOT (see STREAM_LOSS_ABLATIONS.md §2b): replacing sigma
with its per-case ocean mean gives narrower intervals and better calibrated CRPS
at equal coverage. Since a per-cell uncertainty map a planner can route on is the
project's deliverable, the same test has to be run on CorrDiff and DistAttn
before per-cell uncertainty can be claimed as a contribution.

Each model runs at its own reported configuration. For every sigma treatment the
conformal factor is re-fitted (fit on 20 cases, verify on the held-out 20), so
rows are compared at their own calibrated coverage rather than a shared factor.
"""
import sys, warnings
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from _paths import DEV  # noqa: E402
from ddpm_library import CorrDiff, DistAttn, StreamDDPM, metrics   # noqa: E402
from ddpm_library import config as C                               # noqa: E402

LEVEL, SEED, z = 0.90, 20260830, 1.6448536269514722
TREATMENTS = [0.0, 0.8, 3.2, 12.8, "const"]

b = np.load(ROOT / "benchmark" / "ocean_bench_v1.npz")
obs_all, priors_all, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
n, half = len(truth), len(truth) // 2


def fresh(rows, hours):
    ages = (rows[:, 2].max() - rows[:, 2]) / 3600.0
    return rows[ages <= hours]


def treat(u, sigma):
    if sigma == "const":
        out = np.zeros_like(u)
        for c in range(u.shape[-1]):
            out[..., c] = np.where(ocean, u[..., c][ocean].mean(), 0.0)
        return out
    if sigma <= 0:
        return u
    out = np.empty_like(u)
    md = gaussian_filter(ocean.astype(np.float64), sigma, mode="nearest")
    for c in range(u.shape[-1]):
        v = np.where(ocean, u[..., c], 0.0).astype(np.float64)
        out[..., c] = np.where(ocean, gaussian_filter(v, sigma, mode="nearest")
                               / np.maximum(md, 1e-12), 0.0)
    return out


def collect(predict_one):
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = predict_one(i)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    return np.stack(M), np.stack(S)


MODELS = {}
cd = CorrDiff(device=DEV)
MODELS["corrdiff (1h, 20 draws)"] = lambda i, cd=cd: cd.predict(
    [tuple(x) for x in fresh(obs_all[i], 1.0)], priors_all[i],
    n_draws=20, seed=SEED + i, calibrate=False)
da = DistAttn(device=DEV)
MODELS[f"distattn ({C.DISTATTN_DEFAULT_N_DRAWS} draws)"] = lambda i, da=da: da.predict(
    [tuple(x) for x in obs_all[i]], n_draws=C.DISTATTN_DEFAULT_N_DRAWS,
    seed=SEED + i, calibrate=False)
st = StreamDDPM(device=DEV)
MODELS["stream (20 draws, raw sig)"] = lambda i, st=st: st.predict(
    [tuple(x) for x in obs_all[i]], priors_all[i], n_draws=20, seed=SEED + i,
    full_field=True, smooth_uncertainty=False, calibrate=False)

print(f"{'model':<28}{'sigma':>7}{'covraw':>8}{'factor':>8}{'covcal':>8}"
      f"{'width':>8}{'CRPScal':>9}{'r(sig,err)':>12}")
out = {}
for name, fn in MODELS.items():
    m, S0 = collect(fn)
    err_full = np.abs(m - truth)
    for t in TREATMENTS:
        s = np.stack([treat(S0[i], t) for i in range(n)]).astype(np.float32)
        cov = np.mean([metrics.coverage(m[i], s[i], truth[i], level=LEVEL,
                                        ocean_mask=ocean) for i in range(n)])
        err, sg = err_full[:, ocean], s[:, ocean]
        ok = sg[:half] > 1e-9
        f = float(np.quantile(err[:half][ok] / (z * sg[:half][ok]), LEVEL))
        covc = float(np.mean(err[half:] <= z * f * sg[half:]))
        w = float(np.mean(2 * z * f * sg[half:]))
        crps = np.mean([metrics.crps_gaussian(m[i], f * s[i], truth[i],
                                              ocean_mask=ocean) for i in range(n)])
        ru = np.mean([np.corrcoef(s[i][ocean].ravel(), err_full[i][ocean].ravel())[0, 1]
                      for i in range(n)])
        lab = t if isinstance(t, str) else f"{t:.1f}"
        print(f"{name:<28}{lab:>7}{cov:>8.4f}{f:>8.3f}{covc:>8.4f}{w:>8.4f}"
              f"{crps:>9.4f}{ru:>12.3f}", flush=True)
        out[(name, str(t))] = dict(cov_raw=cov, factor=f, cov_cal=covc, width=w,
                                   crps_cal=crps, r_unc=ru)

print("\nper-cell (best blur) vs constant sigma -- does spatial structure pay?")
for name in MODELS:
    blurs = [(k[1], v) for k, v in out.items() if k[0] == name and k[1] != "const"]
    cst = out[(name, "const")]
    bw = min(blurs, key=lambda kv: kv[1]["width"])
    verdict = ("per-cell WINS" if bw[1]["width"] < cst["width"]
               and bw[1]["crps_cal"] <= cst["crps_cal"] + 1e-9 else
               "constant WINS or ties")
    print(f"  {name:<28} best blur sigma={bw[0]:>5}  width {bw[1]['width']:.4f} "
          f"vs const {cst['width']:.4f}   CRPScal {bw[1]['crps_cal']:.4f} vs "
          f"{cst['crps_cal']:.4f}   -> {verdict}")

torch.save({"rows": out, "treatments": [str(t) for t in TREATMENTS],
            "meta": {"seed": SEED, "n_cases": n, "level": LEVEL,
                     "benchmark_md5": "44b866540296490c615be13bacb4242e"}},
           ROOT / "benchmark" / "results_uncertainty_spatial_value.pt")
print("\nsaved benchmark/results_uncertainty_spatial_value.pt")
