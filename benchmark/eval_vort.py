"""Score the vorticity fine-tune arms against the shipped CorrDiff.

Three checkpoints, identical inputs, paired over cases:

    baseline   the shipped checkpoint, untouched
    lam0       fine-tuned with the structural term OFF  -- the control that
               separates "the term helped" from "more training helped"
    lamV       fine-tuned with the vorticity term ON

Both arms saw byte-identical batches in identical order (verified: same seed,
--deterministic_data, explicit DataLoader generator), so lamV - lam0 isolates the
term itself.

Reports RMSE and angle, and -- the reason for the experiment -- the structural
metrics: vorticity RMSE and Okubo-Weiss eddy recall. A structural term that
improves eddies while costing RMSE is a real, reportable trade; one that moves
neither is a clean null.
"""
import sys, warnings, hashlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/DDPMLibrary/src")
sys.path.insert(0, "/workspace/DDPMLibrary/benchmark")
import score
from ddpm_library import CorrDiff, metrics

BENCH = Path("/workspace/DDPMLibrary/benchmark/ocean_bench_v1.npz")
OUT = Path("/workspace/DDPMLibrary/benchmark/results_vorticity.pt")
SEED = 20260830
CKPTS = {
    "baseline": "/workspace/DDPMLibrary/src/ddpm_library/assets/corrdiff_weights.pt",
    "lam0":     "/workspace/vort_lam0/corrdiff_v2_last.pt",
    "lamV":     "/workspace/vort_lam1/corrdiff_v2_last.pt",
}

bench = np.load(BENCH)
obs_all, priors_all, truth = bench["observations"], bench["priors"], bench["truth"]
ocean = np.asarray(bench["ocean_mask"], bool)
n = len(truth)


def curl(f):
    """Relative vorticity of a (H, W, 2) field, matching the training operator."""
    u, v = f[..., 0], f[..., 1]
    return np.gradient(v, axis=1) - np.gradient(u, axis=0)


def structural(pred, tru):
    cp, ct = curl(pred)[ocean], curl(tru)[ocean]
    return {"vorticity_rmse": float(np.sqrt(((cp - ct) ** 2).mean())),
            "eddy_hit_rate": float(metrics.eddy_hit_rate(pred, tru, ocean))}


means, extra = {}, {}
for name, path in CKPTS.items():
    if not Path(path).exists():
        print(f"  SKIP {name}: {path} missing"); continue
    mdl = CorrDiff(device="cuda", weights_path=path)
    M, X = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, _ = mdl.predict([tuple(r) for r in obs_all[i]], list(priors_all[i]),
                               n_draws=20, seed=SEED + i)
        m = np.asarray(m, np.float32)
        M.append(m); X.append(structural(m, truth[i]))
    means[name] = np.stack(M)
    extra[name] = {k: np.array([x[k] for x in X]) for k in X[0]}
    print(f"  scored {name}", flush=True)
    del mdl; torch.cuda.empty_cache()

scored = {k: score.score_model(v, bench) for k, v in means.items()}

print(f"\n{'checkpoint':<12}{'RMSE':>10}{'angle_rms':>12}"
      f"{'vort_RMSE':>12}{'eddy_recall':>13}")
for k in means:
    s, x = scored[k], extra[k]
    print(f"{k:<12}{s['rmse_vector'].mean():>10.4f}{s['angle_rms_rad'].mean():>12.4f}"
          f"{x['vorticity_rmse'].mean():>12.5f}{np.nanmean(x['eddy_hit_rate']):>13.4f}")

def cmp(a, b):
    print(f"\n{a} - {b}, paired over {n} cases:")
    for m in ("rmse_vector", "angle_rms_rad"):
        d, lo, hi = score.bootstrap_ci(scored[a][m] - scored[b][m])
        tag = "significant" if (lo > 0) == (hi > 0) else "TIED"
        print(f"  {m:<16}{d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  {tag}")
    for m in ("vorticity_rmse", "eddy_hit_rate"):
        xa, xb = extra[a][m], extra[b][m]
        ok = ~(np.isnan(xa) | np.isnan(xb))
        d, lo, hi = score.bootstrap_ci(xa[ok] - xb[ok])
        tag = "significant" if (lo > 0) == (hi > 0) else "TIED"
        print(f"  {m:<16}{d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  {tag}"
              f"   (n={int(ok.sum())})")

if "lamV" in means and "lam0" in means:
    cmp("lamV", "lam0")        # the term, isolated
if "lam0" in means and "baseline" in means:
    cmp("lam0", "baseline")    # what extra training alone did
if "lamV" in means and "baseline" in means:
    cmp("lamV", "baseline")    # net effect vs what we ship today

torch.save({
    "meta": {"seed": SEED, "benchmark": BENCH.name,
             "benchmark_md5": hashlib.md5(BENCH.read_bytes()).hexdigest(),
             "n_cases": n, "checkpoints": CKPTS,
             "note": "lamV - lam0 isolates the vorticity term; both arms saw "
                     "byte-identical batches (verified)."},
    "means": {k: torch.from_numpy(v) for k, v in means.items()},
    "per_case_metrics": {k: {m: torch.from_numpy(np.asarray(v))
                             for m, v in s.items()} for k, s in scored.items()},
    "structural": {k: {m: torch.from_numpy(v) for m, v in x.items()}
                   for k, x in extra.items()},
}, OUT)
print(f"\nwrote {OUT}")
