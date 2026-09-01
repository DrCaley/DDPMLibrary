"""RePaint on the realistic benchmark -- the one model never scored on it.

On the idealised simultaneous-observation benchmark RePaint TIED CorrDiff on
accuracy (CRPS -0.0023, CI [-0.0056, +0.0010]) and beat it significantly on eddy
recall. It has never been run on the 2 h time-varying task, so excluding it from
the paper currently rests on an untested assumption.

RePaint is also mechanistically different from everything else here: it is a
DDPM trained on the field, steered at INFERENCE by guided sampling (DPS) rather
than conditioned at training. If observation staleness hurts guidance differently
than it hurts conditioning, that would show up here and nowhere else.

Scored with the 1 h discard as well, since RePaint takes priors and the discard
only helped models that carry them.
"""
import sys, warnings, hashlib
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/workspace/DDPMLibrary/src")
sys.path.insert(0, "/workspace/DDPMLibrary/benchmark")
import score
from ddpm_library import RePaint, RePaintUncond, metrics

BENCH = Path("/workspace/DDPMLibrary/benchmark/ocean_bench_v1.npz")
OUT = Path("/workspace/DDPMLibrary/benchmark/results_repaint_bench.pt")
SEED, STRIDE = 20260830, 5      # stride 5: the setting used in the staleness runs

bench = np.load(BENCH)
obs_all, priors_all, truth = bench["observations"], bench["priors"], bench["truth"]
ocean = np.asarray(bench["ocean_mask"], bool); n = len(truth)
rp, ru = RePaint(device="cuda"), RePaintUncond(device="cuda")


def fresh(rows, hours):
    ages = (rows[:, 2].max() - rows[:, 2]) / 3600.0
    return rows[ages <= hours]


CFG = {
    "repaint (1h cutoff)":   lambda r, p, i: rp.predict(
        [tuple(x) for x in fresh(r, 1.0)], p, n_draws=10, stride=STRIDE, seed=SEED + i)[0],
    "repaint (full track)":  lambda r, p, i: rp.predict(
        [tuple(x) for x in r], p, n_draws=10, stride=STRIDE, seed=SEED + i)[0],
    "repaint_uncond":        lambda r, p, i: ru.predict(
        [tuple(x) for x in r], n_draws=10, stride=STRIDE, seed=SEED + i)[0],
}

means, e_raw, e_rot = {}, {}, {}
for name, fn in CFG.items():
    M, A, B = [], [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.asarray(fn(obs_all[i], list(priors_all[i]), i), np.float32)
        M.append(m)
        A.append(metrics.eddy_hit_rate(m, truth[i], ocean))
        B.append(metrics.eddy_hit_rate(m, truth[i], ocean, rotational=True))
        if (i + 1) % 10 == 0:
            print(f"    {name}: {i+1}/{n}", flush=True)
    means[name] = np.stack(M); e_raw[name] = np.array(A); e_rot[name] = np.array(B)
    print(f"  done {name}", flush=True)

S = {k: score.score_model(v, bench) for k, v in means.items()}
print(f"\n{'model':<24}{'RMSE':>9}{'angle_rms':>11}{'eddy':>8}{'eddy_rot':>10}")
for k in CFG:
    print(f"{k:<24}{S[k]['rmse_vector'].mean():>9.4f}{S[k]['angle_rms_rad'].mean():>11.4f}"
          f"{np.nanmean(e_raw[k]):>8.4f}{np.nanmean(e_rot[k]):>10.4f}")
print(f"\nfor reference, from results_final_comparison.pt:")
print(f"{'corrdiff (1h cutoff)':<24}{0.0618:>9.4f}{0.6842:>11.4f}{0.4308:>8.4f}{0.4397:>10.4f}")
print(f"{'distattn':<24}{0.0738:>9.4f}{0.7875:>11.4f}{0.3579:>8.4f}{0.3826:>10.4f}")

ref = torch.load("/workspace/DDPMLibrary/benchmark/results_final_comparison.pt",
                 map_location="cpu", weights_only=False)
cd1 = ref["per_case_metrics"]["corrdiff (1h cutoff)"]
cdr = ref["eddy_rot"]["corrdiff (1h cutoff)"].numpy()
print(f"\npaired vs corrdiff (1h cutoff), same 40 cases:")
for k in CFG:
    for m, mine, theirs in (("rmse_vector", S[k]["rmse_vector"], cd1["rmse_vector"].numpy()),
                            ("angle_rms_rad", S[k]["angle_rms_rad"], cd1["angle_rms_rad"].numpy())):
        d, lo, hi = score.bootstrap_ci(mine - theirs)
        t = "significant" if (lo > 0) == (hi > 0) else "TIED"
        print(f"  {k:<24}{m:<16}{d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  {t}")
    x, y = e_rot[k], cdr
    ok = ~(np.isnan(x) | np.isnan(y))
    d, lo, hi = score.bootstrap_ci(x[ok] - y[ok])
    t = "significant" if (lo > 0) == (hi > 0) else "TIED"
    print(f"  {k:<24}{'eddy_rot':<16}{d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  {t}")

torch.save({"meta": {"seed": SEED, "stride": STRIDE, "n_cases": n,
                     "benchmark_md5": hashlib.md5(BENCH.read_bytes()).hexdigest()},
            "means": {k: torch.from_numpy(v) for k, v in means.items()},
            "per_case_metrics": {k: {m: torch.from_numpy(np.asarray(v))
                                     for m, v in s.items()} for k, s in S.items()},
            "eddy_raw": {k: torch.from_numpy(v) for k, v in e_raw.items()},
            "eddy_rot": {k: torch.from_numpy(v) for k, v in e_rot.items()}}, OUT)
print(f"\nwrote {OUT}")
