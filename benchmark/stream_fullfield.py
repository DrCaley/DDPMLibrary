"""Does restoring the divergent component rescue Stream?

Stream's output is exactly divergence-free by construction, which puts a provable
floor of 0.0613 RMSE on the real field (Helmholtz projection is orthogonal, so no
divergence-free field can be closer). Every benchmark number so far used
full_field=False, i.e. the constrained output.

StreamDDPM.predict already has full_field=True, which adds back the divergent
component -- borrowed from a VCNN prediction rather than predicted by the model
itself. That is inelegant, but it costs nothing to run and it answers the question
the redesign hinges on: is the missing divergent component actually worth
recovering, or is the rest of Stream's error dominant?

  full_field=False   the shipped default, purely divergence-free
  full_field=True    + VCNN's divergent part

If True is much better, the constraint is the binding problem and predicting a
velocity potential properly is worth building. If it is not, the floor is real but
not what is holding Stream back, and the redesign is not worth the GPU time.
"""
import sys, warnings, hashlib
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/workspace/DDPMLibrary/src")
sys.path.insert(0, "/workspace/DDPMLibrary/benchmark")
import score
from ddpm_library import StreamDDPM, metrics

BENCH = Path("/workspace/DDPMLibrary/benchmark/ocean_bench_v1.npz")
OUT = Path("/workspace/DDPMLibrary/benchmark/results_stream_fullfield.pt")
SEED = 20260830

bench = np.load(BENCH)
obs_all, priors_all, truth = bench["observations"], bench["priors"], bench["truth"]
ocean = np.asarray(bench["ocean_mask"], bool)
n = len(truth)
st = StreamDDPM(device="cuda")


def divergence(f):
    u, v = f[..., 0], f[..., 1]
    return np.gradient(u, axis=1) + np.gradient(v, axis=0)


means, divs = {}, {}
for tag, ff in (("divfree_only", False), ("plus_divergent", True)):
    M, D = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, _ = st.predict([tuple(r) for r in obs_all[i]], list(priors_all[i]),
                              n_draws=20, seed=SEED + i, full_field=ff)
        m = np.asarray(m, np.float32)
        M.append(m); D.append(float(np.sqrt((divergence(m)[ocean] ** 2).mean())))
    means[tag] = np.stack(M); divs[tag] = np.array(D)
    print(f"  scored full_field={ff}", flush=True)

# the truth's own divergence, and the floor, for context
truth_div = float(np.mean([np.sqrt((divergence(t)[ocean] ** 2).mean()) for t in truth]))
scored = {k: score.score_model(v, bench) for k, v in means.items()}
eddy = {k: np.array([metrics.eddy_hit_rate(v[i], truth[i], ocean) for i in range(n)])
        for k, v in means.items()}

print(f"\n{'config':<18}{'RMSE':>9}{'angle_rms':>11}{'RMS div':>10}{'eddy':>8}")
for k in means:
    print(f"{k:<18}{scored[k]['rmse_vector'].mean():>9.4f}"
          f"{scored[k]['angle_rms_rad'].mean():>11.4f}{divs[k].mean():>10.5f}"
          f"{np.nanmean(eddy[k]):>8.4f}")
print(f"{'TRUTH':<18}{'—':>9}{'—':>11}{truth_div:>10.5f}{'—':>8}")

print("\npaired, plus_divergent - divfree_only:")
for m in ("rmse_vector", "angle_rms_rad"):
    d, lo, hi = score.bootstrap_ci(scored["plus_divergent"][m] - scored["divfree_only"][m])
    tag = "significant" if (lo > 0) == (hi > 0) else "TIED"
    print(f"  {m:<16}{d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  {tag}")
a, b = eddy["plus_divergent"], eddy["divfree_only"]
ok = ~(np.isnan(a) | np.isnan(b))
d, lo, hi = score.bootstrap_ci(a[ok] - b[ok])
print(f"  {'eddy_hit_rate':<16}{d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  "
      f"{'significant' if (lo>0)==(hi>0) else 'TIED'}")

print(f"\nreference points:")
print(f"  Stream's divergence-free floor          0.0613")
print(f"  floor removed entirely (quadrature)     0.0839")
print(f"  corrdiff (with priors) on this bench    0.0665")

torch.save({"meta": {"seed": SEED, "benchmark": BENCH.name,
                     "benchmark_md5": hashlib.md5(BENCH.read_bytes()).hexdigest(),
                     "n_cases": n, "floor_rmse": 0.0613},
            "means": {k: torch.from_numpy(v) for k, v in means.items()},
            "per_case_metrics": {k: {m: torch.from_numpy(np.asarray(v))
                                     for m, v in s.items()} for k, s in scored.items()},
            "rms_divergence": {k: torch.from_numpy(v) for k, v in divs.items()},
            "eddy_hit_rate": {k: torch.from_numpy(v) for k, v in eddy.items()},
            "truth_rms_divergence": truth_div}, OUT)
print(f"\nwrote {OUT}")
