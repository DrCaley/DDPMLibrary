"""What do StreamDDPM's shipped defaults cost?

Two of them degrade it, and both were found by auditing rather than by anything
failing:

  n_draws=1     every other diffusion model here defaults to 10 or 20. At 1 the
                "mean" is a single noisy sample rather than an ensemble mean, and
                the returned uncertainty is zeros.
  full_field=False   the output is then exactly divergence-free, which costs 12.5%
                RMSE on a field whose divergent component carries 14% of the energy.

Our benchmark runs always passed n_draws=20 explicitly, so published numbers are
unaffected -- but anyone calling StreamDDPM().predict(obs, priors) with defaults
gets both penalties at once.
"""
import sys, warnings, hashlib
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/workspace/DDPMLibrary/src")
sys.path.insert(0, "/workspace/DDPMLibrary/benchmark")
import score
from ddpm_library import StreamDDPM

BENCH = Path("/workspace/DDPMLibrary/benchmark/ocean_bench_v1.npz")
bench = np.load(BENCH); obs_all, priors_all = bench["observations"], bench["priors"]
n = len(bench["truth"]); st = StreamDDPM(device="cuda"); SEED = 20260830

CFG = {"shipped defaults (n_draws=1, divfree)": dict(n_draws=1,  full_field=False),
       "n_draws=20, divfree":                   dict(n_draws=20, full_field=False),
       "n_draws=1, +divergent":                 dict(n_draws=1,  full_field=True),
       "best (n_draws=20, +divergent)":         dict(n_draws=20, full_field=True)}
means = {}
for name, kw in CFG.items():
    M = []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, _ = st.predict([tuple(r) for r in obs_all[i]], list(priors_all[i]),
                              seed=SEED + i, **kw)
        M.append(np.asarray(m, np.float32))
    means[name] = np.stack(M); print(f"  {name}", flush=True)

S = {k: score.score_model(v, bench) for k, v in means.items()}
print(f"\n{'configuration':<40}{'RMSE':>9}{'angle_rms':>11}")
for k in CFG:
    print(f"{k:<40}{S[k]['rmse_vector'].mean():>9.4f}{S[k]['angle_rms_rad'].mean():>11.4f}")

base = "shipped defaults (n_draws=1, divfree)"
print(f"\ncost of the shipped defaults, paired vs best:")
for m in ("rmse_vector", "angle_rms_rad"):
    d, lo, hi = score.bootstrap_ci(S[base][m] - S["best (n_draws=20, +divergent)"][m])
    t = "significant" if (lo > 0) == (hi > 0) else "TIED"
    pct = 100 * d / S["best (n_draws=20, +divergent)"][m].mean()
    print(f"  {m:<16}{d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  ({pct:+.1f}%)  {t}")

torch.save({"meta": {"seed": SEED, "benchmark_md5": hashlib.md5(BENCH.read_bytes()).hexdigest()},
            "per_case_metrics": {k: {m: torch.from_numpy(np.asarray(v)) for m, v in s.items()}
                                 for k, s in S.items()}},
           "/workspace/DDPMLibrary/benchmark/results_stream_defaults.pt")
print("\nwrote results_stream_defaults.pt")
