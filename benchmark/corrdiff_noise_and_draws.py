"""CorrDiff's two untested dials: ensemble size, and the sensor-noise channel.

Part 1 -- n_draws. The shipped default is 20. Cost is linear in it, and the raw
ensemble std is biased low for small ensembles, so the conformal factor is refit
per size rather than reused. All sizes are nested subsets of ONE 40-draw sampling
run per case (only `ddim_sample_residual` is intercepted, and it is cached by
seed), so the comparison is exactly paired and costs one 40-draw run.

Part 2 -- sensor_noise. It is a trained conditioning channel, not a post-hoc
widening, so a bare sweep on the noise-free benchmark would only measure the cost
of lying to the model. The question worth asking is whether the channel does its
job when the observations really are noisy:

    A  clean obs,  sensor_noise = 0      (shipped baseline)
    B  noisy obs,  sensor_noise = 0      (model not told)
    C  noisy obs,  sensor_noise = sigma  (model told)

B and C see byte-identical noisy observations. If C does not beat B the channel
is another feature that does not earn its place.
"""
import sys, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmark"))
from _paths import DEV  # noqa: E402
import ddpm_library.corrdiff_predict as CD                           # noqa: E402
from ddpm_library import CorrDiff, metrics                           # noqa: E402
import _score                                                         # noqa: E402

SEED = 20260830
NOISE_FRAC = 0.05                       # obs error, as a fraction of field std
MAXDRAWS   = 40

b = np.load(ROOT / "benchmark/ocean_bench_v1.npz")
obs, pri, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
n = len(truth)

_real_sampler = CD.ddim_sample_residual
_cache = {}

def _cached_sampler(model, cond_t, ocean_t, diffusion, device, *,
                    n_draws, steps, seed):
    """Sample MAXDRAWS once per seed, hand back the first n_draws of them."""
    if seed not in _cache:
        _cache[seed] = _real_sampler(model, cond_t, ocean_t, diffusion, device,
                                     n_draws=MAXDRAWS, steps=steps, seed=seed)
    return _cache[seed][:n_draws]

KEYS = ("rmse", "angle", "vort_rmse", "vort_corr", "crps_cal")


def score(m, s, label):
    pc, sm = _score.case_scores(m, s, truth, ocean, crps_fn=metrics.crps_gaussian,
                                keys=KEYS)
    print(_score.row(label, pc, sm), flush=True)
    return pc


def boot(A, B, lab):
    return _score.paired_bootstrap(A, B, lab, keys=KEYS)


def run(model, obs_arr, *, n_draws, sensor_noise):
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = model.predict([tuple(x) for x in obs_arr[i]], pri[i],
                                 n_draws=n_draws, sensor_noise=sensor_noise,
                                 seed=SEED + i, calibrate=False)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    return np.stack(M), np.stack(S)

HDR = _score.HEADER
cd = CorrDiff(device=DEV)
out = {}

# ---- Part 1: ensemble size (nested subsets of one 40-draw run) --------------
print("=== Part 1: n_draws ===\n" + HDR)
CD.ddim_sample_residual = _cached_sampler
for k in (5, 10, 20, 40):
    m, s = run(cd, obs, n_draws=k, sensor_noise=0.0)
    out[f"draws{k}"] = score(m, s, f"n_draws = {k}" + ("  (shipped)" if k == 20 else ""))
CD.ddim_sample_residual = _real_sampler; _cache.clear()
boot(out["draws40"], out["draws20"], "n_draws 40 minus shipped 20")
boot(out["draws10"], out["draws20"], "n_draws 10 minus shipped 20")

# ---- Part 2: the sensor-noise channel ---------------------------------------
sigma = NOISE_FRAC * float(np.mean(cd.data_std))     # m/s, model's own field scale
noisy = obs.copy().astype(np.float64)
for i in range(n):                                   # identical noise for arms B and C
    r = np.random.default_rng(1000 + i)
    noisy[i][:, 3:5] += r.normal(0.0, sigma, size=noisy[i][:, 3:5].shape)
print(f"\n=== Part 2: sensor_noise channel (obs error {NOISE_FRAC:.0%} of field std "
      f"= {sigma:.4f} m/s) ===\n" + HDR)
out["A"] = score(*run(cd, obs,   n_draws=20, sensor_noise=0.0),        "A  clean obs, noise=0 (shipped)")
out["B"] = score(*run(cd, noisy, n_draws=20, sensor_noise=0.0),        "B  noisy obs, noise=0 (not told)")
out["C"] = score(*run(cd, noisy, n_draws=20, sensor_noise=NOISE_FRAC), "C  noisy obs, noise=0.05 (told)")
boot(out["C"], out["B"], "C minus B -- does telling the model help")
boot(out["B"], out["A"], "B minus A -- what the noise costs")

torch.save({**out, "meta": {"seed": SEED, "n_cases": n, "noise_frac": NOISE_FRAC,
                            "sigma_ms": sigma, "max_draws": MAXDRAWS}},
           ROOT / "benchmark/results_corrdiff_noise_draws.pt")
print("\nsaved benchmark/results_corrdiff_noise_draws.pt")
