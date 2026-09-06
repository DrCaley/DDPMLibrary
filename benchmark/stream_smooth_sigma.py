"""Does the uncertainty-smoothing sigma still want 0.8 without the spread term?

STREAM_UNC_SMOOTH_SIGMA = 0.8 exists to remove a grid-scale checkerboard mode
from the ensemble spread. The spread-term ablation shows the no-spread model
carries far less of that noise (single-draw vorticity amplitude ~1.10-1.17 vs
~1.85-1.89), so the tuned value may no longer be right for it. This matters
because swapping in the predecessor means shipping its inference config too.

Raw sigma is fetched once per model (smooth_uncertainty=False) and the smoothing
is applied here, so one inference pass serves every sigma.
"""
import sys, warnings
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from _paths import DEV, MODELS_DIR  # noqa: E402
from ddpm_library import StreamDDPM, metrics

LEVEL, SEED, z = 0.90, 20260830, 1.6448536269514722
SIGMAS = [0.0, 0.8, 1.6, 3.2, 6.4, 12.8, "const"]
# "const" replaces sigma with its own per-case ocean mean -- the smoothing
# limit. If it matches the best blur, the uncertainty map carries no useful
# spatial information, which bears on the planner claim.
DS = MODELS_DIR
ARMS = [("predecessor (no spread)", DS / "StreamFn_Cond_x0_mag.pt"),
        ("shipped (spread on)", ROOT / "src/ddpm_library/assets/stream_dir_weights.pt")]

b = np.load(ROOT / "benchmark/ocean_bench_v1.npz")
obs, pri, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
n, half = len(truth), len(truth) // 2

def smooth(u, sigma):
    """Nan-aware Gaussian smooth over ocean cells, matching _smooth_ocean."""
    if sigma == "const":
        out = np.zeros_like(u)
        for c in range(u.shape[-1]):
            out[..., c] = np.where(ocean, u[..., c][ocean].mean(), 0.0)
        return out
    if sigma <= 0:
        return u
    out = np.empty_like(u)
    m = ocean.astype(np.float64)
    md = gaussian_filter(m, sigma, mode="nearest")
    for c in range(u.shape[-1]):
        v = np.where(ocean, u[..., c], 0.0).astype(np.float64)
        out[..., c] = np.where(ocean, gaussian_filter(v, sigma, mode="nearest")
                               / np.maximum(md, 1e-12), 0.0)
    return out

print(f"{'arm':<26}{'sigma':>6}{'covraw':>8}{'factor':>8}{'covcal':>8}"
      f"{'width':>8}{'CRPScal':>9}{'r(sig,err)':>12}")
best = {}
for name, p in ARMS:
    st = StreamDDPM(device=DEV, dir_weights_path=p)
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = st.predict([tuple(x) for x in obs[i]], pri[i], n_draws=20,
                              seed=SEED + i, full_field=True,
                              smooth_uncertainty=False, calibrate=False)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    m, S0 = np.stack(M), np.stack(S); del st
    err_full = np.abs(m - truth)
    for sg_v in SIGMAS:
        s = np.stack([smooth(S0[i], sg_v) for i in range(n)]).astype(np.float32)
        cov = np.mean([metrics.coverage(m[i], s[i], truth[i], level=LEVEL,
                                        ocean_mask=ocean) for i in range(n)])
        err, sgo = err_full[:, ocean], s[:, ocean]
        ok = sgo[:half] > 1e-9
        f = float(np.quantile(err[:half][ok] / (z * sgo[:half][ok]), LEVEL))
        covc = float(np.mean(err[half:] <= z * f * sgo[half:]))
        w = float(np.mean(2 * z * f * sgo[half:]))
        crps = np.mean([metrics.crps_gaussian(m[i], f * s[i], truth[i],
                                              ocean_mask=ocean) for i in range(n)])
        ru = np.mean([np.corrcoef(s[i][ocean].ravel(), err_full[i][ocean].ravel())[0, 1]
                      for i in range(n)])
        lab = sg_v if isinstance(sg_v, str) else f"{sg_v:.1f}"
        print(f"{name:<26}{lab:>6}{cov:>8.4f}{f:>8.3f}{covc:>8.4f}"
              f"{w:>8.4f}{crps:>9.4f}{ru:>12.3f}", flush=True)
        best.setdefault(name, []).append((sg_v, w, crps, ru, covc))
print("\nsharpest sigma at >=0.88 calibrated coverage (lower width is better):")
for name, rec in best.items():
    ok = [r for r in rec if r[4] >= 0.88]
    pick = min(ok, key=lambda r: r[1]) if ok else None
    print(f"  {name:<26} sigma={pick[0]}  width={pick[1]:.4f}  "
          f"CRPScal={pick[2]:.4f}  r={pick[3]:.3f}" if pick else f"  {name}: none")
torch.save({"best": best, "sigmas": SIGMAS, "seed": SEED,
            "benchmark_md5": "44b866540296490c615be13bacb4242e"},
           ROOT / "benchmark/results_stream_smooth_sigma.pt")
print("\nsaved benchmark/results_stream_smooth_sigma.pt")
