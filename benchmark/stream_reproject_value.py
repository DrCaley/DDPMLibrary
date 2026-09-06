"""Do the reprojection and the VCNN divergent component earn their place?

The shipped pipeline makes the field divergence-free (`helmholtz_project`) and then
immediately adds a divergent component back (from a VCNN). That looks circular, and
the obvious question is whether skipping both is just as good.

The design rationale is that the two divergences differ in kind. Rescaling each
pixel's speed in `coupled_magnitude` introduces divergence as a numerical
side-effect -- nothing chose it, it is not an estimate of anything. The
reprojection removes that, and the VCNN then contributes divergence that was
actually predicted. This measures whether that distinction is worth the two steps.

Four arms, everything else byte-identical (same conditioning, sampler, seeds, land
masking); only the post-fusion handling differs:

    A  reproject + VCNN     the shipped pipeline
    B  reproject only       full_field=False -- previously measured at +12.5% RMSE
    C  neither              keep whatever divergence the magnitude swap produced
    D  VCNN only            no reprojection, but still add the predicted divergence

Also reports each arm's actual |divergence|, since that is the physical quantity the
two steps exist to control.
"""
import sys, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmark"))
from _paths import DEV  # noqa: E402
import ddpm_library.stream_predict as SP                              # noqa: E402
from ddpm_library import StreamDDPM, metrics                          # noqa: E402
from ddpm_library.stream.conditioning import (                        # noqa: E402
    predict_speed_mean_sigma, coupled_magnitude)
import _score                                                          # noqa: E402
from _vorticity import interior_ocean_mask                             # noqa: E402

SEED = 20260830
KEYS = ("rmse", "angle", "div_rmse", "vort_rmse", "vort_corr", "crps_cal")
b = np.load(ROOT / "benchmark/ocean_bench_v1.npz")
obs, pri, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool); land = ~ocean
Mk = interior_ocean_mask(ocean)
n = len(truth)

_real_fuse = SP.fuse_coupled


def _fuse_no_reproject(members, cond, land_np, het_net, hsm, hss, het_clip,
                       data_std, device):
    """coupled_magnitude WITHOUT the Helmholtz reprojection."""
    mu_n, sig_n = predict_speed_mean_sigma(het_net, hsm, hss, land_np, data_std,
                                           device, cond, het_clip)
    return coupled_magnitude(members, mu_n, sig_n, ~land_np)


def mean_abs_div(field):
    """mean |du/dH + dv/dW| over interior ocean cells, central differences."""
    u, v = field[..., 0], field[..., 1]
    dudH = np.zeros_like(u); dudH[1:-1, :] = (u[2:, :] - u[:-2, :]) / 2
    dvdW = np.zeros_like(v); dvdW[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / 2
    return float(np.abs(dudH + dvdW)[Mk].mean())


def run(reproject, vcnn, label):
    SP.fuse_coupled = _real_fuse if reproject else _fuse_no_reproject
    st = StreamDDPM(device=DEV)
    M, S, dv = [], [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = st.predict([tuple(x) for x in obs[i]], pri[i], n_draws=20,
                              seed=SEED + i, full_field=vcnn, calibrate=False)
        m = np.asarray(m, np.float32)
        M.append(m); S.append(np.asarray(s, np.float32)); dv.append(mean_abs_div(m))
    del st
    SP.fuse_coupled = _real_fuse
    pc, sm = _score.case_scores(np.stack(M), np.stack(S), truth, ocean,
                                crps_fn=metrics.crps_gaussian, keys=KEYS)
    print(_score.row(label, pc, sm) + f"{np.mean(dv):>12.2e}", flush=True)
    return pc, float(np.mean(dv))


print(_score.HEADER + f"{'|div| out':>12}")
arms = {}
arms["A reproject + VCNN (shipped)"] = run(True,  True,  "A reproject + VCNN (shipped)")
arms["B reproject only"]             = run(True,  False, "B reproject only")
arms["C neither"]                    = run(False, False, "C neither")
arms["D VCNN only (no reproject)"]   = run(False, True,  "D VCNN only (no reproject)")

base = "A reproject + VCNN (shipped)"
for k in arms:
    if k != base:
        _score.paired_bootstrap(arms[k][0], arms[base][0], f"{k}  minus  {base}", keys=KEYS)

torch.save({k: v[0] for k, v in arms.items()} |
           {"div_out": {k: v[1] for k, v in arms.items()},
            "meta": {"seed": SEED, "n_cases": n, "n_draws": 20,
                     "benchmark_md5": "44b866540296490c615be13bacb4242e"}},
           ROOT / "benchmark/results_stream_reproject.pt")
print("\nsaved benchmark/results_stream_reproject.pt")
