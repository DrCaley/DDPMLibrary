"""Does Stream's magnitude network earn its place?

The two-network design exists because differentiating a stream function suppresses
amplitude and squared error pulls toward a blurred mean, so the diffusion model's
own speeds collapse. A second network predicts speed and the two are fused. That
premise has never been tested end to end: nobody measured the fused output against
simply keeping each diffusion draw's own magnitude.

If the fusion does not pay it is a far larger simplification than any loss term --
it removes a whole network, the coupled-magnitude step and a reprojection.

Method: monkeypatch only `fuse_coupled`, so the conditioning, sampler, seeds,
divergent-component addition and land masking are byte-identical between arms and
the sole difference is whether the magnitude net is consulted.
"""
import sys, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmark"))
from _paths import DEV  # noqa: E402
import ddpm_library.stream_predict as SP                            # noqa: E402
from ddpm_library import StreamDDPM, metrics                        # noqa: E402
from ddpm_library.stream.conditioning import helmholtz_project      # noqa: E402
import _score                                                        # noqa: E402

SEED = 20260830
KEYS = ("rmse", "angle", "vort_rmse", "vort_corr", "crps_cal")
b = np.load(ROOT / "benchmark/ocean_bench_v1.npz")
obs, pri, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
n = len(truth)

_real_fuse = SP.fuse_coupled

def _no_mag_fuse(members, cond, land_np, het_net, hsm, hss, het_clip, data_std, device):
    """Keep each draw's own speed; only reproject. The magnitude net is not used."""
    ocean_np = ~land_np
    return [helmholtz_project(np.asarray(d, dtype=np.float32), ocean_np) for d in members]

def run(use_mag):
    SP.fuse_coupled = _real_fuse if use_mag else _no_mag_fuse
    st = StreamDDPM(device=DEV)
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = st.predict([tuple(x) for x in obs[i]], pri[i], n_draws=20,
                              seed=SEED + i, full_field=True, calibrate=False)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    del st
    return np.stack(M), np.stack(S)

def score(use_mag, label):
    m, s = run(use_mag)
    pc, sm = _score.case_scores(m, s, truth, ocean, crps_fn=metrics.crps_gaussian,
                                keys=KEYS)
    print(_score.row(label, pc, sm), flush=True)
    return pc


print(_score.HEADER)
A = score(True,  "shipped (with magnitude net)")
B = score(False, "no magnitude net")
SP.fuse_coupled = _real_fuse
_score.paired_bootstrap(A, B, "shipped minus no-magnitude-net", keys=KEYS)
torch.save({"shipped": A, "no_mag": B,
            "meta": {"seed": SEED, "n_cases": n,
                     "benchmark_md5": "44b866540296490c615be13bacb4242e"}},
           ROOT / "benchmark/results_stream_magnitude_value.pt")
print("\nsaved benchmark/results_stream_magnitude_value.pt")
