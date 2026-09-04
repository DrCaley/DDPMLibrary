"""Does a vorticity-matching term help the Stream direction network?

Motivation (measured first, see results_stream_vorticity_draws.pt): Stream's
ensemble-mean vorticity RMSE is only 3.5% better than predicting zero vorticity
everywhere, and its vorticity correlates 0.518 with truth against CorrDiff's
0.725. The amplitude is right (ratio 1.06) but the pattern is wrong, and single
draws carry 1.9x the true vorticity amplitude -- grid-scale noise the ensemble
average cancels. An L2 vorticity term attacks exactly that, so unlike on
CorrDiff there is a real deficit for it to close.

Arms: fine-tuned 3000 steps from the shipped checkpoint at lambda_vort =
0 (control), 1 and 5. Byte-identical batches across arms, verified. Both the
control and the treatment arms run with the spread term OFF, since it lives in
a loss variant with no vorticity term -- so these are not the shipped
configuration, but the vorticity contrast is clean.

Reports n_draws = 20 (the reported configuration) and n_draws = 1 (where the
grid-scale noise is largest, so where the term should bite hardest).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddpm_library import StreamDDPM                       # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import MODELS_DIR  # noqa: E402
from _vorticity import curl  # noqa: E402

BENCH = ROOT / "benchmark" / "ocean_bench_v1.npz"
SEED = 20260830
ARMS = {
    "baseline (shipped)": ROOT / "src/ddpm_library/assets/stream_dir_weights.pt",
    "lam0 (control)": MODELS_DIR / "stream_vort/lam00.pt",
    "lamV=1": MODELS_DIR / "stream_vort/lam10.pt",
    "lamV=5": MODELS_DIR / "stream_vort/lam50.pt",
}

b = np.load(BENCH)
obs_all, priors_all, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
interior = np.zeros_like(ocean); interior[1:-1, 1:-1] = True
M = interior & ocean




def score(pred, i):
    d = pred - truth[i]
    rmse = float(np.sqrt((d[ocean] ** 2).sum(-1).mean()))
    a = np.arccos(np.clip(
        (pred[ocean] * truth[i][ocean]).sum(-1)
        / (np.linalg.norm(pred[ocean], axis=-1)
           * np.linalg.norm(truth[i][ocean], axis=-1) + 1e-12), -1, 1))
    p, t = curl(pred)[M], curl(truth[i])[M]
    return {
        "rmse_vector": rmse,
        "angle_rms": float(np.sqrt((a ** 2).mean())),
        "vort_rmse": float(np.sqrt(((p - t) ** 2).mean())),
        "vort_corr": float(np.corrcoef(p, t)[0, 1]),
        "vort_amp": float(np.sqrt((p ** 2).mean()) / np.sqrt((t ** 2).mean())),
    }


def boot(a, b_, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b_
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean(), lo, hi, lo * hi > 0


missing = [k for k, v in ARMS.items() if not Path(v).exists()]
if missing:
    sys.exit(f"missing checkpoints: {missing}")

out = {}
for nd in (20, 1):
    for name, path in ARMS.items():
        st = StreamDDPM(device="mps", dir_weights_path=path)
        rows = []
        for i in range(len(truth)):
            pred = st.predict([tuple(x) for x in obs_all[i]], priors_all[i],
                              n_draws=nd, seed=SEED + i, full_field=True)[0]
            rows.append(score(pred, i))
        out[(nd, name)] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        m = out[(nd, name)]
        print(f"n_draws={nd:2d}  {name:20s} rmse {m['rmse_vector'].mean():.5f}  "
              f"angle {m['angle_rms'].mean():.5f}  vortRMSE {m['vort_rmse'].mean():.5f}  "
              f"corr {m['vort_corr'].mean():.3f}  amp {m['vort_amp'].mean():.3f}",
              flush=True)
        del st

print("\n=== isolated effect of the term (vs the matched lambda=0 control) ===")
for nd in (20, 1):
    for arm in ("lamV=1", "lamV=5"):
        print(f"\nn_draws={nd}, {arm} - lam0:")
        for k in ("rmse_vector", "angle_rms", "vort_rmse", "vort_corr"):
            d, lo, hi, sig = boot(out[(nd, arm)][k], out[(nd, "lam0 (control)")][k])
            print(f"   {k:12s} {d:+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  "
                  f"{'SIGNIFICANT' if sig else 'TIED'}")

print("\n=== fine-tuning at all, vs the shipped checkpoint ===")
for nd in (20, 1):
    for k in ("rmse_vector", "vort_rmse", "vort_corr"):
        d, lo, hi, sig = boot(out[(nd, "lam0 (control)")][k],
                              out[(nd, "baseline (shipped)")][k])
        print(f"   n_draws={nd:2d} {k:12s} lam0-baseline {d:+.6f}  "
              f"CI [{lo:+.6f}, {hi:+.6f}]  {'SIGNIFICANT' if sig else 'TIED'}")

torch.save({"per_case": {f"n{nd}|{n}": v for (nd, n), v in out.items()},
            "meta": {"seed": SEED, "n_cases": len(truth),
                     "benchmark_md5": "44b866540296490c615be13bacb4242e",
                     "steps": 3000, "spread_term": "off in all arms",
                     "note": "arms differ only in lambda_vort; batches verified "
                             "bit-identical at step 1"}},
           ROOT / "benchmark" / "results_stream_vorticity.pt")
print("\nsaved benchmark/results_stream_vorticity.pt")
