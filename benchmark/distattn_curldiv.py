"""Does DistAttn's curl/divergence term do anything? (the soft-physics test)

The paper's three-way claim -- no physics (CorrDiff) / soft physics (DistAttn) /
hard physics (Stream) -- rested on a CorrDiff vorticity ablation. DistAttn's own
curl/divergence term, lambda_cd = 0.002, had never been ablated. This is that test.

Two arms, 8 epochs each fine-tuned from the shipped checkpoint, differing only in
lambda_cd. Batches verified byte-identical across arms at step 1 (eps, cd and obs
terms match to 10 decimals); note CUDA backward non-determinism means the arms are
not bit-reproducible run to run, ~1e-8 relative per step.
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmark"))
from _paths import MODELS_DIR  # noqa: E402
from ddpm_library import DistAttn, metrics                       # noqa: E402
import _score                                                     # noqa: E402

SEED = 20260830
MD = MODELS_DIR / "distattn_ablate"
DEFAULT_ARMS = [("lambda_cd = 0.002 (shipped)", MD / "armA_cd0.002.pt"),
                ("lambda_cd = 0 (control)",     MD / "armB_cd0.0.pt"),
                # 1000x dose: a power check. If even this does not move the model,
                # the term has no usable gradient; if it does, lambda=0.002 is simply
                # too small -- a different claim, and not evidence that 0.002 is right.
                ("lambda_cd = 2.0 (1000x dose)", MD / "armC_cd2.0.pt")]

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--arm", action="append", metavar="NAME=PATH",
                help="arm to score; repeat. The first is the treatment and the "
                     "second the control. Omit to score the default fine-tune arms.")
ap.add_argument("--out", default="benchmark/results_distattn_curldiv.pt",
                help="output .pt, relative to the repo root")
ap.add_argument("--epochs", type=int, default=8, help="recorded in the metadata")
ap.add_argument("--note", default="batches verified byte-identical across arms; CUDA "
                                  "backward non-determinism ~1e-8 per step")
args = ap.parse_args()

if args.arm:
    ARMS = []
    for spec in args.arm:
        name, sep, path = spec.rpartition("=")   # last "=": names contain "=" too
        if not sep or not name.strip() or not path.strip():
            ap.error(f"--arm expects NAME=PATH, got {spec!r}")
        ARMS.append((name.strip(), Path(path.strip())))
    # An explicitly requested arm that is missing must fail loudly: silently
    # dropping one would score a different comparison than the one asked for.
    missing = [str(q) for _, q in ARMS if not q.exists()]
    if missing:
        ap.error("arm checkpoint(s) not found: " + ", ".join(missing))
else:
    ARMS = DEFAULT_ARMS
if len(ARMS) < 2:
    ap.error("need at least two arms (treatment and control)")

b = np.load(ROOT / "benchmark/ocean_bench_v1.npz")
obs, truth = b["observations"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
n = len(truth)

ARMS = [(nm, q) for nm, q in ARMS if Path(q).exists()]
print("arms found:", [nm for nm, _ in ARMS])

rows, per_case = {}, {}
for name, p in ARMS:
    da = DistAttn(device="mps", weights_path=p)
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = da.predict([tuple(x) for x in obs[i]], n_draws=10,
                              seed=SEED + i, calibrate=False)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    del da
    pc_, sm_ = _score.case_scores(np.stack(M), np.stack(S), truth, ocean,
                                  crps_fn=metrics.crps_gaussian)
    per_case[name] = pc_
    rows[name] = dict({k: float(np.mean(v)) for k, v in pc_.items()}, **sm_)
    print("  scored %s" % name, flush=True)

print(f"\n{'arm':<28}{'RMSE':>9}{'angle':>9}{'divRMSE':>9}{'vortRMSE':>10}"
      f"{'vortCorr':>10}{'CRPScal':>9}{'width':>8}")
for k, r in rows.items():
    print(f"{k:<28}{r['rmse']:>9.5f}{r['angle']:>9.5f}{r['div_rmse']:>9.5f}"
          f"{r['vort_rmse']:>10.5f}{r['vort_corr']:>10.3f}{r['crps_cal']:>9.4f}{r['width']:>8.4f}")

rng = np.random.default_rng(0)
A, B = ARMS[0][0], ARMS[1][0]
pairs = [(A, B, f"{A} - {B}")]
if len(ARMS) > 2 and ARMS[2][0] in per_case:
    pairs.append((ARMS[2][0], B, f"{ARMS[2][0]} - {B}"))
for X, Y, label in pairs:
  print("\nisolated effect, %s, paired over 40 cases:" % label)
  for k in ("rmse", "angle", "div_rmse", "vort_rmse", "vort_corr", "crps_cal"):
    d = per_case[X][k] - per_case[Y][k]
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(6000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"   {k:<11}{d.mean():+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  "
          f"{'SIGNIFICANT' if lo * hi > 0 else 'TIED'}")

torch.save({"rows": rows, "per_case": per_case,
            "meta": {"seed": SEED, "n_cases": n, "n_draws": 10,
                     "epochs": args.epochs, "arms": [str(q) for _, q in ARMS],
                     "benchmark_md5": "44b866540296490c615be13bacb4242e",
                     "note": args.note}},
           ROOT / args.out)
print(f"\nsaved {args.out}")
