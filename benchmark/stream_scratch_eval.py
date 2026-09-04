"""Is the shipped Stream direction net as good as one trained without the spread
term from scratch?

The spread term was removed on 2026-09-02 by shipping `StreamFn_Cond_x0_mag.pt`
(epoch 78) -- a checkpoint from the lineage that predates the term. That invites a
fair objection: it is an *earlier* checkpoint, so "no spread term" is confounded
with "less training", and the fine-tune ablation only ever removed the term from an
already-converged model.

This scores the shipped net against one trained from random init for 120k steps with
no spread term at any point, same architecture and cond_ch, `--lambda_vort 0
--lambda_angle 1 --lambda_mag 0.2 --min_snr_gamma 5`. If the from-scratch net matches
or beats the shipped one, "the term is not needed" no longer rests on a checkpoint
that merely stopped early.

Both arms share the magnitude network, sampler, seeds and masking; only the
direction checkpoint differs.

  python benchmark/stream_scratch_eval.py --ckpt scratch=/path/to/stream_scratch.pt
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmark"))
from ddpm_library import StreamDDPM, metrics                        # noqa: E402
import _score                                                        # noqa: E402

SEED = 20260830

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH",
                help="direction checkpoint to score against the shipped one; repeat")
ap.add_argument("--out", default="benchmark/results_stream_scratch.pt")
ap.add_argument("--n_draws", type=int, default=20)
args = ap.parse_args()

ARMS = [("shipped (epoch 78, no spread)", None)]
for spec in args.ckpt:
    name, sep, path = spec.rpartition("=")   # last "=": names contain "=" too
    if not sep or not name.strip() or not path.strip():
        ap.error(f"--ckpt expects NAME=PATH, got {spec!r}")
    q = Path(path.strip())
    if not q.exists():
        ap.error(f"checkpoint not found: {q}")
    ARMS.append((name.strip(), q))

b = np.load(ROOT / "benchmark/ocean_bench_v1.npz")
obs, pri, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
n = len(truth)


print(_score.HEADER)
per_case = {}
for name, ckpt in ARMS:
    st = StreamDDPM(device="mps") if ckpt is None else StreamDDPM(device="mps",
                                                                  dir_weights_path=ckpt)
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = st.predict([tuple(x) for x in obs[i]], pri[i], n_draws=args.n_draws,
                              seed=SEED + i, full_field=True, calibrate=False)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    del st
    pc, sm = _score.case_scores(np.stack(M), np.stack(S), truth, ocean,
                                crps_fn=metrics.crps_gaussian)
    print(_score.row(name, pc, sm), flush=True)
    per_case[name] = pc

base = ARMS[0][0]
for name, _ in ARMS[1:]:
    _score.paired_bootstrap(per_case[name], per_case[base], f"{name} minus {base}")

torch.save({"per_case": per_case,
            "meta": {"seed": SEED, "n_cases": n, "n_draws": args.n_draws,
                     "arms": [(nm, str(q)) for nm, q in ARMS],
                     "benchmark_md5": "44b866540296490c615be13bacb4242e"}},
           ROOT / args.out)
print(f"\nsaved {args.out}")
