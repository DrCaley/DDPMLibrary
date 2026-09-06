"""Is each model's default ensemble size the right one?

CorrDiff's was swept and confirmed at 20 (`corrdiff_noise_and_draws.py`). The other
two were never tested: `STREAM_DEFAULT_N_DRAWS = 20` is justified in config only as
"matches CorrDiff", and DistAttn's was inherited from the collaborator's evaluation
at 10, and this sweep found 20 measurably better there (see DEFAULTS_AND_DIALS 3b for
why the default was left alone). Since CorrDiff lost significantly on all five metrics when
halved from 20 to 10, DistAttn at 10 is worth checking.

Every size is a nested subset of ONE sampling run per case: the sampler is cached, so
sizes are exactly paired and the sweep costs a single max-size ensemble. DistAttn
seeds each draw with `(seed + 1) * 100003 + k`, so its draw k does not depend on the
ensemble size at all; Stream's members come from one batched call, and slicing that call's output
gives the same nesting.

The conformal factor is refit per size -- the raw ensemble std shrinks with ensemble
size, so reusing one factor across sizes would confound spread with calibration.

  python benchmark/n_draws_sweep.py --model distattn
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmark"))
from ddpm_library import metrics, config as C                                     # noqa: E402
import _score                                                        # noqa: E402
from _paths import DEV                                               # noqa: E402

SEED = 20260830

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--model", required=True, choices=("stream", "distattn"))
ap.add_argument("--sizes", default="5,10,20,40")
ap.add_argument("--device", default=DEV)
ap.add_argument("--out", default=None)
args = ap.parse_args()
SIZES = sorted({int(s) for s in args.sizes.split(",")})
MAXD = max(SIZES)

b = np.load(ROOT / "benchmark/ocean_bench_v1.npz")
obs, pri, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
n = len(truth)

cache = {}
_case = {"i": -1}   # current case, part of the DistAttn cache key

if args.model == "stream":
    import ddpm_library.stream_predict as MOD
    from ddpm_library import StreamDDPM
    _real = MOD.dpmpp_ensemble

    def _cached(stream_model, diffusion, cond, land_np, *, n_members, **kw):
        """Draw MAXD members once per seed; hand back a prefix."""
        key = kw.get("seed")
        if key not in cache:
            cache[key] = _real(stream_model, diffusion, cond, land_np,
                               n_members=MAXD, **kw)
        return [m.copy() for m in cache[key][:n_members]]

    MOD.dpmpp_ensemble = _cached
    model = StreamDDPM(device=args.device)
    call = lambda k, i: model.predict([tuple(x) for x in obs[i]], pri[i], n_draws=k,
                                      seed=SEED + i, full_field=True, calibrate=False)
    default = C.STREAM_DEFAULT_N_DRAWS
else:
    import ddpm_library.distattn_predict as MOD
    from ddpm_library import DistAttn
    _real = MOD.ddpm_sample

    def _cached(*a, **kw):
        """Key on (case, per-draw seed).

        The seed alone is NOT a valid key. The predictor used to set `seed + k` for
        draw k, so case i draw k and case i+1 draw k-1 collided on the same value --
        and the draw depends on this case's observation tokens, so a collision
        silently served one case's field for another. Measured cost of that bug:
        DistAttn RMSE 0.125 instead of 0.073. The predictor now spreads its seeds,
        but the case index stays in the key so this cannot regress.
        """
        key = (_case["i"], torch.initial_seed())
        if key not in cache:
            cache[key] = _real(*a, **kw)
        return cache[key].clone()

    MOD.ddpm_sample = _cached
    model = DistAttn(device=args.device)

    def call(k, i):
        _case["i"] = i
        return model.predict([tuple(x) for x in obs[i]], n_draws=k,
                             seed=SEED + i, calibrate=False)
    default = C.DISTATTN_DEFAULT_N_DRAWS

print(f"{args.model}: sweeping n_draws {SIZES} (default {default}), "
      f"{n} cases, one {MAXD}-draw run per case\n")
print(_score.HEADER)
per_case, summaries = {}, {}
for k in SIZES:
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # the ensemble-size guard, expected here
            m, s = call(k, i)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    lab = f"n_draws = {k}" + ("  (shipped)" if k == default else "")
    pc, sm = _score.case_scores(np.stack(M), np.stack(S), truth, ocean,
                                crps_fn=metrics.crps_gaussian)
    per_case[k], summaries[k] = pc, sm
    print(_score.row(lab, pc, sm), flush=True)

# restore the real sampler on whichever module we patched
setattr(MOD, "dpmpp_ensemble" if args.model == "stream" else "ddpm_sample", _real)

verdicts = {}
for k in SIZES:
    if k != default:
        verdicts[k] = _score.paired_bootstrap(per_case[k], per_case[default],
                                              f"n_draws {k} minus shipped {default}")

# Stream caches one batched ensemble per case; DistAttn caches each draw separately.
# Either way a short cache means two distinct draws shared a key, and one case was
# served another case's field -- measured cost of that bug: DistAttn RMSE 0.125
# instead of 0.073. Fail loudly rather than report it.
expected = n if args.model == "stream" else n * MAXD
what = f"{n} cases" if args.model == "stream" else f"{n} cases x {MAXD} draws"
if len(cache) != expected:
    raise SystemExit(f"cache holds {len(cache)} entries, expected {expected} "
                     f"({what}) -- keys collided, results invalid")
print(f"\ncache check: {len(cache)} entries == {what}")

out = args.out or f"benchmark/results_{args.model}_n_draws.pt"
torch.save({"per_case": per_case, "summaries": summaries, "verdicts": verdicts,
            "meta": {"seed": SEED, "n_cases": n, "sizes": SIZES, "default": default,
                     "model": args.model, "device": args.device,
                     "benchmark_md5": "44b866540296490c615be13bacb4242e"}},
           ROOT / out)
print(f"\nsaved {out}")
