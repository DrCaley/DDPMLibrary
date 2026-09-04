"""Are stale observations worth keeping at all?

CorrDiff pays +21.9% under 2 h collection because it fits observations tightly
(0.0060 residual vs DistAttn's 0.0194) and a two-hour-old reading is simply wrong
about the present field. The crudest possible trust mechanism is to throw the old
readings away. If that HELPS, it is a zero-training, deploy-tomorrow result, and
it tells the path planner how long a measurement stays worth having.

THE CONFOUND, and the control for it. Filtering by age also shrinks the spatial
extent of the observations: keeping only the last 30 minutes keeps only the last
~1.9 km of track. We already know extent matters enormously (spreading the same
90 readings over the domain instead of along a track moved GP by 76%). So a naive
age sweep confounds "fresher" with "more concentrated".

Each age cutoff is therefore paired with a MATCHED-COUNT control: the same number
of readings, but sampled evenly across the whole 2 h track, so it has the original
spatial extent and the original mixed ages. Comparing the two isolates freshness
from coverage:

    fresh<=T   vs  spread@N   ->  is freshness worth the lost extent?
    spread@N   vs  all        ->  what does dropping readings cost by itself?

Runs on ocean_bench_v1, so the observations, truth, mask and metrics are the
frozen shared ones.
"""
import sys, warnings, json
from pathlib import Path

import os                                                          # noqa: E402
#: "auto" resolves cuda / mps / cpu, so these run off the GPU box too.
DEV = os.environ.get("DDPM_DEVICE", "auto")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmark"))
import numpy as np
import score
from ddpm_library import CorrDiff, DistAttn, StreamDDPM, GP   # noqa: F401

bench = np.load(str(ROOT / "benchmark/ocean_bench_v1.npz"))
obs_all, priors_all = bench["observations"], bench["priors"]
ocean = bench["ocean_mask"]
cd, da, gp = CorrDiff(device=DEV), DistAttn(device=DEV), GP()
MODELS = {"corrdiff+priors": (cd, True,  {"n_draws": 20}),
          "corrdiff-priors": (cd, False, {"n_draws": 20}),
          "distattn":        (da, False, {"n_draws": 10}),
          "gp":              (gp, False, {})}
CUTOFFS_H = (0.5, 1.0)          # plus the full track as the baseline


def subsets(rows):
    """condition -> boolean keep-mask over the readings of one case."""
    ages = (rows[:, 2].max() - rows[:, 2]) / 3600.0
    out = {"all": np.ones(len(rows), bool)}
    for T in CUTOFFS_H:
        keep = ages <= T
        out[f"fresh<={T}h"] = keep
        n = int(keep.sum())
        # matched count, spread over the WHOLE track: same data volume, original
        # spatial extent, original mixed ages.
        idx = np.unique(np.linspace(0, len(rows) - 1, n).round().astype(int))
        ctrl = np.zeros(len(rows), bool); ctrl[idx] = True
        out[f"spread@{n}"] = ctrl
    return out


CONDS = ["all"] + [c for T in CUTOFFS_H for c in (f"fresh<={T}h",)]
preds, meta = {}, {c: {"n": [], "cells": [], "mean_age": []} for c in ["all"]}
acc = {}
for i in range(len(obs_all)):
    rows = obs_all[i]
    priors = [priors_all[i][k] for k in range(priors_all.shape[1])]
    for cond, keep in subsets(rows).items():
        sel = rows[keep]
        ages = (rows[:, 2].max() - sel[:, 2]) / 3600.0
        cells = {(round(a, 6), round(b, 6)) for a, b in sel[:, :2]}
        meta.setdefault(cond, {"n": [], "cells": [], "mean_age": []})
        meta[cond]["n"].append(len(sel))
        meta[cond]["cells"].append(len(cells))
        meta[cond]["mean_age"].append(float(ages.mean()))
        obs = [tuple(r) for r in sel]
        for n, (mdl, pri, kw) in MODELS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                call = dict(kw)
                if n != "gp":
                    call["seed"] = 5000 + i
                mean, _ = mdl.predict(*((obs, priors) if pri else (obs,)), **call)
            acc.setdefault((n, cond), []).append(
                score.case_metrics(mean, bench["truth"][i], ocean))
    print(f"  case {i + 1}/{len(obs_all)}", flush=True)

conds = sorted({c for _, c in acc}, key=lambda c: (c != "all", c))
pf = {f"{n}|{c}": {m: np.array([r[m] for r in acc[(n, c)]]) for m in score.METRICS}
      for n, c in acc}

print("\nobservation subsets (per case, averaged)")
print(f"  {'condition':<14}{'readings':>10}{'cells':>8}{'mean age h':>12}")
for c in conds:
    m = meta[c]
    print(f"  {c:<14}{np.mean(m['n']):>10.0f}{np.mean(m['cells']):>8.0f}"
          f"{np.mean(m['mean_age']):>12.2f}")

print(f"\n{'model':<18}" + "".join(f"{c:>14}" for c in conds) + "   (RMSE mean)")
for n in MODELS:
    print(f"{n:<18}" + "".join(
        f"{pf[f'{n}|{c}']['rmse_vector'].mean():>14.4f}" for c in conds))

print("\npaired vs the full track (negative = dropping readings HELPS)")
for n in MODELS:
    print(f"  {n}")
    for c in conds:
        if c == "all":
            continue
        d, lo, hi = score.bootstrap_ci(pf[f"{n}|{c}"]["rmse_vector"]
                                       - pf[f"{n}|all"]["rmse_vector"])
        base = pf[f"{n}|all"]["rmse_vector"].mean()
        tag = "significant" if (lo > 0) == (hi > 0) else "tied"
        print(f"    {c:<14}{d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"({100 * d / base:+.1f}%)  {tag}")

print("\nfreshness isolated: fresh<=T vs the SAME NUMBER spread over the whole track")
for n in MODELS:
    line = f"  {n:<18}"
    for T in CUTOFFS_H:
        f_c = f"fresh<={T}h"
        s_c = [c for c in conds if c.startswith("spread@")
               and abs(np.mean(meta[c]["n"]) - np.mean(meta[f_c]["n"])) < 1e-6]
        if not s_c:
            continue
        d, lo, hi = score.bootstrap_ci(pf[f"{n}|{f_c}"]["rmse_vector"]
                                       - pf[f"{n}|{s_c[0]}"]["rmse_vector"])
        tag = "sig" if (lo > 0) == (hi > 0) else "tied"
        line += f"  {T}h: {d:+.4f} [{lo:+.4f},{hi:+.4f}] {tag}"
    print(line)

json.dump({k: {m: list(map(float, v)) for m, v in d.items()} for k, d in pf.items()},
          open(str(ROOT / "benchmark/age_cutoff.json"), "w"), indent=2)
print("\nwrote age_cutoff.json")
