"""Head-to-head comparison of every predictor in this library on one metric suite.

Any model implementing the library contract -- ``predict(observations, priors=...)
-> (mean, uncertainty)``, both ``(44, 94, 2)`` in m/s -- can be dropped in and scored
identically, including models contributed by collaborators. That is the point of
this script: one command produces a like-for-like table.

    python scripts/compare_models.py --pickle /path/to/data_raw_chrono.pickle \
        --n-frames 40 --models vcnn corrdiff stream

Fairness rules enforced here (deviating from them invalidates the comparison):

  * every model sees the SAME frames, the SAME observation set and the SAME priors;
  * all models are scored on a COMMON ocean mask -- the intersection of every
    model's own mask -- so nobody is credited for cells another calls land;
  * deterministic and probabilistic models are compared with CRPS, which reduces
    exactly to mean absolute error when the uncertainty is zero, so no model is
    advantaged by the choice of score.

Adding a collaborator's model: implement the contract, then register it in
``MODEL_REGISTRY`` below with a one-line factory. Nothing else needs to change.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddpm_library import metrics                                    # noqa: E402
from ddpm_library.config import OCEAN_H, OCEAN_W                    # noqa: E402


# --------------------------------------------------------------------------- #
# Model registry -- add collaborator models here.
# Each entry: name -> (factory, needs_priors, predict_kwargs)
# --------------------------------------------------------------------------- #
def _vcnn(device):
    from ddpm_library import VCNN
    return VCNN(device=device)


def _corrdiff(device):
    from ddpm_library import CorrDiff
    return CorrDiff(device=device)


def _stream(device):
    from ddpm_library import StreamDDPM
    return StreamDDPM(device=device)


def _ddpm(device):
    from ddpm_library import DDPM
    return DDPM(device=device)


def _repaint(device):
    from ddpm_library import RePaint
    return RePaint(device=device)


MODEL_REGISTRY = {
    # name        factory     needs_priors  predict kwargs
    "vcnn":      (_vcnn,      False, {}),
    "corrdiff":  (_corrdiff,  True,  {"n_draws": 20}),
    "stream":    (_stream,    True,  {"n_draws": 20}),
    "ddpm":      (_ddpm,      False, {}),
    "repaint":   (_repaint,   True,  {"n_draws": 10}),   # collaborator model
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_frames(pickle_path, n_frames, lags=(13, 25), n_obs=90, seed=0):
    """Yield (truth, observations, priors, observed_mask) tuples from the dataset.

    ``truth`` and the priors are (44, 94, 2) m/s in library orientation. Observations
    are drawn along a contiguous pseudo-track so the sparsity pattern resembles a
    vehicle transect rather than scattered points -- a uniformly random sample is a
    much easier problem and would flatter every model equally but unrealistically.
    """
    import pickle

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    # accept either a dict of splits or a raw array
    arr = data["test"] if isinstance(data, dict) and "test" in data else data
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"expected a (T, H, W, 2) array; got shape {arr.shape}")
    if arr.shape[1:3] != (OCEAN_H, OCEAN_W):
        if arr.shape[1:3] == (OCEAN_W, OCEAN_H):
            arr = np.transpose(arr, (0, 2, 1, 3))
        else:
            raise ValueError(f"unexpected grid {arr.shape[1:3]}")

    from ddpm_library.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    lats = np.linspace(LAT_MIN, LAT_MAX, OCEAN_H)
    lons = np.linspace(LON_MIN, LON_MAX, OCEAN_W)

    rng = np.random.default_rng(seed)
    lo = max(lags)
    usable = np.arange(lo, arr.shape[0])
    picks = rng.choice(usable, size=min(n_frames, len(usable)), replace=False)

    for t in sorted(int(x) for x in picks):
        truth = np.nan_to_num(arr[t]).astype(np.float32)
        priors = [np.nan_to_num(arr[t - L]).astype(np.float32) for L in lags]

        # contiguous random-walk track of observation cells
        r = int(rng.integers(0, OCEAN_H)); c = int(rng.integers(0, OCEAN_W))
        cells, seen = [], set()
        while len(cells) < n_obs:
            if (r, c) not in seen:
                seen.add((r, c)); cells.append((r, c))
            r = int(np.clip(r + rng.integers(-1, 2), 0, OCEAN_H - 1))
            c = int(np.clip(c + rng.integers(-1, 2), 0, OCEAN_W - 1))
        obs, omask = [], np.zeros((OCEAN_H, OCEAN_W), bool)
        for (rr, cc) in cells:
            u, v = truth[rr, cc]
            obs.append((float(lats[rr]), float(lons[cc]), 1_700_000_000.0 + t * 3600,
                        float(u), float(v)))
            omask[rr, cc] = True
        yield truth, obs, priors, omask


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pickle", required=True, help="chronological dataset pickle")
    ap.add_argument("--models", nargs="+", default=["vcnn", "corrdiff"],
                    choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--n-obs", type=int, default=90)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None, help="also write results as JSON")
    args = ap.parse_args()

    print(f"loading models: {', '.join(args.models)}")
    models = {}
    for name in args.models:
        factory, needs_priors, kwargs = MODEL_REGISTRY[name]
        models[name] = (factory(args.device), needs_priors, kwargs)

    # Common ocean mask: the intersection over all models that expose one.
    common = np.ones((OCEAN_H, OCEAN_W), bool)
    for name, (mdl, _, _) in models.items():
        if hasattr(mdl, "ocean_mask"):
            common &= np.asarray(mdl.ocean_mask) > 0.5
    print(f"common ocean mask: {common.sum()} cells "
          f"({common.mean() * 100:.1f}% of the grid)")

    frames = list(load_frames(args.pickle, args.n_frames, n_obs=args.n_obs,
                              seed=args.seed))
    print(f"scoring {len(frames)} frames\n")

    # climatology for the anomaly correlation
    clim = np.mean([t for t, _, _, _ in frames], axis=0)

    acc = {name: [] for name in models}
    for i, (truth, obs, priors, omask) in enumerate(frames):
        for name, (mdl, needs_priors, kwargs) in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if needs_priors:
                    mean, unc = mdl.predict(obs, priors, **kwargs)
                else:
                    mean, unc = mdl.predict(obs, **kwargs)
            acc[name].append(metrics.evaluate(
                mean, unc, truth, ocean_mask=common,
                observed_mask=omask, climatology=clim))
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(frames)}", flush=True)

    keys = ["crps", "rmse", "rmse_observed", "rmse_unobserved", "angle_error",
            "spread_skill_ratio", "coverage_90", "ke_ratio_small",
            "fss_skilful_scale", "eddy_hit_rate", "ssim", "anomaly_correlation"]
    summary = {n: {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
               for n, rows in acc.items()}

    hdr = f"{'model':<12}" + "".join(f"{k[:9]:>11}" for k in keys)
    print("\n" + hdr)
    print("-" * len(hdr))
    for name in sorted(summary, key=lambda n: summary[n]["crps"]):
        print(f"{name:<12}" + "".join(f"{summary[name][k]:>11.4f}" for k in keys))
    print("\nCRPS is the headline (m/s, lower better); it equals mean absolute error "
          "for a\ndeterministic model, so all rows are directly comparable. "
          "NaN spread-skill /\ncoverage simply means that model reports no uncertainty.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"summary": summary, "n_frames": len(frames),
             "common_ocean_cells": int(common.sum()), "config": vars(args)}, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
