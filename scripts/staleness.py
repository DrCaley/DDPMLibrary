"""Non-simultaneous observations: measure the cost, and what fixes it.

A real vehicle needs over an hour to collect a 90-cell transect, and the field
moves further than any model's error in that time -- so the usual benchmark,
which feeds every observation the target frame's value, scores a task that cannot
physically occur. Findings and numbers: ``docs/STALENESS_FINDINGS.md``.

Four subcommands, one investigation:

  eval         score models on identical tracks under simultaneous vs realistic
               observations. With --ablate-age, also hide the observation ages to
               separate "knows how stale a reading is" from "does not trust
               readings much" -- a model that never trusted them cannot be hurt.
  dial         sweep CorrDiff's sensor-noise dial under stale observations: does
               a knob for observation error absorb staleness?
  recalibrate  refit the conformal factor for stale observations by split
               conformal, and verify coverage out-of-sample.
  corrector    fit a forward-projector on (reading, age, priors) and test whether
               correcting observations BEFORE inference helps.

Every subcommand uses PAIRED SEEDS: each condition samples identical diffusion
noise and differs only in the observations. Independent draws leave a ~0.001 CRPS
Monte-Carlo floor, larger than several of the effects here.

    python scripts/staleness.py eval --pickle data_raw_chrono.pickle \\
        --frames-file scripts/fair_eval_frames.json --models corrdiff repaint
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddpm_library import metrics                                     # noqa: E402
from ddpm_library.config import (CORRDIFF_SIGMA_SCALE, LAT_MAX,      # noqa: E402
                                 LAT_MIN, LON_MAX, LON_MIN, OCEAN_H, OCEAN_W)

from _harness import (LAGS, bootstrap_ci, common_ocean_mask,         # noqa: E402
                      frame_pool, interp_at, load_fields, lookback_ok,
                      make_track, track_ages)

_LATS = np.linspace(LAT_MIN, LAT_MAX, OCEAN_H)
_LONS = np.linspace(LON_MIN, LON_MAX, OCEAN_W)

#: Conditions, in report order. `stale_noage` is the age ablation: identical
#: stale values, every timestamp collapsed so all ages read zero.
CONDITIONS = ("clean", "stale", "stale_noage")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_models(names, device, repaint_stride=1, corrdiff_weights=None,
                 calibrate=True):
    """name -> (model, needs_priors, predict_kwargs)."""
    import ddpm_library as L
    cd_kw = {"n_draws": 20, "calibrate": calibrate}
    rp_kw = {"n_draws": 10, "stride": repaint_stride}
    factories = {
        "vcnn":     (lambda: L.VCNN(device=device), False, {}),
        "gp":       (lambda: L.GP(), False, {}),
        "corrdiff": (lambda: L.CorrDiff(device=device, weights_path=corrdiff_weights),
                     True, cd_kw),
        "distattn": (lambda: L.DistAttn(device=device), False, {"n_draws": 10}),
        "repaint":  (lambda: L.RePaint(device=device), True, rp_kw),
        "repaint_uncond": (lambda: L.RePaintUncond(device=device), False, rp_kw),
    }
    bad = set(names) - set(factories)
    if bad:
        raise SystemExit(f"unknown model(s) {sorted(bad)}; have {sorted(factories)}")
    import inspect
    built = {}
    for n in names:
        mdl = factories[n][0]()
        kw = dict(factories[n][2])
        # Deterministic models (VCNN) take no seed; paired sampling is automatic
        # for them, so only pass it where it exists.
        if "seed" in inspect.signature(mdl.predict).parameters:
            kw["_takes_seed"] = True
        built[n] = (mdl, factories[n][1], kw)
    return built


# --------------------------------------------------------------------------- #
# One evaluation case
# --------------------------------------------------------------------------- #
def make_case(arr, t, common, n_obs, block, rng, straight_bias):
    """Everything needed to score one frame, or None if the frame is unusable.

    Returns a dict with the truth, priors, observed mask, and three observation
    lists that differ ONLY in their values and timestamps:

      clean        every reading taken from the target frame (the status quo)
      stale        each reading taken from the field interpolated to the moment
                   the vehicle actually reached that cell
      stale_noage  the same stale values with every timestamp set to the end of
                   the run, so a timestamp-aware model cannot tell they are old

    Ages come from the walk's own step count, not from list position: the walk
    revisits cells, so position understates elapsed time (see _harness).
    """
    cells, steps = make_track(common, n_obs, rng, return_steps=True,
                              straight_bias=straight_bias)
    ages = track_ages(len(cells), steps)                     # hours
    if not lookback_ok(t, float(ages.max()), block):
        return None

    truth = arr[t]
    t_end = 1_700_000_000.0 + t * 3600.0
    obs = {c: [] for c in CONDITIONS}
    for k, (rr, cc) in enumerate(cells):
        lat, lon = float(_LATS[rr]), float(_LONS[cc])
        u_c, v_c = truth[rr, cc]
        obs["clean"].append((lat, lon, t_end, float(u_c), float(v_c)))
        fld = interp_at(arr, t - ages[k])          # ages are already HOURS
        u_s, v_s = fld[rr, cc]
        obs["stale"].append((lat, lon, t_end - float(ages[k]) * 3600.0,
                             float(u_s), float(v_s)))
        obs["stale_noage"].append((lat, lon, t_end, float(u_s), float(v_s)))

    omask = np.zeros((OCEAN_H, OCEAN_W), bool)
    for rr, cc in cells:
        omask[rr, cc] = True
    # Sanity: the simulation must actually make the observations stale. A unit
    # slip here (ages are HOURS, not seconds) silently produces a no-op and the
    # whole experiment reports "no effect".
    drift = float(np.abs(np.array([o[3:] for o in obs["stale"]])
                         - np.array([o[3:] for o in obs["clean"]])).mean())
    if ages.max() > 0.25 and drift < 1e-4:
        raise RuntimeError(
            f"staleness simulation is a no-op: span {ages.max():.2f} h but mean "
            f"observation drift only {drift:.2e} m/s -- check the age units")
    return {"truth": truth, "priors": [arr[t - L] for L in LAGS], "omask": omask,
            "obs": obs, "cells": cells, "ages": ages, "span_h": float(ages.max())}


def score(mdl, needs_priors, kw, case, cond, seed, **extra):
    """Score one model on one condition. `seed` is shared across conditions."""
    args = (case["obs"][cond], case["priors"]) if needs_priors else (case["obs"][cond],)
    call = {k: v for k, v in {**kw, **extra}.items() if k != "_takes_seed"}
    if kw.get("_takes_seed"):
        call["seed"] = seed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, unc = mdl.predict(*args, **call)
    return metrics.evaluate(mean, unc, case["truth"], ocean_mask=case["_common"],
                            observed_mask=case["omask"])


def cases(args, common, arr, block, conds):
    """Yield usable cases, reporting how many frames were skipped."""
    rng = np.random.default_rng(args.seed)
    picks = frame_pool(args.frames_file, args.n_frames, arr.shape[0], rng)
    spans, skipped = [], 0
    for i, t in enumerate(picks):
        case = make_case(arr, t, common, args.n_obs, block, rng, args.straight_bias)
        if case is None:
            skipped += 1
            continue
        case["_common"], case["_seed"], case["_i"] = common, 5000 + i, i
        case["obs"] = {c: case["obs"][c] for c in conds}
        spans.append(case["span_h"])
        yield case
    print(f"\n  track span {np.mean(spans):.2f} h (max {np.max(spans):.2f}), "
          f"{len(spans)} frames scored, {skipped} skipped", flush=True)


KEYS = ("crps", "rmse", "rmse_observed", "rmse_unobserved", "angle_error",
        "spread_skill_ratio", "coverage_90", "ke_ratio_small")


def report(acc, conds, baseline="clean"):
    """Print a condition x metric table plus paired CIs against `baseline`."""
    print(f"\n  {'model':<16}{'condition':<13}" + "".join(f"{k[:9]:>12}" for k in KEYS))
    print("  " + "-" * (29 + 12 * len(KEYS)))
    summary = {}
    for name, per_cond in acc.items():
        for c in conds:
            rows = per_cond[c]
            if not rows:
                continue
            summary[f"{name}|{c}"] = {k: float(np.nanmean([r[k] for r in rows]))
                                      for k in KEYS}
            print(f"  {name:<16}{c:<13}"
                  + "".join(f"{summary[f'{name}|{c}'][k]:>12.5f}" for k in KEYS))
    if baseline in conds:
        print(f"\n  paired CRPS change vs '{baseline}' (positive = worse):")
        for name, per_cond in acc.items():
            base = np.array([r["crps"] for r in per_cond[baseline]])
            for c in conds:
                if c == baseline or not per_cond[c]:
                    continue
                mu, lo, hi = bootstrap_ci(np.array([r["crps"] for r in per_cond[c]]) - base)
                sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "ns"
                print(f"    {name:<16}{c:<13}{mu:+.5f}  95% CI "
                      f"[{lo:+.5f}, {hi:+.5f}]  {sig}")
    return summary


def dump(args, payload):
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_eval(args, arr, block, models, common):
    conds = ("clean", "stale", "stale_noage") if args.ablate_age else ("clean", "stale")
    acc = {n: {c: [] for c in conds} for n in models}
    for case in cases(args, common, arr, block, conds):
        for name, (mdl, needs, kw) in models.items():
            for c in conds:
                acc[name][c].append(score(mdl, needs, kw, case, c, case["_seed"]))
    summary = report(acc, conds)
    dump(args, {"summary": summary, "conditions": list(conds),
                "per_frame": {f"{n}|{c}": {k: [r[k] for r in acc[n][c]] for k in KEYS}
                              for n in acc for c in conds},
                "config": vars(args)})


def cmd_dial(args, arr, block, models, common):
    """Does a knob for observation NOISE absorb observation STALENESS?"""
    sigmas = [float(s) for s in args.sigmas]
    conds = ("clean", "stale")
    acc = {f"corrdiff@{s:.2f}": {c: [] for c in conds} for s in sigmas}
    mdl, needs, kw = models["corrdiff"]
    for case in cases(args, common, arr, block, conds):
        for s in sigmas:
            for c in conds:
                acc[f"corrdiff@{s:.2f}"][c].append(
                    score(mdl, needs, kw, case, c, case["_seed"], sensor_noise=s))
    summary = report(acc, conds)
    print("\n  Staleness is spatially COHERENT while this dial injects iid noise,")
    print("  so it is the wrong error structure, not just the wrong magnitude.")
    dump(args, {"summary": summary, "sigmas": sigmas, "config": vars(args)})


def cmd_recalibrate(args, arr, block, models, common):
    """Refit the conformal factor for stale observations; verify out-of-sample."""
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + args.level / 2.0))
    mdl, needs, kw = models["corrdiff"]
    kw = {k: v for k, v in kw.items() if k != "_takes_seed"}
    kw["calibrate"] = False                          # fit on the RAW spread
    conds = ("clean", "stale")
    per_frame = {c: [] for c in conds}
    for case in cases(args, common, arr, block, conds):
        for c in conds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean, raw = mdl.predict(case["obs"][c], case["priors"],
                                        seed=case["_seed"], **kw)
            oc = case["_common"]
            e, s = np.abs(mean - case["truth"])[oc], raw[oc]
            keep = s > 1e-9
            per_frame[c].append((e[keep], s[keep]))

    n = len(per_frame["clean"]); half = n // 2
    print(f"  fitting on {half} frames, verifying on {n - half}; "
          f"z_{args.level:.2f}={z:.4f}, shipped factor {CORRDIFF_SIGMA_SCALE}\n")
    print(f"  {'observations':<16}{'factor':>10}{'cov (fit)':>12}"
          f"{'cov (held out)':>16}{'width':>10}")
    out = {}
    for c in conds:
        fit, ver = per_frame[c][:half], per_frame[c][half:]
        ratios = np.concatenate([e.ravel() / (z * s.ravel()) for e, s in fit])
        scale = float(np.quantile(ratios, args.level))
        cov = lambda rows: float(np.mean(np.concatenate(
            [(e.ravel() <= z * scale * s.ravel()) for e, s in rows])))
        width = float(np.mean(np.concatenate(
            [2 * z * scale * s.ravel() for _, s in ver])))
        out[c] = {"scale": scale, "coverage_fit": cov(fit),
                  "coverage_holdout": cov(ver), "mean_width": width}
        print(f"  {c:<16}{scale:>10.4f}{out[c]['coverage_fit']:>12.4f}"
              f"{out[c]['coverage_holdout']:>16.4f}{width:>10.4f}")
    r = out["stale"]["scale"] / out["clean"]["scale"]
    print(f"\n  stale observations need {r:.2f}x the factor -- intervals that much wider.")
    print(f"  Pass it as CorrDiff.predict(..., sigma_scale=...); the shipped default")
    print(f"  assumes simultaneity and under-covers on real vehicle data.")
    dump(args, {"level": args.level, "z": z, "n_frames": n, "results": out,
                "config": vars(args)})


def cmd_corrector(args, arr, block, models, common):
    """Correct the observations before inference, instead of changing the model."""
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB

    def rows(frames_seed, n):
        """Features available at inference: reading, age, both priors at that cell."""
        rng = np.random.default_rng(frames_seed)
        picks = frame_pool(args.frames_file, n, arr.shape[0], rng)
        X, Y = [], []
        for t in picks:
            case = make_case(arr, t, common, args.n_obs, block, rng, args.straight_bias)
            if case is None:
                continue
            idx = np.array(case["cells"])
            st = np.array([[o[3], o[4]] for o in case["obs"]["stale"]], np.float32)
            p13 = case["priors"][0][idx[:, 0], idx[:, 1]]
            p25 = case["priors"][1][idx[:, 0], idx[:, 1]]
            X.append(_feats(st, case["ages"], p13, p25))
            Y.append(case["truth"][idx[:, 0], idx[:, 1]])
        return np.concatenate(X), np.concatenate(Y)

    Xf, Yf = rows(10_000, args.n_fit)
    corr = [HGB(max_iter=200, random_state=0).fit(Xf, Yf[:, k]) for k in (0, 1)]
    ins = np.sqrt(((np.stack([c.predict(Xf) for c in corr], 1) - Yf) ** 2).sum(1).mean())
    print(f"  corrector fitted on {Xf.shape[0]} rows; in-sample obs RMSE {ins:.5f} "
          f"vs raw {np.sqrt(((Xf[:, :2] - Yf) ** 2).sum(1).mean()):.5f}", flush=True)

    conds = ("clean", "stale", "corrected")
    mdl, needs, kw = models["corrdiff"]
    acc = {"corrdiff": {c: [] for c in conds}}
    obs_err = {"stale": [], "corrected": []}
    for case in cases(args, common, arr, block, ("clean", "stale")):
        idx = np.array(case["cells"])
        st = np.array([[o[3], o[4]] for o in case["obs"]["stale"]], np.float32)
        p13 = case["priors"][0][idx[:, 0], idx[:, 1]]
        p25 = case["priors"][1][idx[:, 0], idx[:, 1]]
        fixed = np.stack([c.predict(_feats(st, case["ages"], p13, p25))
                          for c in corr], 1).astype(np.float32)
        tru = case["truth"][idx[:, 0], idx[:, 1]]
        obs_err["stale"].append(float(np.sqrt(((st - tru) ** 2).sum(1).mean())))
        obs_err["corrected"].append(float(np.sqrt(((fixed - tru) ** 2).sum(1).mean())))
        case["obs"]["corrected"] = [
            (o[0], o[1], o[2], float(fixed[k, 0]), float(fixed[k, 1]))
            for k, o in enumerate(case["obs"]["stale"])]
        for c in conds:
            acc["corrdiff"][c].append(score(mdl, needs, kw, case, c, case["_seed"]))

    print(f"\n  observation RMSE out-of-sample: stale {np.mean(obs_err['stale']):.5f} "
          f"-> corrected {np.mean(obs_err['corrected']):.5f} "
          f"({100 * (1 - np.mean(obs_err['corrected']) / np.mean(obs_err['stale'])):+.2f}%)")
    summary = report(acc, conds, baseline="stale")
    dump(args, {"summary": summary,
                "obs_rmse": {k: float(np.mean(v)) for k, v in obs_err.items()},
                "config": vars(args)})


def _feats(stale, ages, p13, p25):
    """The corrector's features -- all obtainable at inference time."""
    su, sv, a = stale[:, 0], stale[:, 1], np.asarray(ages, np.float32)
    return np.stack([su, sv, p13[:, 0], p13[:, 1], p25[:, 0], p25[:, 1],
                     a, a * su, a * sv,
                     a * (p13[:, 0] - su), a * (p13[:, 1] - sv)], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("eval", "dial", "recalibrate", "corrector"))
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--frames-file", default=None,
                    help="restrict to frames no model was trained on "
                         "(scripts/fair_eval_frames.json)")
    ap.add_argument("--models", nargs="+", default=["corrdiff"])
    ap.add_argument("--n-frames", type=int, default=40)
    ap.add_argument("--n-obs", type=int, default=90)
    ap.add_argument("--straight-bias", type=float, default=0.75,
                    help="track directional persistence. 0.75 matches the walk the "
                         "models were TRAINED on and keeps revisits low so ages are "
                         "coherent; 0.0 reproduces the earlier published runs.")
    ap.add_argument("--repaint-stride", type=int, default=1,
                    help="RePaint reverse-chain stride; >1 keeps a paired comparison "
                         "valid while making the run affordable")
    ap.add_argument("--corrdiff-weights", default=None,
                    help="override the CorrDiff checkpoint (e.g. a fine-tuned one)")
    ap.add_argument("--ablate-age", action="store_true",
                    help="eval: also score with observation ages hidden")
    ap.add_argument("--sigmas", nargs="+",
                    default=["0.0", "0.02", "0.04", "0.06", "0.08", "0.10"])
    ap.add_argument("--level", type=float, default=0.90)
    ap.add_argument("--n-fit", type=int, default=600,
                    help="corrector: frames used to fit, disjoint from evaluation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    needed = {"dial": ["corrdiff"], "recalibrate": ["corrdiff"],
              "corrector": ["corrdiff"]}.get(args.cmd)
    if needed and args.models != needed:
        args.models = needed
        print(f"  ({args.cmd} operates on CorrDiff; --models ignored)")

    print(f"loading: {', '.join(args.models)}", flush=True)
    models = build_models(args.models, args.device,
                          repaint_stride=args.repaint_stride,
                          corrdiff_weights=args.corrdiff_weights)
    common = common_ocean_mask(m for m, _, _ in models.values())
    arr, block = load_fields(args.pickle)
    print(f"common ocean mask: {int(common.sum())} cells", flush=True)

    {"eval": cmd_eval, "dial": cmd_dial, "recalibrate": cmd_recalibrate,
     "corrector": cmd_corrector}[args.cmd](args, arr, block, models, common)


if __name__ == "__main__":
    main()
