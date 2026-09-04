"""One implementation of the ablation metric suite, shared by the ablation scripts.

Six scripts had grown their own copy of this block. They agreed, but only by
inspection, and a copy is where a convention quietly drifts -- see `score.py`'s
docstring for the last time that cost us a result.

Conventions pinned here, matching what the ablation scripts already computed so
that migrating changes no published number:

  rmse       sqrt(mean_cells(du^2 + dv^2))       -- vector RMSE, ocean cells
  angle      sqrt(mean(per-cell angle^2))        -- angle RMS, in radians
  div_rmse   central-difference divergence error, interior ocean only
  vort_rmse  central-difference curl error, interior ocean only
  vort_corr  Pearson r between predicted and true curl
  crps_cal   CRPS on the CALIBRATED sigma (never on raw -- see PAPER_NUMBERS)

NOTE on the angle: this divides by ``norm(p) * norm(t) + 1e-12`` and keeps every
ocean cell, whereas `score.py` drops cells where either vector is below 1e-6
because their direction is undefined. The two differ slightly. `score.py` is the
paper-facing reference; this module reproduces the ablation-script convention so
that arm-to-arm comparisons stay comparable with results already recorded in
`docs/`. Do not "fix" one without re-running everything that depends on it.

Calibration is split conformal: the factor is fitted on the first half of the
cases and its coverage verified on the second, so a factor is never fitted and
reported on the same data.

Two scripts deliberately keep their own copy, for reasons that are not oversight:

  stream_new_numbers.py   also reports CRPS and coverage on the RAW sigma, to show
                          what calibration buys. Adding raw variants here would grow
                          the surface for one caller.
  stream_vorticity.py     scores one case at a time inside a per-arm loop, a
                          different shape from the (n, H, W, 2) stacks used here.

Everything else imports this: corrdiff_noise_and_draws, corrdiff_noise_curve,
distattn_curldiv, n_draws_sweep, stream_magnitude_value, stream_scratch_eval. Each was
checked against its pre-migration output -- 45 metric arrays across 9 CorrDiff arms and
every array of the magnitude ablation came back bit-identical.
"""
from __future__ import annotations

import numpy as np

from _vorticity import curl, interior_ocean_mask

#: 90% two-sided normal quantile: mu +/- Z90 * sigma covers 90%.
Z90 = 1.6448536269514722
LEVEL = 0.90

KEYS = ("rmse", "angle", "div_rmse", "vort_rmse", "vort_corr", "crps_cal")


def _divergence(f: np.ndarray) -> np.ndarray:
    """Central-difference divergence of an (H, W, 2) field; edges left at zero."""
    u, v = f[..., 0], f[..., 1]
    dudx = np.zeros_like(u); dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2
    dvdy = np.zeros_like(v); dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / 2
    return dudx + dvdy


def case_scores(mean, sigma, truth, ocean, *, crps_fn, level=LEVEL, keys=KEYS):
    """Per-case metric arrays plus the calibration summary.

    ``mean``, ``sigma`` and ``truth`` are (n, H, W, 2) stacks in m/s; ``ocean`` is
    the (H, W) ocean mask. ``crps_fn`` is ``ddpm_library.metrics.crps_gaussian``,
    passed in so this module does not depend on the library.

    Returns ``(per_case, summary)``. ``per_case`` maps each requested key to an
    (n,) array for paired bootstrapping; ``summary`` carries the conformal factor,
    the held-out coverage and the mean interval width.
    """
    mean = np.asarray(mean); sigma = np.asarray(sigma); truth = np.asarray(truth)
    ocean = np.asarray(ocean, bool)
    n = len(truth); half = n // 2
    if half < 1:
        raise ValueError("need at least two cases to split-calibrate")
    Mk = interior_ocean_mask(ocean)

    out = {}
    if "rmse" in keys:
        out["rmse"] = np.array([np.sqrt(((mean[i] - truth[i])[ocean] ** 2).sum(-1).mean())
                                for i in range(n)])
    if "angle" in keys:
        ang = []
        for i in range(n):
            p, t = mean[i][ocean], truth[i][ocean]
            c = (p * t).sum(-1) / (np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1) + 1e-12)
            ang.append(np.sqrt((np.arccos(np.clip(c, -1, 1)) ** 2).mean()))
        out["angle"] = np.array(ang)
    if "div_rmse" in keys:
        out["div_rmse"] = np.array([
            np.sqrt(((_divergence(mean[i]) - _divergence(truth[i]))[Mk] ** 2).mean())
            for i in range(n)])
    if "vort_rmse" in keys or "vort_corr" in keys:
        vr, vc = [], []
        for i in range(n):
            pc, tc = curl(mean[i])[Mk], curl(truth[i])[Mk]
            vr.append(np.sqrt(((pc - tc) ** 2).mean()))
            vc.append(np.corrcoef(pc, tc)[0, 1])
        if "vort_rmse" in keys:
            out["vort_rmse"] = np.array(vr)
        if "vort_corr" in keys:
            out["vort_corr"] = np.array(vc)

    # split conformal: fit the factor on the first half, verify on the second
    err, sg = np.abs(mean - truth)[:, ocean], sigma[:, ocean]
    ok = sg[:half] > 1e-9
    factor = float(np.quantile(err[:half][ok] / (Z90 * sg[:half][ok]), level))
    if "crps_cal" in keys:
        out["crps_cal"] = np.array([crps_fn(mean[i], factor * sigma[i], truth[i],
                                            ocean_mask=ocean) for i in range(n)])
    summary = {"factor": factor,
               "cov_cal": float(np.mean(err[half:] <= Z90 * factor * sg[half:])),
               "width": float(np.mean(2 * Z90 * factor * sg[half:]))}
    return out, summary


HEADER = (f"{'variant':<34}{'RMSE':>9}{'angle':>9}{'divRMSE':>9}{'vortRMSE':>10}"
          f"{'vortCorr':>10}{'CRPScal':>9}{'factor':>8}{'covcal':>8}{'width':>8}")


def row(label, per_case, summary) -> str:
    """One fixed-width table row; '-' where a metric was not requested."""
    def g(k, w, p):
        return f"{np.mean(per_case[k]):>{w}.{p}f}" if k in per_case else f"{'-':>{w}}"
    return (f"{label:<34}{g('rmse',9,5)}{g('angle',9,5)}{g('div_rmse',9,5)}"
            f"{g('vort_rmse',10,5)}{g('vort_corr',10,3)}{g('crps_cal',9,4)}"
            f"{summary['factor']:>8.3f}{summary['cov_cal']:>8.4f}{summary['width']:>8.4f}")


def paired_bootstrap(A, B, label, *, keys=KEYS, n_boot=6000, seed=0):
    """Print the paired difference A - B with a percentile bootstrap CI per metric."""
    rng = np.random.default_rng(seed)
    ks = [k for k in keys if k in A and k in B]
    n = len(A[ks[0]])
    print(f"\n{label}, paired over {n} cases:")
    verdicts = {}
    for k in ks:
        d = np.asarray(A[k]) - np.asarray(B[k])
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = lo * hi > 0
        verdicts[k] = {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
                       "significant": bool(sig)}
        print(f"   {k:<11}{d.mean():+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  "
              f"{'SIGNIFICANT' if sig else 'TIED'}")
    return verdicts
