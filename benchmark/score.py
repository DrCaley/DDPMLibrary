"""Reference metrics for ocean_bench_v1. Import these rather than reimplementing.

Every convention that has caused a mismatch between our results is pinned here in
code instead of prose:

  rmse_vector     sqrt(mean_cells(du^2 + dv^2))     -- vector magnitude
  rmse_component  sqrt(mean over cells AND components) = rmse_vector / sqrt(2)
  angle_mean_rad  mean per-cell angle between predicted and true vectors
  angle_rms_rad   sqrt(mean of the SQUARED per-cell angle)

Both RMSE conventions are reported because they differ by a factor of 1.414 and
that alone accounted for most of one earlier disagreement. Likewise both angle
statistics: "angular RMSE" is ambiguous between the mean and the RMS, and they
rank models differently.

Usage
-----
    python score.py predictions.npz

where predictions.npz holds `predictions` of shape (n_cases, 44, 94, 2) in m/s,
ordered exactly as the benchmark's cases, u in [..., 0] and v in [..., 1].
Land cells are ignored, so their value does not matter.

    import numpy as np, score
    b = np.load("ocean_bench_v1.npz")
    preds = np.stack([my_model(obs) for obs in b["observations"]])
    print(score.report({"my_model": preds}, b))
"""
from __future__ import annotations

import sys

import numpy as np

METRICS = ("rmse_vector", "rmse_component", "angle_mean_rad", "angle_rms_rad",
           "angle_mean_deg")


def case_metrics(pred: np.ndarray, truth: np.ndarray,
                 ocean_mask: np.ndarray) -> dict:
    """All metrics for one (44, 94, 2) prediction against one truth field."""
    pred = np.asarray(pred, np.float64)
    truth = np.asarray(truth, np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction {pred.shape} != truth {truth.shape}")
    ok = np.asarray(ocean_mask, bool)
    d = (pred - truth)[ok]
    p, t = pred[ok], truth[ok]
    pn, tn = np.linalg.norm(p, axis=-1), np.linalg.norm(t, axis=-1)
    # Cells where either vector is ~zero have no defined direction; excluding
    # them stops near-zero cells from injecting arbitrary angles.
    v = (pn > 1e-6) & (tn > 1e-6)
    a = np.arccos(np.clip((p[v] * t[v]).sum(-1) / (pn[v] * tn[v]), -1.0, 1.0))
    return {"rmse_vector": float(np.sqrt((d ** 2).sum(-1).mean())),
            "rmse_component": float(np.sqrt((d ** 2).mean())),
            "angle_mean_rad": float(a.mean()),
            "angle_rms_rad": float(np.sqrt((a ** 2).mean())),
            "angle_mean_deg": float(np.degrees(a).mean())}


def score_model(predictions: np.ndarray, bench) -> dict:
    """Per-case metrics for one model. Returns metric -> array over cases."""
    preds = np.asarray(predictions)
    truth, ok = bench["truth"], bench["ocean_mask"]
    if len(preds) != len(truth):
        raise ValueError(f"got {len(preds)} predictions for {len(truth)} cases")
    rows = [case_metrics(preds[i], truth[i], ok) for i in range(len(truth))]
    return {m: np.array([r[m] for r in rows]) for m in METRICS}


def bootstrap_ci(x, level=0.95, n=20000, seed=0):
    """(mean, lo, hi) by resampling CASES -- cells within a case are correlated,
    so treating them as independent gives indefensibly narrow intervals."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (float("nan"),) * 3
    d = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n)])
    tail = (1.0 - level) / 2.0 * 100.0
    return float(x.mean()), float(np.percentile(d, tail)), \
        float(np.percentile(d, 100.0 - tail))


def report(models: dict, bench) -> str:
    """Formatted table: mean and median per metric, for each model."""
    scored = {n: score_model(p, bench) for n, p in models.items()}
    w = max([len(n) for n in scored] + [12]) + 2
    out = [f"{int(np.asarray(bench['ocean_mask']).sum())} scored cells, "
           f"{len(bench['truth'])} cases, "
           f"{float(np.mean(bench['collection_span_hours'])):.2f} h collection span",
           "", f"{'model':<{w}}" + "".join(f"{m:>22}" for m in METRICS),
           f"{'':<{w}}" + f"{'mean / median':>22}" * len(METRICS),
           "-" * (w + 22 * len(METRICS))]
    for n, s in scored.items():
        out.append(f"{n:<{w}}" + "".join(
            f"{s[m].mean():>11.4f} /{np.median(s[m]):>10.4f}" for m in METRICS))
    if len(scored) > 1:
        out += ["", "paired differences, 95% CI over cases (negative = row better):"]
        names = list(scored)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                for m in ("rmse_vector", "angle_rms_rad"):
                    d, lo, hi = bootstrap_ci(scored[a][m] - scored[b][m])
                    tag = "significant" if (lo > 0) == (hi > 0) else "TIED"
                    out.append(f"  {m:<15}{a} - {b}: {d:+.4f} "
                               f"CI [{lo:+.4f}, {hi:+.4f}]  {tag}")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    bench = np.load("ocean_bench_v1.npz")
    models = {}
    for path in sys.argv[1:]:
        f = np.load(path)
        models[path.rsplit("/", 1)[-1].replace(".npz", "")] = f["predictions"]
    print(report(models, bench))
