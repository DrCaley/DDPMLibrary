"""Evaluation metrics for ocean-current field reconstruction.

A single, shared metric suite so that ANY predictor implementing this library's
contract -- ``predict(...) -> (mean, uncertainty)``, both ``(44, 94, 2)`` in m/s --
can be scored identically and compared head to head.

Design note: the contract exposes a mean and a per-cell 1-sigma, not raw ensemble
members, so the probabilistic scores use the CLOSED-FORM GAUSSIAN CRPS. This is
deliberate and has an important property: as ``sigma -> 0`` it reduces exactly to
the absolute error, so a deterministic predictor and a probabilistic one land on
the same axis with no special-casing. (A model that can emit real ensemble members
may prefer the ensemble CRPS estimator; the two agree for Gaussian ensembles.)

The suite (see the project's METRICS_FINAL note for why each is here):

  ACCURACY      rmse, rmse_observed, rmse_unobserved, angle_error
  PROBABILISTIC crps, spread_skill_ratio, coverage_90, pit_histogram
  STRUCTURAL    ke_spectrum / ke_ratio_small, fss (+ skilful scale),
                eddy_hit_rate (Okubo-Weiss)
  SUPPORTING    ssim, anomaly_correlation, sal_structure

References
----------
Gneiting & Raftery (2007), JASA -- proper scoring rules, closed-form Gaussian CRPS.
Fortin et al. (2014), J. Hydrometeorol. -- spread-skill ratio.
Roberts & Lean (2008), Mon. Wea. Rev. -- Fractions Skill Score.
Wernli et al. (2008), Mon. Wea. Rev. -- SAL.
Okubo (1970) / Weiss (1991); Chelton et al. (2011) -- eddy identification.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.special import erf

__all__ = [
    "evaluate", "rmse", "angle_error", "crps_gaussian", "spread_skill_ratio",
    "coverage", "pit_values", "ke_spectrum", "ke_ratio_small", "fss",
    "eddy_hit_rate", "ssim", "anomaly_correlation", "sal_structure",
]

_SQRT_PI = math.sqrt(math.pi)
_SQRT_2 = math.sqrt(2.0)
Z90 = 1.6448536269514722          # standard-normal 90% central quantile


def _as_ocean(ocean_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if ocean_mask is None:
        return np.ones(shape, dtype=bool)
    m = np.asarray(ocean_mask)
    return (m > 0.5) if m.dtype != bool else m


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
def rmse(mean: np.ndarray, truth: np.ndarray, ocean_mask=None) -> float:
    """Root-mean-square vector error over ocean cells, m/s."""
    o = _as_ocean(ocean_mask, truth.shape[:2])
    d = (np.asarray(mean, float) - np.asarray(truth, float))[o]
    return float(np.sqrt((d ** 2).mean()))


def angle_error(mean: np.ndarray, truth: np.ndarray, ocean_mask=None,
                min_speed: float = 1e-6) -> float:
    """Mean angular error between predicted and true current vectors, degrees.

    Reported separately from RMSE because on a vector field RMSE is confounded by
    magnitude collapse: shrinking every vector toward zero *improves* RMSE while
    destroying directional information. The angle is scale-invariant, so it
    isolates "does the model know which way the water goes" -- which is also what
    a path planner consumes.
    """
    o = _as_ocean(ocean_mask, truth.shape[:2])
    p = np.asarray(mean, float)[o]; t = np.asarray(truth, float)[o]
    np_, nt = np.linalg.norm(p, axis=-1), np.linalg.norm(t, axis=-1)
    valid = (np_ > min_speed) & (nt > min_speed)
    if not valid.any():
        return float("nan")
    cos = (p[valid] * t[valid]).sum(-1) / (np_[valid] * nt[valid])
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).mean())


# --------------------------------------------------------------------------- #
# Probabilistic
# --------------------------------------------------------------------------- #
def crps_gaussian(mean: np.ndarray, sigma: np.ndarray, truth: np.ndarray,
                  ocean_mask=None) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution, m/s (lower better).

        CRPS(N(mu, sigma), y) = sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],
        z = (y - mu) / sigma

    Averaged over both velocity components and all ocean cells. Where sigma == 0
    this reduces exactly to |y - mu|, so deterministic predictors are scored on
    the same axis as probabilistic ones (their CRPS is simply their MAE).
    """
    o = _as_ocean(ocean_mask, truth.shape[:2])
    mu = np.asarray(mean, float)[o]
    sd = np.asarray(sigma, float)[o]
    y = np.asarray(truth, float)[o]

    out = np.abs(y - mu)                                  # sigma == 0 -> MAE
    pos = sd > 1e-12
    if pos.any():
        z = (y[pos] - mu[pos]) / sd[pos]
        cdf = 0.5 * (1.0 + erf(z / _SQRT_2))
        pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        out[pos] = sd[pos] * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / _SQRT_PI)
    return float(out.mean())


def spread_skill_ratio(mean: np.ndarray, sigma: np.ndarray, truth: np.ndarray,
                       ocean_mask=None) -> float:
    """sqrt(mean predicted variance) / RMSE of the mean. 1.0 = calibrated.

    Below 1 the model is over-confident (spread too small for its own error);
    above 1 it is under-confident. NaN for a deterministic predictor.
    """
    o = _as_ocean(ocean_mask, truth.shape[:2])
    sd = np.asarray(sigma, float)[o]
    if not np.any(sd > 1e-12):
        return float("nan")
    err = (np.asarray(mean, float) - np.asarray(truth, float))[o]
    denom = np.sqrt((err ** 2).mean())
    return float(np.sqrt((sd ** 2).mean()) / max(denom, 1e-12))


def coverage(mean: np.ndarray, sigma: np.ndarray, truth: np.ndarray,
             level: float = 0.90, ocean_mask=None) -> float:
    """Empirical coverage of the central ``level`` Gaussian interval.

    A calibrated predictor returns ~``level``. NaN for a deterministic predictor
    (a point forecast has no interval).
    """
    o = _as_ocean(ocean_mask, truth.shape[:2])
    sd = np.asarray(sigma, float)[o]
    if not np.any(sd > 1e-12):
        return float("nan")
    z = abs(float(np.sqrt(2.0) * _erfinv(level)))
    err = np.abs((np.asarray(mean, float) - np.asarray(truth, float))[o])
    return float(np.mean(err <= z * np.maximum(sd, 1e-12)))


def _erfinv(p: float) -> float:
    """Inverse error function at the central-probability level p (scipy-free)."""
    from scipy.special import erfinv
    return float(erfinv(p))


def pit_values(mean: np.ndarray, sigma: np.ndarray, truth: np.ndarray,
               ocean_mask=None) -> np.ndarray:
    """Probability-integral-transform values, the continuous rank histogram.

    Returns Phi((y - mu)/sigma) over ocean cells and both components. A calibrated
    predictor gives values UNIFORM on [0, 1]; a U-shaped histogram means
    over-confident (truth too often in the tails), a dome means over-dispersed.
    Empty array for a deterministic predictor.
    """
    o = _as_ocean(ocean_mask, truth.shape[:2])
    sd = np.asarray(sigma, float)[o]
    if not np.any(sd > 1e-12):
        return np.empty(0)
    mu = np.asarray(mean, float)[o]; y = np.asarray(truth, float)[o]
    ok = sd > 1e-12
    z = (y[ok] - mu[ok]) / sd[ok]
    return 0.5 * (1.0 + erf(z / _SQRT_2))


# --------------------------------------------------------------------------- #
# Structural
# --------------------------------------------------------------------------- #
def ke_spectrum(field: np.ndarray, ocean_mask=None, n_bins: int = 24):
    """Radially-averaged kinetic-energy spectrum E(k) of a (H, W, 2) field.

    A Hann window is applied first: the domain is non-periodic and land-zeroed, so
    an unwindowed transform leaks the edge discontinuity into high wavenumbers --
    exactly where we look for the small-scale deficit that reveals over-smoothing.

    Returns (k_centres, E), k in cycles per domain length.
    """
    f = np.asarray(field, float)
    o = _as_ocean(ocean_mask, f.shape[:2])
    u = np.where(o, f[..., 0], 0.0); v = np.where(o, f[..., 1], 0.0)
    H, W = u.shape
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    U = np.fft.fftshift(np.fft.fft2(u * win))
    V = np.fft.fftshift(np.fft.fft2(v * win))
    ke = 0.5 * (np.abs(U) ** 2 + np.abs(V) ** 2) / (H * W) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(H)) * H
    kx = np.fft.fftshift(np.fft.fftfreq(W)) * W
    kr = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
    edges = np.linspace(0.5, min(H, W) / 2.0, n_bins + 1)
    idx = np.digitize(kr.ravel(), edges) - 1
    flat = ke.ravel()
    E = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            E[b] = flat[m].mean()
    return 0.5 * (edges[:-1] + edges[1:]), E


def ke_ratio_small(mean: np.ndarray, truth: np.ndarray, ocean_mask=None,
                   n_bins: int = 24, small_frac: float = 0.5) -> float:
    """Predicted / true kinetic energy in the small-scale half of the spectrum.

    The blurriness check: 1.0 = correct fine-scale energy, < 1 = over-smoothed
    (the characteristic signature of a deterministic regression), > 1 = spurious
    small-scale noise. More discriminating than the total KE ratio, which is
    dominated by the large scales and looks fine even for a visibly blurred field.
    """
    k, Ep = ke_spectrum(mean, ocean_mask, n_bins)
    _, Et = ke_spectrum(truth, ocean_mask, n_bins)
    ok = np.isfinite(Ep) & np.isfinite(Et) & (Et > 0)
    small = ok & (k >= small_frac * k.max())
    if not small.any():
        return float("nan")
    return float(np.nansum(Ep[small]) / np.nansum(Et[small]))


def fss(mean: np.ndarray, truth: np.ndarray, ocean_mask=None, pctl: float = 75.0,
        scales=(1, 3, 5, 9, 15, 21, 31, 41)) -> dict:
    """Fractions Skill Score on speed, and the SKILFUL SCALE.

    Compares fractional coverage of "fast water" (speed above the ``pctl``
    percentile of the TRUE field) inside neighbourhoods of growing size. This is
    the standard neighbourhood answer to the double-penalty problem: a feature
    predicted slightly displaced is not punished twice, and the smallest
    neighbourhood at which the forecast becomes useful is a directly meaningful
    number -- "this reconstruction has skill at scales >= X cells".

    A percentile (not absolute m/s) threshold holds the event base rate fixed
    across frames and models, which is what makes the score comparable.

    Returns {'fss_<n>': ..., 'skilful_scale': float or nan, 'base_rate': float};
    the useful threshold is 0.5 + base_rate/2 (Roberts & Lean 2008).
    """
    o = _as_ocean(ocean_mask, truth.shape[:2])
    sp_p = np.linalg.norm(np.asarray(mean, float), axis=-1)
    sp_t = np.linalg.norm(np.asarray(truth, float), axis=-1)
    thr = np.percentile(sp_t[o], pctl)
    bp = (sp_p > thr) & o
    bt = (sp_t > thr) & o
    f0 = float(bt[o].mean())
    useful = 0.5 + f0 / 2.0

    def frac(binary, n):
        if n <= 1:
            return binary.astype(float)
        num = ndimage.uniform_filter(np.where(o, binary, 0).astype(float), size=n,
                                     mode="constant")
        den = ndimage.uniform_filter(o.astype(float), size=n, mode="constant")
        return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)

    out, skilful = {}, float("nan")
    for n in scales:
        Pp, Pt = frac(bp, n)[o], frac(bt, n)[o]
        den = np.sum(Pp ** 2) + np.sum(Pt ** 2)
        val = float(1.0 - np.sum((Pp - Pt) ** 2) / den) if den > 0 else float("nan")
        out[f"fss_{n}"] = val
        if np.isnan(skilful) and np.isfinite(val) and val >= useful:
            skilful = float(n)
    out["skilful_scale"] = skilful
    out["base_rate"] = f0
    return out


def _okubo_weiss(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Okubo-Weiss parameter W = s_n^2 + s_s^2 - w^2 (negative => rotation-dominated)."""
    dudx = np.gradient(u, axis=1); dudy = np.gradient(u, axis=0)
    dvdx = np.gradient(v, axis=1); dvdy = np.gradient(v, axis=0)
    s_n = dudx - dvdy                      # normal strain
    s_s = dvdx + dudy                      # shear strain
    w = dvdx - dudy                        # relative vorticity
    return s_n ** 2 + s_s ** 2 - w ** 2


def _detect_eddies(field: np.ndarray, ocean: np.ndarray, thresh_factor: float = 0.2,
                   min_size: int = 4):
    """Eddy cores as connected components where W < -thresh_factor * std(W).

    Returns a list of (row, col, sign) with sign = +1 cyclonic / -1 anticyclonic
    (from the mean vorticity inside the core).
    """
    u = np.where(ocean, field[..., 0], 0.0); v = np.where(ocean, field[..., 1], 0.0)
    W = _okubo_weiss(u, v)
    vort = np.gradient(v, axis=1) - np.gradient(u, axis=0)
    sd = float(W[ocean].std()) if ocean.any() else 0.0
    if sd <= 0:
        return []
    lab, n = ndimage.label((W < -thresh_factor * sd) & ocean)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < min_size:
            continue
        r, c = ndimage.center_of_mass(m)
        out.append((float(r), float(c), 1.0 if vort[m].mean() > 0 else -1.0))
    return out


def eddy_hit_rate(mean: np.ndarray, truth: np.ndarray, ocean_mask=None,
                  match_dist: float = 5.0) -> float:
    """Fraction of true eddies recovered, matched by position and rotation sense.

    Okubo-Weiss identifies rotation-dominated cores in both fields; a true eddy
    counts as recovered if a predicted core of the SAME sense lies within
    ``match_dist`` cells. Displacement-tolerant by construction, and the most
    directly meaningful structural check for a flow field -- an eddy is what a
    vehicle actually has to route around.

    NaN when the true field contains no eddies (report the mean over frames that
    do, and say how many that was).
    """
    o = _as_ocean(ocean_mask, truth.shape[:2])
    te = _detect_eddies(np.asarray(truth, float), o)
    if not te:
        return float("nan")
    pe = _detect_eddies(np.asarray(mean, float), o)
    hits = 0
    for r, c, s in te:
        if any(sp == s and math.hypot(rp - r, cp - c) <= match_dist for rp, cp, sp in pe):
            hits += 1
    return hits / len(te)


# --------------------------------------------------------------------------- #
# Supporting
# --------------------------------------------------------------------------- #
def ssim(mean: np.ndarray, truth: np.ndarray, ocean_mask=None, win: int = 7,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity computed on SPEED (a scalar field), higher better.

    Deliberately not averaged over signed u and v: SSIM's local-statistics model
    assumes an image-like non-negative field, and speed is where magnitude
    collapse shows up. Reported for comparability with the ocean-ML literature.
    """
    o = _as_ocean(ocean_mask, truth.shape[:2])
    x = np.linalg.norm(np.asarray(mean, float), axis=-1)
    y = np.linalg.norm(np.asarray(truth, float), axis=-1)
    L = float(y[o].max() - y[o].min())
    if L <= 0:
        return float("nan")
    C1, C2 = (k1 * L) ** 2, (k2 * L) ** 2
    mx = ndimage.uniform_filter(x, win); my = ndimage.uniform_filter(y, win)
    vx = ndimage.uniform_filter(x * x, win) - mx * mx
    vy = ndimage.uniform_filter(y * y, win) - my * my
    vxy = ndimage.uniform_filter(x * y, win) - mx * my
    s = ((2 * mx * my + C1) * (2 * vxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))
    return float(s[o].mean())


def anomaly_correlation(mean: np.ndarray, truth: np.ndarray, climatology: np.ndarray,
                        ocean_mask=None) -> float:
    """Pearson correlation of anomalies from ``climatology`` (the mean field).

    Correlation on the RAW fields is inflated by the mean circulation -- every
    method scores ~0.95 and the metric cannot discriminate. Removing the
    climatology makes it informative.
    """
    o = _as_ocean(ocean_mask, truth.shape[:2])
    a = (np.asarray(mean, float) - np.asarray(climatology, float))[o].ravel()
    b = (np.asarray(truth, float) - np.asarray(climatology, float))[o].ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sal_structure(mean: np.ndarray, truth: np.ndarray, ocean_mask=None,
                  coastal_buffer: int = 4) -> float:
    """SAL's Structure component on |vorticity| objects (0 = perfect).

    S > 0 means the predicted features are too flat and spread out (blurred);
    S < 0 means too peaked. Only S is returned: on a space-filling field like
    ocean vorticity, SAL's Location term degenerates (every method's centre of
    mass sits near the domain centre regardless of skill), so it is not
    informative here -- use :func:`fss` and :func:`eddy_hit_rate` for placement.

    Two properties worth knowing. S is amplitude-INVARIANT by construction:
    uniformly scaling a field leaves S at 0, so S says nothing about magnitude
    bias (that is what :func:`ke_ratio_small` is for). Consequently S cannot
    detect smoothing of a single-wavenumber field, where blurring is merely an
    amplitude change; it responds to blur only on multi-scale fields, which real
    ocean vorticity is.
    """
    o = _as_ocean(ocean_mask, truth.shape[:2])
    if coastal_buffer > 0:
        interior = o & (ndimage.distance_transform_edt(o) > coastal_buffer)
        if interior.any():
            o = interior

    def scaled_volume(f):
        u, v = np.asarray(f, float)[..., 0], np.asarray(f, float)[..., 1]
        z = np.abs(np.gradient(v, axis=1) - np.gradient(u, axis=0))
        vals = z[o]
        if vals.size == 0 or not np.isfinite(vals).any():
            return None
        thr = np.percentile(vals, 95) / 15.0            # ICP convention
        lab, n = ndimage.label((z > thr) & o)
        if n == 0:
            return None
        num = den = 0.0
        for i in range(1, n + 1):
            m = lab == i
            mass = float(np.where(m, z, 0.0).sum())
            peak = float(z[m].max())
            if mass <= 0 or peak <= 0:
                continue
            num += mass * (mass / peak); den += mass
        return (num / den) if den > 0 else None

    Vp, Vt = scaled_volume(mean), scaled_volume(truth)
    if Vp is None or Vt is None or (Vp + Vt) <= 0:
        return float("nan")
    return float((Vp - Vt) / (0.5 * (Vp + Vt)))


# --------------------------------------------------------------------------- #
# One-call evaluation
# --------------------------------------------------------------------------- #
def evaluate(mean: np.ndarray, uncertainty: np.ndarray, truth: np.ndarray,
             ocean_mask=None, observed_mask=None, climatology=None) -> dict:
    """Score one prediction against truth with the full settled suite.

    Parameters
    ----------
    mean, uncertainty : (44, 94, 2) float arrays, m/s
        Exactly what any library predictor returns. ``uncertainty`` may be all
        zeros (a deterministic predictor); probabilistic metrics then return NaN
        except CRPS, which correctly degenerates to the mean absolute error.
    truth : (44, 94, 2) float array, m/s
    ocean_mask : (44, 94), optional
        1/True = ocean. Defaults to all cells.
    observed_mask : (44, 94), optional
        1/True where an observation was supplied. Enables the observed vs
        unobserved RMSE split -- the direct check that a reconstruction honours
        the measurements it was given.
    climatology : (44, 94, 2), optional
        Mean field for the anomaly correlation. Skipped if omitted.

    Returns
    -------
    dict of scalar metrics (plus ``fss_*`` entries).
    """
    mean = np.asarray(mean, float)
    uncertainty = np.asarray(uncertainty, float)
    truth = np.asarray(truth, float)
    if mean.shape != truth.shape or uncertainty.shape != truth.shape:
        raise ValueError(
            f"mean {mean.shape}, uncertainty {uncertainty.shape} and truth "
            f"{truth.shape} must all have the same shape.")
    o = _as_ocean(ocean_mask, truth.shape[:2])

    out = {
        "rmse": rmse(mean, truth, o),
        "angle_error": angle_error(mean, truth, o),
        "crps": crps_gaussian(mean, uncertainty, truth, o),
        "spread_skill_ratio": spread_skill_ratio(mean, uncertainty, truth, o),
        "coverage_90": coverage(mean, uncertainty, truth, 0.90, o),
        "ke_ratio_small": ke_ratio_small(mean, truth, o),
        "eddy_hit_rate": eddy_hit_rate(mean, truth, o),
        "ssim": ssim(mean, truth, o),
        "sal_structure": sal_structure(mean, truth, o),
    }
    f = fss(mean, truth, o)
    out["fss_skilful_scale"] = f["skilful_scale"]
    out["fss_9"] = f["fss_9"]

    if observed_mask is not None:
        om = _as_ocean(observed_mask, truth.shape[:2]) & o
        um = (~_as_ocean(observed_mask, truth.shape[:2])) & o
        out["rmse_observed"] = rmse(mean, truth, om) if om.any() else float("nan")
        out["rmse_unobserved"] = rmse(mean, truth, um) if um.any() else float("nan")
    if climatology is not None:
        out["anomaly_correlation"] = anomaly_correlation(mean, truth, climatology, o)
    return out
