"""`benchmark/_score.py` is the single metric implementation behind six ablation
scripts, so a silent change there would move published numbers everywhere at once.

The first test is the important one: it reimplements the inline block those scripts
carried before the 2026-09-04 consolidation and asserts the shared module reproduces
it EXACTLY. That equivalence was verified once by hand during the migration; this is
the permanent version, so the guarantee survives future edits.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

from ddpm_library import metrics                       # noqa: E402
import _score                                          # noqa: E402
from _vorticity import curl, interior_ocean_mask       # noqa: E402


@pytest.fixture
def case():
    rng = np.random.default_rng(7)
    n, H, W = 8, 94, 44
    ocean = rng.random((H, W)) > 0.25
    truth = rng.normal(0, 0.1, (n, H, W, 2)).astype(np.float32)
    mean = (truth + rng.normal(0, 0.02, truth.shape)).astype(np.float32)
    sigma = np.abs(rng.normal(0.03, 0.005, truth.shape)).astype(np.float32)
    return mean, sigma, truth, ocean


def _inline(mean, sigma, truth, ocean):
    """The block the ablation scripts each used to carry, verbatim in spirit."""
    z, LEVEL = _score.Z90, _score.LEVEL
    n = len(truth); half = n // 2
    Mk = interior_ocean_mask(ocean)
    rmse = np.array([np.sqrt(((mean[i] - truth[i])[ocean] ** 2).sum(-1).mean()) for i in range(n)])
    ang, vr, vc, dv = [], [], [], []
    for i in range(n):
        pv, tv = mean[i][ocean], truth[i][ocean]
        c = (pv * tv).sum(-1) / (np.linalg.norm(pv, axis=-1) * np.linalg.norm(tv, axis=-1) + 1e-12)
        ang.append(np.sqrt((np.arccos(np.clip(c, -1, 1)) ** 2).mean()))
        pc, tc = curl(mean[i])[Mk], curl(truth[i])[Mk]
        vr.append(np.sqrt(((pc - tc) ** 2).mean())); vc.append(np.corrcoef(pc, tc)[0, 1])
        u, v = mean[i][..., 0], mean[i][..., 1]
        dudx = np.zeros_like(u); dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2
        dvdy = np.zeros_like(v); dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / 2
        ut, vt = truth[i][..., 0], truth[i][..., 1]
        tdx = np.zeros_like(ut); tdx[:, 1:-1] = (ut[:, 2:] - ut[:, :-2]) / 2
        tdy = np.zeros_like(vt); tdy[1:-1, :] = (vt[2:, :] - vt[:-2, :]) / 2
        dv.append(np.sqrt((((dudx + dvdy) - (tdx + tdy))[Mk] ** 2).mean()))
    err, sg = np.abs(mean - truth)[:, ocean], sigma[:, ocean]
    ok = sg[:half] > 1e-9
    f = float(np.quantile(err[:half][ok] / (z * sg[:half][ok]), LEVEL))
    crps = np.array([metrics.crps_gaussian(mean[i], f * sigma[i], truth[i], ocean_mask=ocean)
                     for i in range(n)])
    return ({"rmse": rmse, "angle": np.array(ang), "div_rmse": np.array(dv),
             "vort_rmse": np.array(vr), "vort_corr": np.array(vc), "crps_cal": crps},
            {"factor": f,
             "cov_cal": float(np.mean(err[half:] <= z * f * sg[half:])),
             "width": float(np.mean(2 * z * f * sg[half:]))})


def test_matches_the_inline_block_it_replaced(case):
    pc, sm = _score.case_scores(*case, crps_fn=metrics.crps_gaussian)
    ref_pc, ref_sm = _inline(*case)
    for k, want in ref_pc.items():
        assert np.array_equal(pc[k], want), f"{k} diverged from the inline implementation"
    for k, want in ref_sm.items():
        assert sm[k] == want, f"{k} diverged"


def test_factor_is_fitted_on_the_first_half_only(case):
    """Split conformal: perturbing only the verify half must not move the factor."""
    mean, sigma, truth, ocean = case
    base, _ = _score.case_scores(mean, sigma, truth, ocean, crps_fn=metrics.crps_gaussian)
    _, sm0 = _score.case_scores(mean, sigma, truth, ocean, crps_fn=metrics.crps_gaussian)
    sig2 = sigma.copy(); sig2[len(truth) // 2:] *= 3.0
    _, sm1 = _score.case_scores(mean, sig2, truth, ocean, crps_fn=metrics.crps_gaussian)
    assert sm1["factor"] == sm0["factor"], "verify-half data leaked into the fitted factor"
    assert sm1["cov_cal"] != sm0["cov_cal"], "coverage should respond to the verify half"


def test_keys_filter_skips_unrequested_metrics(case):
    pc, _ = _score.case_scores(*case, crps_fn=metrics.crps_gaussian, keys=("rmse",))
    assert set(pc) == {"rmse"}


def test_bootstrap_calls_a_real_effect_significant_and_noise_tied():
    n = 40
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 0.05, n)
    A = {"rmse": base}
    shifted = {"rmse": base + 0.5}                       # large consistent effect
    same = {"rmse": base + rng.normal(0, 0.05, n)}       # noise only
    v = _score.paired_bootstrap(shifted, A, "shifted", keys=("rmse",))
    assert v["rmse"]["significant"] is True
    v = _score.paired_bootstrap(same, A, "same", keys=("rmse",))
    assert v["rmse"]["significant"] is False


def test_needs_two_cases_to_split(case):
    mean, sigma, truth, ocean = case
    with pytest.raises(ValueError, match="split-calibrate"):
        _score.case_scores(mean[:1], sigma[:1], truth[:1], ocean,
                           crps_fn=metrics.crps_gaussian)
