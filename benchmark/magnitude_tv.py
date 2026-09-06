"""Does the magnitude network's TV smoothness term earn its place?

lambda_tv = 0.05 is a script default the shipped checkpoint does not record, so it
was inferred rather than read -- the last untested constant in the Stream pipeline.
The term smooths the predicted log-variance; its stated purpose is to stop the
pointwise Gaussian NLL producing a salt-and-pepper sigma map.

Two arms, 6 epochs each from the shipped magnitude checkpoint, differing only in
--smooth_weight. Both use --legacy_obs (10-channel conditioning) and
--head_hidden 0 (the logvar_conv head), matching the shipped architecture exactly:
without those the trainer silently reinitialises the input layer and the whole
uncertainty head, which would make the ablation about a different model.

Scored on what the term is for -- the spatial character of sigma -- as well as the
calibration metrics that decide whether a smoother sigma is actually better.
"""
import sys, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from _paths import DEV, MODELS_DIR  # noqa: E402
from ddpm_library import StreamDDPM, metrics                     # noqa: E402

z, LEVEL, SEED = 1.6448536269514722, 0.90, 20260830
MD = MODELS_DIR / "mag_ablate"
ARMS = [("lambda_tv = 0.05 (shipped)", MD / "magarm_tv0.05.pt"),
        ("lambda_tv = 0 (control)",    MD / "magarm_tv0.0.pt")]
ARMS = [(n, p) for n, p in ARMS if p.exists()]
print("arms found:", [n for n, _ in ARMS])

b = np.load(ROOT / "benchmark/ocean_bench_v1.npz")
obs, pri, truth = b["observations"], b["priors"], b["truth"]
ocean = np.asarray(b["ocean_mask"], bool)
n, half = len(truth), len(truth) // 2

def roughness(s):
    """Mean |difference| between neighbouring ocean cells of sigma -- directly the
    salt-and-pepper character the TV term exists to suppress."""
    tot, cnt = 0.0, 0
    for c in range(s.shape[-1]):
        v = s[..., c]
        for a, bm in ((v[:, 1:], v[:, :-1]), (v[1:, :], v[:-1, :])):
            m = ocean[:, 1:] & ocean[:, :-1] if a.shape == v[:, 1:].shape else ocean[1:, :] & ocean[:-1, :]
            tot += float(np.abs(a - bm)[m].sum()); cnt += int(m.sum())
    return tot / max(cnt, 1)

rows, per_case = {}, {}
for name, p in ARMS:
    st = StreamDDPM(device=DEV, mag_weights_path=p)
    M, S = [], []
    for i in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, s = st.predict([tuple(x) for x in obs[i]], pri[i], n_draws=20,
                              seed=SEED + i, full_field=True, calibrate=False)
        M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
    m, s = np.stack(M), np.stack(S); del st
    rmse = np.array([np.sqrt(((m[i] - truth[i])[ocean] ** 2).sum(-1).mean()) for i in range(n)])
    rough = np.array([roughness(s[i]) for i in range(n)])
    r_unc = np.array([np.corrcoef(s[i][ocean].ravel(), np.abs(m[i] - truth[i])[ocean].ravel())[0, 1]
                      for i in range(n)])
    err, sg = np.abs(m - truth)[:, ocean], s[:, ocean]
    ok = sg[:half] > 1e-9
    f = float(np.quantile(err[:half][ok] / (z * sg[:half][ok]), LEVEL))
    crps = np.array([metrics.crps_gaussian(m[i], f * s[i], truth[i], ocean_mask=ocean) for i in range(n)])
    per_case[name] = dict(rmse=rmse, roughness=rough, r_unc=r_unc, crps_cal=crps)
    rows[name] = dict(rmse=rmse.mean(), roughness=rough.mean(), r_unc=r_unc.mean(),
                      crps_cal=crps.mean(), factor=f,
                      cov_cal=float(np.mean(err[half:] <= z * f * sg[half:])),
                      width=float(np.mean(2 * z * f * sg[half:])))
    print("  scored %s" % name, flush=True)

print(f"\n{'arm':<28}{'RMSE':>9}{'sigma rough':>13}{'r(sig,err)':>12}"
      f"{'CRPScal':>9}{'factor':>8}{'covcal':>8}{'width':>8}")
for k, r in rows.items():
    print(f"{k:<28}{r['rmse']:>9.5f}{r['roughness']:>13.6f}{r['r_unc']:>12.3f}"
          f"{r['crps_cal']:>9.4f}{r['factor']:>8.3f}{r['cov_cal']:>8.4f}{r['width']:>8.4f}")

if len(ARMS) == 2:
    rng = np.random.default_rng(0)
    A, B = ARMS[0][0], ARMS[1][0]
    print("\nisolated effect of the TV term (shipped minus control), paired:")
    for k in ("rmse", "roughness", "r_unc", "crps_cal"):
        d = per_case[A][k] - per_case[B][k]
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(6000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"   {k:<11}{d.mean():+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  "
              f"{'SIGNIFICANT' if lo * hi > 0 else 'TIED'}")

torch.save({"rows": rows, "per_case": per_case,
            "meta": {"seed": SEED, "n_cases": n, "epochs": 6,
                     "benchmark_md5": "44b866540296490c615be13bacb4242e"}},
           ROOT / "benchmark/results_magnitude_tv.pt")
print("\nsaved benchmark/results_magnitude_tv.pt")
