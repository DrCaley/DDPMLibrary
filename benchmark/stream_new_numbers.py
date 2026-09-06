"""The replacement Stream numbers, if the spread term is dropped.

Scores Models/StreamFn_Cond_x0_mag.pt -- the shipped model's own no-spread
predecessor -- in exactly the fields the paper tables report, on both benchmarks,
alongside the currently-shipped checkpoint for direct substitution.

Conformal protocol matches uncertainty_final.py: fit on the first half, verify on
the second. The v1-fitted factor is additionally applied blind to all of v1b.
"""
import sys, warnings
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from _paths import DEV, MODELS_DIR  # noqa: E402
from ddpm_library import StreamDDPM, metrics

z, LEVEL, DRAWS = 1.6448536269514722, 0.90, 20
DS = MODELS_DIR
ARMS = [("predecessor (NO spread)", DS / "StreamFn_Cond_x0_mag.pt"),
        # the PREVIOUS (spread-term) weights, kept for the comparison; the shipped
        # asset is now the predecessor, so pointing at it would compare a model
        # with itself.
        ("previous (spread on)",
         MODELS_DIR / "StreamFn_Cond_x0_mag_spread.pt")]
BENCH = [("v1", "ocean_bench_v1.npz", 20260830), ("v1b", "ocean_bench_v1b.npz", 20260901)]

out = {}
for bname, bfile, seed in BENCH:
    b = np.load(ROOT / "benchmark" / bfile)
    obs, pri, truth = b["observations"], b["priors"], b["truth"]
    ocean = np.asarray(b["ocean_mask"], bool)
    n, half = len(truth), len(truth) // 2
    for aname, p in ARMS:
        st = StreamDDPM(device=DEV, dir_weights_path=p)
        M, S = [], []
        for i in range(n):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m, s = st.predict([tuple(x) for x in obs[i]], pri[i],
                                  n_draws=DRAWS, seed=seed + i, full_field=True, calibrate=False)
            M.append(np.asarray(m, np.float32)); S.append(np.asarray(s, np.float32))
        m, s = np.stack(M), np.stack(S); del st
        rmse = np.array([np.sqrt(((m[i] - truth[i])[ocean] ** 2).sum(-1).mean())
                         for i in range(n)])
        ang = []
        for i in range(n):
            pv, tv = m[i][ocean], truth[i][ocean]
            c = (pv * tv).sum(-1) / (np.linalg.norm(pv, axis=-1)
                                     * np.linalg.norm(tv, axis=-1) + 1e-12)
            a = np.arccos(np.clip(c, -1, 1))
            ang.append(np.sqrt((a ** 2).mean()))
        crps_raw = np.array([metrics.crps_gaussian(m[i], s[i], truth[i], ocean_mask=ocean)
                             for i in range(n)])
        cov_raw = np.array([metrics.coverage(m[i], s[i], truth[i], level=LEVEL,
                                             ocean_mask=ocean) for i in range(n)])
        err, sg = np.abs(m - truth)[:, ocean], s[:, ocean]
        ok = sg[:half] > 1e-9
        f = float(np.quantile(err[:half][ok] / (z * sg[:half][ok]), LEVEL))
        cov_cal = float(np.mean(err[half:] <= z * f * sg[half:]))
        width = float(np.mean(2 * z * f * sg[half:]))
        crps_cal = np.array([metrics.crps_gaussian(m[i], f * s[i], truth[i],
                                                   ocean_mask=ocean) for i in range(n)])
        out[(bname, aname)] = dict(
            rmse=rmse.mean(), angle=float(np.mean(ang)), crps_raw=crps_raw.mean(),
            crps_cal=crps_cal.mean(), cov_raw=cov_raw.mean(), factor=f,
            cov_cal=cov_cal, width=width,
            per_case=dict(rmse=rmse, angle=np.array(ang), crps_cal=crps_cal))
        print(f"  scored {bname} / {aname}", flush=True)

print(f"\n{'bench':<5}{'arm':<26}{'RMSE':>8}{'angleRMS':>10}{'CRPSraw':>9}"
      f"{'CRPScal':>9}{'covraw':>8}{'factor':>8}{'covcal':>8}{'width':>8}")
for (bn, an), r in out.items():
    print(f"{bn:<5}{an:<26}{r['rmse']:>8.5f}{r['angle']:>10.5f}{r['crps_raw']:>9.4f}"
          f"{r['crps_cal']:>9.4f}{r['cov_raw']:>8.4f}{r['factor']:>8.3f}"
          f"{r['cov_cal']:>8.4f}{r['width']:>8.4f}")

rng = np.random.default_rng(0)
print("\npredecessor - previous, paired:")
for bn, _, _ in BENCH:
    for k in ("rmse", "angle", "crps_cal"):
        d = (out[(bn, "predecessor (NO spread)")]["per_case"][k]
             - out[(bn, "previous (spread on)")]["per_case"][k])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(6000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"   {bn:<4}{k:<10}{d.mean():+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  "
              f"{'SIGNIFICANT' if lo * hi > 0 else 'TIED'}")

torch.save({"rows": {f"{a}|{b}": {k: v for k, v in r.items() if k != 'per_case'}
                     for (a, b), r in out.items()},
            "per_case": {f"{a}|{b}": r["per_case"] for (a, b), r in out.items()},
            "meta": {"n_draws": DRAWS, "level": LEVEL,
                     "v1_md5": "44b866540296490c615be13bacb4242e",
                     "v1b_md5": "08f1a69fbbc6186b4bda90dfe2e79280"}},
           ROOT / "benchmark/results_stream_new_numbers.pt")
print("\nsaved benchmark/results_stream_new_numbers.pt")
