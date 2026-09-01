"""Run this library's models on ocean_bench_v1 and emit the reference table."""
import sys, warnings, json
sys.path.insert(0, "/workspace/DDPMLibrary/src")
sys.path.insert(0, "/workspace/DDPMLibrary/benchmark")
import numpy as np
import score
from ddpm_library import CorrDiff, DistAttn, StreamDDPM, GP, VCNN

bench = np.load("/workspace/DDPMLibrary/benchmark/ocean_bench_v1.npz")
obs_all, priors_all = bench["observations"], bench["priors"]
cd, da, st, gp, vc = (CorrDiff(device="cuda"), DistAttn(device="cuda"),
                      StreamDDPM(device="cuda"), GP(), VCNN(device="cuda"))
MODELS = {"corrdiff":         (cd, True,  {"n_draws": 20}),
          "corrdiff_noprior": (cd, False, {"n_draws": 20}),
          "distattn":         (da, False, {"n_draws": 10}),
          "stream":           (st, True,  {"n_draws": 20}),
          "gp":               (gp, False, {}),
          "vcnn":             (vc, False, {})}

preds = {n: [] for n in MODELS}
for i in range(len(obs_all)):
    obs = [tuple(r) for r in obs_all[i]]
    priors = [priors_all[i][k] for k in range(priors_all.shape[1])]
    for n, (mdl, pri, kw) in MODELS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            call = dict(kw)
            if n not in ("gp", "vcnn"):
                call["seed"] = 5000 + i
            mean, _ = mdl.predict(*((obs, priors) if pri else (obs,)), **call)
        preds[n].append(np.asarray(mean, np.float32))
    print(f"  case {i + 1}/{len(obs_all)}", flush=True)

preds = {n: np.stack(v) for n, v in preds.items()}
np.savez_compressed("/workspace/DDPMLibrary/benchmark/reference_predictions.npz", **preds)
print("\n" + score.report(preds, bench))
json.dump({n: {m: list(map(float, s)) for m, s in score.score_model(p, bench).items()}
           for n, p in preds.items()},
          open("/workspace/DDPMLibrary/benchmark/reference_scores.json", "w"), indent=2)
print("\nwrote reference_predictions.npz and reference_scores.json")
