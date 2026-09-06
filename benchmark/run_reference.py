"""Run this library's models on ocean_bench_v1 and emit the reference table."""
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
from ddpm_library import CorrDiff, DistAttn, StreamDDPM, GP, VCNN
from ddpm_library import config as C

bench = np.load(str(ROOT / "benchmark/ocean_bench_v1.npz"))
obs_all, priors_all = bench["observations"], bench["priors"]
cd, da, st, gp, vc = (CorrDiff(device=DEV), DistAttn(device=DEV),
                      StreamDDPM(device=DEV), GP(), VCNN(device=DEV))
MODELS = {"corrdiff":         (cd, True,  {"n_draws": 20}),
          "corrdiff_noprior": (cd, False, {"n_draws": 20}),
          "distattn":         (da, False, {"n_draws": C.DISTATTN_DEFAULT_N_DRAWS}),
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
np.savez_compressed(str(ROOT / "benchmark/reference_predictions.npz"), **preds)
print("\n" + score.report(preds, bench))
json.dump({n: {m: list(map(float, s)) for m, s in score.score_model(p, bench).items()}
           for n, p in preds.items()},
          open(str(ROOT / "benchmark/reference_scores.json"), "w"), indent=2)
print("\nwrote reference_predictions.npz and reference_scores.json")
