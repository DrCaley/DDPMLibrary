# RePaint — archived research code

The collaborator's original training and evaluation scripts for the two RePaint
models. **Nothing here is imported by the library** and none of it ships in the
installed package (`pyproject.toml` packages `src/` only) — it is kept for
provenance and retraining.

To *run* either model, use the library instead:

```python
from ddpm_library import RePaint, RePaintUncond

mean, unc = RePaint(device="auto").predict(observations, priors, n_draws=10)
mean, unc = RePaintUncond(device="auto").predict(observations, n_draws=10)
```

## The two models

| | `RePaint` | `RePaintUncond` |
|---|---|---|
| Conditioning | ocean state 13 h + 25 h earlier (4 channels) | none |
| `cond_ch` | 4 | 0 |
| Checkpoint | `repaint_timecond_weights.pt` | `repaint_uncond_weights.pt` |
| Training pickle | `data_chrono_raw.pickle` | `data.pickle` |
| Published results | DPS RMSE 0.0410, MCG 0.0414 | none |
| Docs | `README_timecond.md` | `README_uncond.md` |

Both share one architecture: `cond_ch=0` reproduces the unconditional network
exactly, which is why a single vendored copy of the model code serves both.

> The two were trained on **different pickles**, so their validation losses are
> not comparable and their train/test splits may not agree. Settle any
> accuracy comparison with `scripts/compare_models.py` on common frames, not
> with the numbers in the two READMEs.

## Contents

| File | Was | Role |
|---|---|---|
| `train_timecond.py` | `Linear Best Model - Time Conditioned/train_TimeConditioned.py` | training entry point, time-conditioned |
| `train_uncond.py` | `Linear Best Model - Not Time Conditioned/train.py` | training entry point, unconditional |
| `chrono_dataset.py` | (time-conditioned folder) | chronological loader that supplies the 13 h / 25 h priors |
| `dataset.py` | (unconditional folder) | plain loader, no history |
| `run_mcg_dps_z004.py` | (unconditional folder) | published DPS/MCG evaluation script, `z=0.04` |
| `stride_inference_overview.txt` | both folders (identical) | notes on chain striding |
| `results/` | (time-conditioned folder) | published 100-seed uncertainty validation |
| `checkpoints_timecond_README.md` | (time-conditioned folder) | checkpoint provenance |

## What was removed

`repaint_model.py`, `diffusion.py`, `loss_functions.py` and `repaint_infer.py`
appeared in **both** original folders. They are now vendored once at
`src/ddpm_library/repaint/` (the time-conditioned versions, which are strict
supersets — the only edit is `from loss_functions` → `from .loss_functions`).
`repaint_infer.py` became `sampler.py` there, with `cond` threaded through.

The 171 MB `model/best_model_linear.pt` was byte-identical to the bundled
`repaint_uncond_weights.pt` asset and was deleted as a duplicate.

All of these remain in git history if you need the originals.
