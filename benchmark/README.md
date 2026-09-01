# ocean_bench_v1 — a frozen benchmark for current-field reconstruction

Everyone in this group is currently scoring a different task. Different vehicle
tracks, different observation timing, different cell masks, different RMSE and
angle conventions. Two independent reimplementations of "the same" experiment
agreed on model *ordering* and disagreed on absolute numbers by 15–50%, which
makes cross-checking impossible.

This file removes every one of those choices. Run your model on these inputs,
score with `score.py`, and any remaining disagreement is genuinely in the model
rather than in the harness.

## Using it

```python
import numpy as np, score

bench = np.load("ocean_bench_v1.npz")
preds = []
for i, obs in enumerate(bench["observations"]):
    priors = list(bench["priors"][i])              # only if your model takes them
    mean, _ = my_model.predict([tuple(r) for r in obs], priors)
    preds.append(mean)                              # (44, 94, 2) in m/s

print(score.report({"my_model": np.stack(preds)}, bench))
```

Or save `predictions` of shape `(40, 44, 94, 2)` to an npz and run
`python score.py my_predictions.npz`.

## What is in the file

| key | shape | meaning |
|---|---|---|
| `observations` | (40, 200, 5) | `(lat, lon, unix_t, u, v)`, u/v in m/s |
| `truth` | (40, 44, 94, 2) | the field to predict, at the END of the run |
| `priors` | (40, 2, 44, 94, 2) | the field 13 h and 25 h before the target |
| `ocean_mask` | (44, 94) bool | the 3749 cells that are scored |
| `lats`, `lons` | (44,), (94,) | grid coordinates |
| `frame_index` | (40,) | source frame, for traceability |

Also stored: `seed`, `n_readings`, `collection_span_hours`, `reading_drift_ms`,
`vehicle_speed_ms`, `turn_radius_metres`, `cell_metres`, `prior_lags_hours`.

**Pass the priors if your model takes them.** CorrDiff and StreamDDPM are
*conditioned* on them and silently zero them when omitted — that alone costs
CorrDiff 20% and is not a fair comparison against models that never use them.

## The observation process

A Dubins-style vehicle (constant speed 1.06 m/s, minimum turning radius ~102 m,
random walk in the steering command) drives for **2 hours** from a random start.
Each reading is the field **at the moment the vehicle reached that cell** — not
the target frame's value. Mean drift between a reading and the target frame's
value at the same cell is 0.031 m/s, so the staleness is real and not a rounding
effect.

This matters more than anything else in the file. Scoring against instantaneous
observations measures a task no vehicle can perform, and it flatters models
trained that way: the same models that tie here separate by 20%+ under the
idealised version.

Observation *density* is deliberately not a variable — we swept 25, 200 and 1527
readings along identical trajectories and the results moved by under 1%, because
a 2 h track only ever touches ~160 of 3749 cells and extra readings inside that
patch add nothing.

## Metric conventions

`score.py` is the definition; the prose here is only a summary. It reports both
of each ambiguous pair, because both ambiguities have already caused a
disagreement in this group:

- `rmse_vector` = `sqrt(mean_cells(du² + dv²))` — vector magnitude
- `rmse_component` = `sqrt(mean over cells AND components)` = `rmse_vector / √2`
- `angle_mean_rad` — mean per-cell angle error
- `angle_rms_rad` — RMS of the per-cell angle error

The two RMSE conventions differ by a factor of 1.414. The two angle statistics
**rank models differently**. Always say which you mean.

Cells where either vector is near zero are excluded from the angle only — their
direction is undefined. Confidence intervals resample *cases*, never cells: cells
within a case are strongly correlated and treating them as independent gives
indefensibly narrow intervals.

## Reference numbers (this library, v0.7.0)

40 cases, 3749 cells. `rmse_vector`, mean / median:

| model | priors | RMSE | angle_rms (rad) |
|---|---|---|---|
| **corrdiff** | yes | **0.0665** / 0.0611 | **0.725** |
| distattn | no | 0.0735 / 0.0655 | 0.790 |
| vcnn | no | 0.0791 / 0.0743 | 0.822 |
| corrdiff | **no** | 0.0797 / 0.0692 | 0.817 |
| stream | yes | 0.1039 / 0.0964 | 1.017 |
| gp | no | 0.1254 / 0.1111 | 1.091 |

What is and is not separated, by paired bootstrap over cases:

- **corrdiff vs distattn: TIED** (−0.0070, CI [−0.0152, +0.0010]). Do not claim
  either is more accurate on this task.
- **distattn vs vcnn: TIED** (−0.0056, CI [−0.0134, +0.0022]).
- **corrdiff-without-priors vs vcnn: TIED** (+0.0006, CI [−0.0026, +0.0041]).
- The priors are worth −0.0132 to CorrDiff (CI [−0.0182, −0.0089]), significant,
  and are the only thing that lifts anything clear of the pack.
- stream and gp are significantly behind everything else.

So on the realistic task, **a plain CNN matches both diffusion models unless
CorrDiff is given its temporal priors**. That is a much narrower claim than the
simultaneous-observation benchmark supports, and it is the honest one.

`reference_predictions.npz` holds the actual fields, so anyone can re-score them
under a different convention without re-running the models.

## Regenerating

`python make_benchmark.py` — deterministic given `SEED = 20260829`.
