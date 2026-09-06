# The Okubo-Weiss eddy metric is biased by divergence

> **Being regenerated (2026-09-05).** Every DistAttn and RePaint number below
> predates the fix for their per-draw seed aliasing, and every Stream number
> predates the `helmholtz_project` symbol fix. CorrDiff, VCNN and GP are
> unaffected. Do not quote the affected rows until this note is gone.

**The bias is proven. The correction is not reliable. Read both halves.**

A model is penalised for representing divergence correctly, and a divergence-free
model is credited for structure it does not have. That much is mathematical and
verified. The obvious fix -- project the divergence out of both fields first --
depends on a boundary condition this domain does not determine, and two reasonable
choices produce OPPOSITE model rankings. Do not report `eddy_rot` as a result.

Applies to our `metrics.eddy_hit_rate` and to any Okubo-Weiss eddy metric,
including the eddy-IoU in the collaborator's mission plots.

## The mechanism

Okubo-Weiss classifies a cell as rotation-dominated when

```
W = strain² − vorticity² < 0
```

Adding a curl-free component to a field **cannot change its vorticity** — the curl
of a gradient is identically zero. It *does* add strain. So a model that correctly
reproduces the divergent part of the flow raises `W`, pushes cells out of the
rotation-dominated class, and loses eddies it never got wrong.

Measured directly, by restoring Stream's divergent component (40 benchmark cases):

| | divergence-free only | + divergent part | change |
|---|---|---|---|
| vorticity RMS | 0.026836 | 0.026695 | **−0.5%** (numerical) |
| strain RMS | 0.026303 | 0.026743 | +1.7% |
| cells with W < 0 | 33.9% | 32.5% | — |
| **eddy recall** | 0.4071 | 0.3752 | **−7.8%, significant** |

The rotational structure is *identical*. The entire recall drop is reclassification.

## The correction, and why it does not work here

Run the same detector after Helmholtz-projecting **both** the prediction and the
truth, so neither side carries divergence and the strain bias cancels. Reported as
`eddy_rot` alongside the raw `eddy`.

**This does not survive contact with the coastline.** The Helmholtz decomposition
on a bounded domain is not unique without boundary conditions, and ~30% of this
grid is land. Two defensible choices disagree completely, on identical predictions:

| projection | boundary condition | ranking produced |
|---|---|---|
| FFT (library `helmholtz_project`) | periodic | stream > distattn > corrdiff > vcnn > gp |
| sparse Poisson (`Utils/poisson_projection.py`) | Neumann, no flow through land | corrdiff > vcnn > distattn > stream > gp |

Stream is **best** under one and **fourth** under the other.

The FFT projection also fails its own sanity checks: it removes only 65% of the
divergence, perturbs vorticity by 7% (correlation 0.939 with the original), and
gets *worse* with more iterations -- it alternates a periodic FFT projection with
zeroing land, and on an irregular coastline those two operations fight until they
settle on something that is not a projection at all.

The Poisson solver is the more physically defensible choice (no flow through a
coastline is the right boundary condition) but is singular on our mask out of the
box: the ocean has **two connected components, of 3748 cells and 1 cell**, and that
lone isolated cell leaves a null direction. It runs once that cell is excluded.

## What to do instead

- **Compare models of similar divergence on the raw metric.** The bias only
  distorts comparisons where divergence differs.
- **Report vorticity RMSE.** It needs no decomposition and no boundary condition,
  so it carries none of this ambiguity.
- **Do not report `eddy_rot` as a result** until the boundary condition is settled
  and the projector validated on this mask.

## What it changes

Final comparison, 40 cases on `ocean_bench_v1`, each model at its own optimum.
**The `stream` rows are the pre-2026-09-02 direction weights** (spread term on); the
eddy columns have not been re-measured since the swap. Current Stream RMSE/angle are
0.0865 / 0.8663 — see `PAPER_NUMBERS.md`.

| model | RMSE | angle_rms | eddy (raw) | eddy_rot |
|---|---|---|---|---|
| **corrdiff (1 h cutoff)** | **0.0618** | **0.6842** | 0.4308 | 0.4397 |
| corrdiff (full track) | 0.0673 | 0.7237 | 0.4426 | 0.4454 |
| distattn | 0.0738 | 0.7875 | 0.3579 | 0.3826 |
| stream (+divergent)† | 0.0908 | 0.9068 | 0.3752 | **0.4559** |
| stream (divfree only)† | 0.1038 | 1.0195 | 0.4071 | **0.4594** |
| vcnn | 0.0791 | 0.8220 | 0.3761 | 0.4196 |
| gp | 0.1254 | 1.0908 | 0.1430 | 0.1451 |

The `eddy_rot` column above is **retained for the record only** -- it is the FFT
projection, and the Poisson projection reorders it. Three comparisons change
verdict under the FFT correction:

| comparison | raw eddy | corrected |
|---|---|---|
| corrdiff vs stream | **+0.0556, significant** | −0.0163, **TIED** |
| stream +div vs divfree | **−0.0319, significant** | −0.0035, **TIED** |
| distattn vs stream | −0.0173, TIED | **−0.0733, significant** |

One false positive, one false negative, one hidden difference -- in four
comparisons. That the raw metric is unreliable here is solid; that these
particular corrected verdicts are right is not, since they move under a different
boundary condition.

## What remains true

The **bias itself** is not in doubt. Adding a pure gradient field to a prediction
cannot change its vorticity, yet it drives the raw metric from 0.458 to 0.042 on
real benchmark fields (amplitude 0.02) and to exactly 0 at amplitude 0.05. That is
pinned by `tests/test_metrics.py`.

So any eddy comparison between models whose divergence differs -- which includes
every comparison involving Stream in either configuration -- is confounded on the
raw metric. That conclusion stands independently of whether the correction works.

What is **withdrawn** is the ranking claim built on `eddy_rot`: that Stream has the
best rotational structure in the suite. It is best under the FFT projection and
fourth under the Poisson projection, and nothing here decides between them.

## Related: Stream must be run with `full_field=True`

`StreamDDPM.predict` defaults to `full_field=False`, whose output is exactly
divergence-free. On a field whose divergent component carries 14% of the energy
that costs **12.5% RMSE and 11% angle**, both significant:

† pre-2026-09-02 direction weights (spread term on).

```
stream, divfree only   RMSE 0.1038   angle 1.0195      (pre-2026-09-02 weights)
stream, +divergent     RMSE 0.0908   angle 0.9068      (pre-2026-09-02 weights)
                       -0.01294 CI [-0.01775, -0.00860]  significant
                       -0.11273 CI [-0.16120, -0.06284]  significant
```

Every Stream number published before this — ours and the collaborator's — used the
constrained default and understates it.

Note the current mechanism is inelegant: the divergent component is borrowed from
a VCNN prediction rather than predicted by the model. The principled version is a
complete Helmholtz-Hodge decomposition, `v = ∇×ψ + ∇φ`, predicting both scalar
potentials from one network — which would also collapse the pipeline from two
networks and four stages to one of each. Not built; see
`OBSERVATION_AGE_AND_STRUCTURE.md` §4 for why it is unlikely to overtake CorrDiff
even so.

## Reproducing

```bash
python benchmark/final_comparison.py     # the table, both eddy metrics
python benchmark/stream_fullfield.py     # the divergence-free cost, in isolation
```

Results in `benchmark/results_final_comparison.pt` and
`benchmark/results_stream_fullfield.pt`, both stamped with the benchmark MD5.
