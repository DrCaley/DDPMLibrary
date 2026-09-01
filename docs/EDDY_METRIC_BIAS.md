# The Okubo-Weiss eddy metric is biased by divergence

**A model is penalised for representing divergence correctly, and a
divergence-free model is credited for structure it does not have. Correcting for
it produced one false positive and one false negative in a four-way comparison.**

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

## The correction

Run the same detector after Helmholtz-projecting **both** the prediction and the
truth, so neither side carries divergence and the strain bias cancels. Reported as
`eddy_rot` alongside the raw `eddy`.

## What it changes

Final comparison, 40 cases on `ocean_bench_v1`, each model at its own optimum:

| model | RMSE | angle_rms | eddy (raw) | eddy_rot |
|---|---|---|---|---|
| **corrdiff (1 h cutoff)** | **0.0618** | **0.6842** | 0.4308 | 0.4397 |
| corrdiff (full track) | 0.0673 | 0.7237 | 0.4426 | 0.4454 |
| distattn | 0.0738 | 0.7875 | 0.3579 | 0.3826 |
| stream (+divergent) | 0.0908 | 0.9068 | 0.3752 | **0.4559** |
| stream (divfree only) | 0.1038 | 1.0195 | 0.4071 | **0.4594** |
| vcnn | 0.0791 | 0.8220 | 0.3761 | 0.4196 |
| gp | 0.1254 | 1.0908 | 0.1430 | 0.1451 |

Three comparisons change verdict:

| comparison | raw eddy | corrected |
|---|---|---|
| corrdiff vs stream | **+0.0556, significant** | −0.0163, **TIED** |
| stream +div vs divfree | **−0.0319, significant** | −0.0035, **TIED** |
| distattn vs stream | −0.0173, TIED | **−0.0733, significant** |

One false positive, one false negative, one hidden difference — in four
comparisons. Every eddy claim needs the correction before it is reported.

## The substantive consequence

**Stream has the best rotational structure in the suite** (0.4559 / 0.4594 on
`eddy_rot`, tied with CorrDiff and ahead of everything else) while being the worst
on RMSE. That is a clean perception-distortion result: the stream-function
architecture genuinely does capture eddies best, and pays for it in pointwise
accuracy.

The raw metric was hiding that *and* mis-crediting it at the same time — it
credited the divergence-free variant for its artificially low strain, then
penalised the corrected variant for fixing the physics.

## Related: Stream must be run with `full_field=True`

`StreamDDPM.predict` defaults to `full_field=False`, whose output is exactly
divergence-free. On a field whose divergent component carries 14% of the energy
that costs **12.5% RMSE and 11% angle**, both significant:

```
stream, divfree only   RMSE 0.1038   angle 1.0195
stream, +divergent     RMSE 0.0908   angle 0.9068
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
