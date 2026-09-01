# Two experiments: discarding stale readings, and a structural loss term

Both run on `benchmark/ocean_bench_v1.npz` (40 frozen cases, 2 h Dubins track,
time-varying observations, 3749 scored cells, seed 20260830). Raw predictions,
per-case metrics and the fitted factors are in `benchmark/results_age_calibration.pt`
and `benchmark/results_vorticity.pt`; both stamp the benchmark's MD5, so every
number here can be re-derived without re-running a model.

---

## 1. Discard observations older than one hour

**Result: −8.2% RMSE and 34% narrower intervals, and it only works with priors.**

| cutoff | readings | RMSE | vs full 2 h track | conformal factor | interval width |
|---|---|---|---|---|---|
| 0.5 h | 50 | 0.0639 | −5.1% (tied) | 1.7504 | 0.1573 |
| 0.75 h | 75 | 0.0631 | −6.2% (tied) | 1.9571 | 0.1625 |
| **1.0 h** | 100 | **0.0618** | **−8.2%, CI [−0.0102, −0.0010]** | **2.2006** | **0.1742** |
| 1.25 h | 125 | 0.0628 | −6.7%, significant | 2.4834 | 0.1852 |
| 1.5 h | 150 | 0.0636 | −5.6%, significant | 2.8729 | 0.2074 |
| full 2 h | 200 | 0.0673 | — | 3.9142 | 0.2643 |

One hour is a genuine interior optimum, not "fewer readings is always better" — the
curve turns back up on both sides, and angle error minimises at the same cutoff
(0.6842 rad). Accuracy and sharpness improve together, which is unusual enough to
be worth stating: the discard both lowers the error and lets the intervals close by
a third at matched 90% coverage.

**Only models carrying an independent temporal prior benefit.** Measured earlier on
the same benchmark, at a 1 h cutoff:

| model | effect of discarding |
|---|---|
| corrdiff **with** priors | −9.9%, significant |
| corrdiff **without** priors | +2.5%, tied |
| distattn | **+15.5% worse**, significant |
| gp | **+10.9% worse**, significant |

Same weights, same architecture for the two CorrDiff rows — the only difference is
whether the priors were supplied. A model with no other source of information needs
the stale reading: it is bad information, but it beats none. A model with the 13/25 h
history can afford to throw it away, and gains by doing so.

**The confound, and the control for it.** Filtering by age also shrinks the spatial
extent of the observations, and extent matters enormously (spreading the same 90
readings over the domain instead of along a track moved GP by 76%). Every cutoff was
therefore paired with a matched-count control: the same number of readings sampled
evenly across the whole 2 h track. Those controls moved RMSE by ≤0.0016, so the gain
is freshness and not a coverage artefact.

---

## 2. The conformal factor was fitted for the wrong task

**This was a live correctness bug, not an improvement.**

The shipped `CORRDIFF_SIGMA_SCALE = 1.6787` was fitted for *simultaneous*
observations. Everything we intend to report is 2 h time-varying collection. Refit by
split conformal (20 cases to fit, 20 held out, level 0.90):

| observation process | factor | coverage (held out) |
|---|---|---|
| simultaneous (shipped) | 1.6787 | — |
| 2 h collection, 1 h cutoff | **2.2006** | 0.9200 |
| 2 h collection, full track | **3.9142** | 0.9232 |

Any interval published on the full 2 h task using the shipped factor would have been
**2.3x too narrow**.

Convenient coincidence: at the 1 h cutoff the required factor is 2.2006, essentially
the `CORRDIFF_SIGMA_SCALE_TIMED = 2.1801` already in the library. So adopting the
cutoff means the existing timed factor is already correct; only the uncut 2 h track
needs the much larger 3.9.

---

## 3. A vorticity loss term does nothing

**Result: null, on every metric including the one it directly optimises.**

CorrDiff's loss has always been a single v-prediction MSE — verified across all git
history, no CorrDiff trainer ever carried a structural term. RePaint and DistAttn both
have a curl+div term; Stream is divergence-free by construction. That made "add a
structural term to CorrDiff" the obvious untested lever, and Seth's new eddy metric
made it timely.

**Vorticity only, deliberately not curl+div.** On 60 held-out frames this field's RMS
divergence is **48% of its RMS vorticity** — it is emphatically not divergence-free, so
a divergence penalty would push the model away from real dynamics. (See §4: that is
also why Stream is the weakest model.) Vorticity is what an eddy is, and what the eddy
metrics measure.

Fine-tuned from the shipped checkpoint, 3000 gradient steps, λ = 1.0 (the term
contributes 22% of the total loss). The control arm is identical in every respect with
the term switched off, so the comparison isolates the term from the effect of simply
training longer. Both arms saw byte-identical batches in identical order — verified
empirically: two seeded runs produce bit-identical losses.

| checkpoint | RMSE | angle_rms | vorticity RMSE | eddy recall |
|---|---|---|---|---|
| baseline (shipped) | 0.0673 | 0.7237 | 0.01273 | 0.4426 |
| control, λ = 0 | 0.0674 | 0.7195 | 0.01279 | 0.4450 |
| vorticity, λ = 1 | 0.0669 | 0.7190 | 0.01282 | 0.4576 |

The isolated comparison, paired over 40 cases:

```
lamV - lam0   rmse_vector     -0.00050  CI [-0.00128, +0.00022]  TIED
              angle_rms       -0.00051  CI [-0.00657, +0.00545]  TIED
              vorticity_rmse  +0.00003  CI [-0.00002, +0.00009]  TIED
              eddy_hit_rate   +0.01259  CI [-0.00037, +0.02665]  TIED
```

**The control is what makes this call.** Against the *baseline*, the λ=1 arm shows a
significant eddy-recall gain (+0.0150, CI [+0.0007, +0.0302]). Without the λ=0 arm
that would have been reported as "the vorticity term improves eddy recall." Against
the control it is +0.0126 with the interval grazing zero — suggestive, not established.

Two details sharpen the null:

- **It optimised the target and did not transfer.** Training vorticity loss fell from
  0.00399 (control) to 0.00362 with the term, about 9% better. Test-set vorticity RMSE
  moved +0.00003 — nothing.
- **Vorticity RMSE got significantly worse in both arms versus baseline** (+0.00006
  control, +0.00009 term). Fine-tuning at all degraded it slightly, and the term did
  not prevent that.

---

## 4. Why the divergence-free architecture cannot win

Stream trains on `data_divfree_chrono.pickle`, whose targets are Helmholtz-projected
to remove divergence, and is evaluated against the real field.

Helmholtz projection is an *orthogonal* projection, so `divfree(truth)` is the closest
divergence-free field to the truth. Stream's output is exactly divergence-free by
construction. Therefore every field Stream can emit is at least
`‖truth − divfree(truth)‖` from the truth — measured at **0.0613 m/s** on 60 held-out
frames.

| | RMSE |
|---|---|
| **Stream's floor** | **0.0613** — best achievable, ever |
| corrdiff (actual) | 0.0665 |
| stream (actual) | 0.1039 |

A *perfect* Stream would beat CorrDiff by 8%. The constraint spends nearly the whole
error budget before the model does anything. This is a provable ceiling, not a tuning
result, and 59% of Stream's measured error is the train/eval mismatch rather than
model quality. Report Stream with this stated, or not at all.

---

## What is now closed

- Structural loss terms on CorrDiff — tested, null, properly controlled.
- The divergence-free architecture — provably capped, and the cap is nearly at
  CorrDiff's current accuracy.
- The age-cutoff optimum — located at 1 h, previously only bracketed.
- The conformal factor for realistic collection — measured, and the shipped one was
  wrong for this task by 2.3x.

## Reproducing

```bash
bash benchmark/run_all.sh          # both experiments, sequential, ~3.5 h on one GPU
```

Deterministic given the frozen benchmark and seed 20260830. The fine-tune arms
additionally require `--deterministic_data`, without which the two arms see different
batches and the comparison is meaningless.
