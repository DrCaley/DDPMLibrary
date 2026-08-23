# Non-simultaneous observations: what we measured

Results were spread across five scripts, six JSON files and a chat log; those scripts are now `scripts/staleness.py`. This is
the single record. Every number here is from a paired, leakage-free run on
frames held out of training by **both** dataset splits in use across the group's
checkpoints (`scripts/fair_eval_frames.json`, 2460 frames).

## The problem

Every model in this library except `DistAttn` is trained and scored as if a
transect were collected instantaneously. It isn't. At the collaborator's measured
vehicle rate (0.628 cells / 30 s) a 90-cell track takes over an hour, and the
field moves further than any model's total error in that time:

| lag | RMS field change | vs CorrDiff's 0.047 RMSE |
|---|---|---|
| 1 h | 0.0594 | 1.26× |
| 2 h | 0.1000 | 2.13× |
| 3 h | 0.1323 | 2.82× |
| 6 h | 0.1830 | 3.89× (decorrelated) |
| 12 h | 0.1306 | tidal recorrelation (M2) |

So the standard benchmark scores a task that cannot physically occur.

## What it costs

CorrDiff, shipped configuration, 39 frames, identical tracks:

| metric | simultaneous | time-varying | change |
|---|---|---|---|
| CRPS | 0.0250 | 0.0276 | **+10.4%** |
| RMSE | 0.0464 | 0.0504 | +8.6% |
| RMSE @observed | 0.0060 | 0.0205 | +242% |
| RMSE @unobserved | 0.0470 | 0.0509 | +8.3% |
| **coverage @90%** | **0.887** | **0.824** | **−6.3 pts** |

Replicated at +10.7%, +9.9%, +11.2% across independent runs. Paired bootstrap:
**+0.00247 CRPS, 95% CI [+0.00104, +0.00394]**. The coverage drop is also
significant (−0.063, CI [−0.085, −0.043]) and 0.90 falls outside the
time-varying CI.

**It is not architecture-specific.** Four unrelated methods lose the same amount:

| model | CRPS degradation | 95% CI | vs CorrDiff |
|---|---|---|---|
| corrdiff (trained conditioning) | +0.00260 | [0.00110, 0.00431] | — |
| repaint (guided sampling + priors) | +0.00253 | [0.00074, 0.00434] | indistinguishable |
| repaint_uncond (guided, no priors) | +0.00366 | [0.00081, 0.00703] | indistinguishable |
| vcnn (plain CNN) | +0.00330 | [0.00140, 0.00541] | — |

## Four attempted fixes, all null

| fix | result | why it failed |
|---|---|---|
| **Age conditioning** (DistAttn's mechanism) | +0.00018, CI [−0.0017, +0.0020] | No effect. Hiding the ages entirely costs less than the 0.001 noise floor set by a negative control. DistAttn's apparent robustness comes from fitting observations loosely (0.0201 vs CorrDiff's 0.0060), not from knowing their ages. |
| **Sensor-noise dial** | no setting significant | Staleness needs σ≈0.38; the dial's ceiling is 0.10. More fundamentally the dial models **iid** noise while staleness error is spatially coherent (0.60 correlated at 12 cells), so it is the wrong error *structure*, not just the wrong magnitude. |
| **Fine-tuning on the real observation process** (30 epochs, 7.4 GPU-h) | +10.8% vs +11.2% | Nothing. Loss moved 0.0126 → 0.0124; the v-prediction MSE is dominated by noise-matching and barely registers observation-trust behaviour. |
| **Correcting observations before inference** | +0.00011, ns | The corrector overfits: 45% better in-sample, **4.2% worse** out-of-sample. It gains +6.8% at a 2 h span but the operationally relevant span is 1.2 h, where there is too little correctable signal. |

## Why the accuracy penalty is not recoverable cheaply

The anchor diagnostic isolates it. CorrDiff's V-CNN anchor under both conditions:

| | RMSE | @obs | **@unobs** |
|---|---|---|---|
| simultaneous | 0.0724 | 0.0097 | 0.0732 |
| time-varying | 0.0715 | 0.0249 | **0.0722** |

**Far-field accuracy is untouched.** Only agreement at the observed cells
degrades, and that is correct behaviour: a stale reading genuinely disagrees with
the target frame. Stale observations are not corrupted noise, they are a coherent
ocean state displaced in time. Water travels ~13 cells per hour against a field
correlation length of ~20 cells, so the displacement is comparable to the
correlation scale — local gradients cannot express it.

**Not proven impossible.** A dense forward-projection upper bound (full fields,
1.38 M rows) recovers 6.0% at 1 h and 18.1% at 2 h, so the information is not
absent. But the expected field-level payoff is ~1% RMSE (0.0003 CRPS) even if a
model exploited it fully, and four cheaper interventions found nothing. Recorded
as future work, not as a closed question.

## What was fixable: the uncertainty

The penalty is irreducible; the *calibration* was not. Refitting the conformal
factor by split conformal, fitted and verified on disjoint frames:

| observations | factor | coverage (held out) | width |
|---|---|---|---|
| simultaneous | 1.7541 | 0.9055 | 0.1517 |
| **realistic** | **2.1801** | **0.9096** | 0.1891 (+24%) |

Refitting for simultaneous observations independently reproduces the shipped
1.6787 (→1.7541, within 4.5%), which validates the procedure.

**In the library:** `CORRDIFF_SIGMA_SCALE_TIMED = 2.1801`, and a `sigma_scale=`
argument on `CorrDiff.predict`. Anyone whose measurements span time — which is
anyone with real vehicle data — should pass it.

## Two caveats on these numbers

**1. Ages are understated, so the penalty is a floor.** `make_track` returns the
sequence of *first* visits, but the walk keeps moving over cells it has already
seen. It needs ~199 steps to collect 90 distinct cells, so the real elapsed span
is ~2.6 h, not the 1.18 h the experiments assumed — ages are low by ~2.2×. Every
conclusion above (direction, significance, which fixes fail) is unaffected, but
the magnitude corresponds to a ~1.2 h planned transect rather than to this
particular wandering track. `make_track(..., return_steps=True)` and
`track_ages(n, steps)` now give the correct timing, and `staleness.py` uses
them; a mid-run guard raises if the simulation ever becomes a no-op again.

**2. The published numbers used a track that did not match training — now
fixed in the tool, not yet in the numbers.** Training walks a persistent,
transect-like path (`straight_bias=0.75`, ~18% revisit overhead, 1.40 h for 90
cells, radius of gyration 8.4). The runs above used an unbiased walk: 118%
revisit overhead, 2.60 h actually elapsed, radius of gyration 6.2 — a more
compact, more wandering path than any planned survey.

`scripts/staleness.py` now defaults to `--straight-bias 0.75`, so shape, revisit
rate and age accounting are coherent, and `--straight-bias 0.0` reproduces the
runs above. **The headline numbers still need one re-run under the new default
before they go in the paper.**

## Reproducing

```bash
python scripts/staleness.py eval --pickle data_raw_chrono.pickle \
    --frames-file scripts/fair_eval_frames.json --n-frames 40 \
    --models corrdiff repaint repaint_uncond --repaint-stride 5 --ablate-age
python scripts/staleness.py recalibrate --pickle data_raw_chrono.pickle \
    --frames-file scripts/fair_eval_frames.json --n-frames 60
```

Use **paired seeds** for anything comparing conditions. Independent draws leave a
~0.001 CRPS Monte-Carlo floor, larger than several of the effects above; the
paired design is what made the tight CIs possible.
