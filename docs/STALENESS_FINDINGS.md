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

**These are the corrected-walk numbers** (training-consistent transect,
`straight_bias=0.75`, ages from the walk's own step count). 40 frames, paired
seeds, identical tracks across models.

| model | clean CRPS | stale CRPS | penalty | 95% CI | significant |
|---|---|---|---|---|---|
| corrdiff (trained conditioning) | 0.02207 | 0.02631 | **+19.2%** | [+0.00219, +0.00711] | yes |
| repaint (guided sampling + priors) | 0.02213 | 0.02684 | **+21.3%** | [+0.00288, +0.00697] | yes |
| repaint_uncond (guided, no priors) | 0.02723 | 0.03320 | **+21.9%** | [+0.00340, +0.00882] | yes |

Coverage at the 90% level degrades in step:

| model | clean | stale |
|---|---|---|
| corrdiff | 0.8964 | 0.8022 |
| repaint | 0.7358 | 0.6205 |
| repaint_uncond | 0.7600 | 0.6461 |

**It is not architecture-specific.** Trained conditioning, guided sampling with
priors, and guided sampling without priors all lose the same ~20%, and their
confidence intervals overlap completely. A plain CNN (vcnn) lost a comparable
amount in earlier runs.

*Earlier runs on the unbiased walk reported ~11%. That walk understated
observation ages (see caveats) and did not match the training path; the ~20%
figures above supersede them.*

## Four attempted fixes, all null

| fix | result | why it failed |
|---|---|---|
| **Age conditioning** (DistAttn's mechanism) | +0.00061, CI [−0.00092, +0.00223] | No effect, replicated on the corrected walk. Hiding the ages entirely costs nothing measurable, and DistAttn does not significantly degrade under staleness at all (−0.00029, ns). Its robustness comes from fitting observations loosely (0.0194 vs CorrDiff's 0.0060), not from knowing their ages. |
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

## Method notes

**Ages come from the walk's step count, not list position.** `make_track`
returns first visits, but the walk keeps moving over cells it has already seen,
so cell *k* was reached at step *s_k > k*. Using list position understates
staleness — on the unbiased walk by ~2.2x, which is why the earlier runs
reported ~11% instead of ~20%. `make_track(..., return_steps=True)` and
`track_ages(n, steps)` give the correct timing, `staleness.py` uses them, and a
mid-run guard raises if the simulation ever becomes a no-op again.

**The evaluation track matches the training track.** `--straight-bias 0.75` is
the generator the models were trained with: transect-like, ~18% revisit
overhead, 1.40 h for 90 cells, radius of gyration 8.4 (training measures 8.7).
`--straight-bias 0.0` reproduces the earlier unbiased-walk runs (118% revisit
overhead, radius of gyration 6.2) if you need them.

**Paired seeds throughout.** Each condition samples identical diffusion noise
and differs only in the observations. Independent draws leave a ~0.001 CRPS
Monte-Carlo floor, larger than several of the effects here.

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

---

## Related: is CorrDiff more accurate than RePaint?

No. 40 frames, per-frame bootstrap, common mask 3787 cells, standard
(simultaneous) benchmark:

| metric | corrdiff | repaint | difference | verdict |
|---|---|---|---|---|
| CRPS | 0.0232 | 0.0255 | −0.0023 [−0.0056, +0.0010] | tied |
| RMSE | 0.0434 | 0.0453 | −0.0020 [−0.0076, +0.0038] | tied |
| RMSE @unobserved | 0.0439 | 0.0459 | −0.0020 [−0.0078, +0.0039] | tied |
| angle error | 26.37 | 27.36 | −0.99 [−5.37, +4.68] | tied |
| **eddy recall** | 0.442 | **0.487** | −0.046 [−0.085, −0.006] | **RePaint better** |
| SSIM | 0.5699 | 0.5693 | +0.0006 [−0.018, +0.019] | tied |

Every accuracy metric ties; the one significant difference favours RePaint.
Do not claim an accuracy win in either direction. CorrDiff's defensible
advantages are **calibration** (coverage 0.896 vs 0.736 on clean observations)
and **speed** (~15 s vs ~4 min per field).
