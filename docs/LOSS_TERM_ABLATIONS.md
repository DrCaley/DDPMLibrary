# Every loss term in the three paper models, ablated

The paper's central argument is a physics ladder: none (CorrDiff), soft (DistAttn's
curl/divergence penalty), hard (Stream's stream function). Before 2026-09-02 only
one rung had been tested. This is all of them, each against a matched control that
differs in exactly one term.

**Result: no auxiliary loss term in any of the three models earns its place.** Two
are actively harmful when pushed, two are inert, and none improves the metric it
was designed to improve. The base objectives -- squared error on the field, and the observation
term -- are doing the work.

| model | term | weight | verdict |
|---|---|---|---|
| Stream direction | spread correlation | 1.0 | **harmful** -- removed 2026-09-02 |
| Stream magnitude | TV smoothness on log-variance | 0.05 | **inert** |
| CorrDiff | vorticity matching | (tested at 1.0) | **null** |
| DistAttn | curl/divergence | 0.002 | **inert at the shipped weight; harmful at 1000x** |

Section 5 asks the same question of a whole component rather than a loss term --
Stream's magnitude network -- and there the answer is a clear yes. It is included as
the **power check**: the same paired test that returns TIED four times returns three
significant effects when a real effect is present.

---

## 1. Stream's spread term: harmful (removed)

Full treatment in `STREAM_LOSS_ABLATIONS.md`. It lowered the uncertainty-error
correlation it existed to raise (0.296 -> 0.223), degraded vorticity fidelity and
widened intervals, for a tied RMSE. Monotone in its weight over [0, 1], so not a
tuning problem. Confirmed across a short fine-tune, a 20,000-step fine-tune on
different hardware, and the two historical checkpoints, on both benchmarks.

**Closed 2026-09-04 by a from-scratch run.** The remaining objection was that the
shipped no-spread checkpoint is an *earlier* one (epoch 78), so "no spread term" was
confounded with "less training". A direction net trained from random init for 120,000
steps that never saw the term at any point comes out **tied on all six metrics**
(RMSE 0.08714 against 0.08651, CI [-0.000328, +0.001735]; divergence RMSE identical at
0.00946). So the shipped model is not merely a run that stopped early -- training
without the term from scratch reproduces it. -> `results_stream_scratch.pt`

## 2. Stream's magnitude TV term: inert

`lambda_tv = 0.05` was a script default the checkpoint does not record. Two arms,
6 epochs from the shipped magnitude checkpoint, differing only in
`--smooth_weight`.

Matching the shipped architecture mattered more than the ablation itself. The
trainer's defaults build an 11-channel model with a 3-layer `logvar_head`; the
shipped checkpoint is 10-channel with a `logvar_conv` head. Left alone the warm
start silently reinitialises the input layer and the entire uncertainty head --
val NLL +0.356 against -0.501 once corrected. The arms below use `--legacy_obs`
and `--head_hidden 0`.

| arm | RMSE | sigma roughness | r(sigma, err) | CRPS cal | width |
|---|---|---|---|---|---|
| lambda_tv = 0.05 (shipped) | 0.08651 | 0.000567 | 0.302 | 0.0344 | 0.2069 |
| lambda_tv = 0 (control) | 0.08651 | 0.000564 | 0.300 | 0.0344 | 0.2071 |

```
shipped - control   rmse       -0.000005  CI [-0.000010, -0.000001]  SIGNIFICANT
                    roughness  +0.000003  CI [-0.000004, +0.000010]  TIED
                    r_unc      +0.001569  CI [-0.000725, +0.003854]  TIED
                    crps_cal   -0.000016  CI [-0.000045, +0.000013]  TIED
```

**Adequately powered, and null.** The only trainable parameters are the logvar
head, and they diverged **4.5%** between arms -- a thousand times more than the
DistAttn arms below -- so the term did change the model. It simply changed nothing
that matters, including the sigma roughness it exists to suppress. The significant
RMSE difference is 5e-6, i.e. 0.006%, and is not a reason to keep it.

Sigma roughness here is the mean absolute difference between neighbouring ocean
cells of the predicted sigma -- directly the salt-and-pepper character the term was
added to remove.

-> `results_magnitude_tv.pt`

## 3. CorrDiff's vorticity term: null

Full treatment in `PAPER_NUMBERS.md` §3. Fine-tuned +/- a vorticity term on
byte-identical batches against a lambda = 0 control: tied on every metric including
vorticity RMSE itself, while the training term improved 9%. Without the control it
would have been reported as a significant eddy-recall win.

The same term *does* transfer on Stream (§7 there) -- significant and monotone in
lambda -- but moves vorticity correlation only 0.560 to 0.570 across a 5x weight
change, against a 0.165 gap to CorrDiff. Real, and far too weak to matter.

## 4. DistAttn's curl/divergence term: inert at 0.002, harmful at 1000x

This is the one the paper's "soft physics" rung actually depends on. It is now
settled -- see the paired retrain at the end of this section.

Two arms, 8 epochs from the shipped checkpoint, differing only in `lambda_cd`.
Batches verified byte-identical across arms at step 1 -- the eps, curl/div and
observation terms all match to ten decimals, and the totals differ by exactly the
term (`0.00016037 + 0.002 x 0.00984855 + 0.00031052 = 0.00049058`). CUDA backward
non-determinism means the arms are not bit-reproducible run to run, ~1e-8 per step.

At the shipped weight the arms came out **1.1e-5 apart** and every metric tied at
the 1e-6 level -- a test with no power, since 8 epochs cannot move a model against
142 epochs of the term already in the checkpoint.

A **1000x dose arm** (`lambda_cd = 2.0`) resolves it. It moved the weights **144x
further** from the control (1.6e-3 against 1.1e-5), so it has real power:

| arm | RMSE | angle | divRMSE | vortRMSE | vortCorr | width |
|---|---|---|---|---|---|---|
| 0.002 (shipped) | 0.07315 | 0.78569 | 0.00809 | 0.01385 | 0.629 | 0.1892 |
| 0 (control) | 0.07315 | 0.78574 | 0.00808 | 0.01385 | 0.629 | 0.1892 |
| **2.0 (1000x)** | 0.07347 | 0.78429 | **0.00812** | 0.01397 | 0.623 | 0.1865 |

```
dose 2.0 - control   div_rmse   +0.000035  CI [+0.000002, +0.000066]  SIGNIFICANT
                     rmse       +0.000315  CI [-0.001455, +0.001841]  TIED
                     vort_corr  -0.005662  CI [-0.012248, +0.001070]  TIED
                     crps_cal   -0.000037  CI [-0.000804, +0.000606]  TIED
```

**Divergence RMSE gets significantly worse** -- the one quantity the term exists to
control. Its validation loss also came out worse than the resumed baseline (0.00071
against 0.00070), so the trainer never wrote a best-checkpoint at all.

So the term's gradient does not point at better divergence fidelity: at 0.002 it is
too weak to do anything, and at 2.0 it actively degrades its own target. Across three
orders of magnitude there is no weight at which it helps. That is the same signature
as Stream's spread term.

**Caveat closed 2026-09-04 by a paired retrain from a term-naive base.** The arms
above are fine-tunes from a checkpoint that had already absorbed 142 epochs of the
term, which is why the 0.002 arm had no power. This is the test that objection asks
for: both arms warm-started from the **pre-timecond DistAttn base (epoch 136), whose
trainer had no curl/divergence term at all**, then trained 40 epochs on the same
Titan Xp with the same seed, differing only in `lambda_cd`.

| arm | RMSE | angle | divRMSE | vortRMSE | vortCorr | CRPScal | width |
|---|---|---|---|---|---|---|---|
| 0.002, from term-naive base | 0.07527 | 0.82256 | 0.00815 | 0.01409 | 0.607 | 0.0301 | 0.2021 |
| 0, control | 0.07535 | 0.80941 | 0.00813 | 0.01404 | 0.613 | 0.0302 | 0.2017 |

```
0.002 - control, paired over 40 cases:
   rmse       -0.000080  CI [-0.001050, +0.000927]  TIED
   angle      +0.013148  CI [-0.009003, +0.038849]  TIED
   div_rmse   +0.000018  CI [-0.000003, +0.000039]  TIED
   vort_rmse  +0.000056  CI [-0.000035, +0.000155]  TIED
   vort_corr  -0.005338  CI [-0.014937, +0.002638]  TIED
   crps_cal   -0.000099  CI [-0.000507, +0.000276]  TIED
```

**And this time the test has power.** The arms ended **1.5e-1 apart** in relative
weight distance -- 13,700x further than the 8-epoch pair (1.1e-5) and 100x further
than the 1000x dose arm (1.6e-3). Two models 15% apart in weight space, one trained
with the term active for its entire run and one that never saw it, are
indistinguishable on every metric. The control also reached the nominally *better*
validation loss (0.000761 against 0.000784), and divergence RMSE -- the term's own
target -- is again nominally worse with the term, matching the 1000x dose.

So the term is not merely too weak at 0.002: across an entire training run from a
base that never saw it, it does nothing, and at 1000x it degrades its own objective.
-> `results_distattn_paired_retrain.pt`

-> `results_distattn_curldiv.pt`

## 5. Stream's magnitude network: the one component that pays (power check)

Not a loss term but a whole second network, and the same question -- does it earn
its place? It doubles as the control on everything above: if four ablations in a row
come back null, the obvious objection is that the test cannot see anything. It can.

Stream predicts a scalar stream function and takes its curl, which is exactly
divergence-free but suppresses amplitude, and squared error then pulls toward a
blurred mean. A separate network predicts speed and the two are fused. That premise
had never been tested end to end. Method: monkeypatch only `fuse_coupled`, so
conditioning, sampler, seeds, divergent-component addition and land masking stay
byte-identical across arms and the sole difference is whether the magnitude net is
consulted.

| variant | RMSE | angle | vortRMSE | vortCorr | CRPScal | factor | covcal | width |
|---|---|---|---|---|---|---|---|---|
| shipped (with magnitude net) | **0.08651** | 0.86630 | 0.01654 | **0.571** | **0.0344** | 3.162 | 0.894 | 0.2068 |
| no magnitude net | 0.10352 | 0.89865 | 0.01718 | 0.503 | 0.0435 | 11.497 | 0.899 | 0.3040 |

```
shipped - no-magnitude-net, paired over 40 cases:
   rmse       -0.017005  CI [-0.025041, -0.009069]  SIGNIFICANT
   vort_corr  +0.067982  CI [+0.041978, +0.094385]  SIGNIFICANT
   crps_cal   -0.009136  CI [-0.012000, -0.006226]  SIGNIFICANT
   angle      -0.032354  CI [-0.067642, +0.004595]  TIED
   vort_rmse  -0.000638  CI [-0.001394, +0.000087]  TIED
```

**The magnitude network earns its place**: 16% lower RMSE, vorticity correlation
0.571 against 0.503, calibrated CRPS 21% better.

The sharpest number is the calibration factor. Without the magnitude net the raw
ensemble spread needs an **11.5x** inflation to reach 90% coverage, against 3.16x
with it. The diffusion draws' own speeds really do collapse, by a factor measurable
to a few percent -- the magnitude-collapse premise the two-network design was built
on, confirmed directly for the first time. The calibrated intervals are 47% wider
without it (0.304 against 0.207) at the same coverage: same guarantee, far less
information.

**As a power check this is the important one.** The same 40-case paired bootstrap
that returns TIED for four loss terms returns three significant effects here, two
with confidence intervals nowhere near zero. The nulls above are properties of the
terms, not of the test.

-> `results_stream_magnitude_value.pt`

## What this means for the paper

The physics ladder survives, and the soft rung is now directly evidenced rather
than inferred. DistAttn's own curl/divergence term is inert at its shipped weight
and significantly degrades divergence fidelity when pushed 1000x -- so "soft physics
does nothing measurable" is supported both by the CorrDiff vorticity ablation and by
a powered dose-response on DistAttn itself.

As of 2026-09-04 it no longer rests on fine-tunes at all. The two objections a
reviewer would raise first -- *you only removed the term from a model that had
already absorbed it*, and *your no-spread Stream checkpoint merely stopped early* --
are both answered by full-length training runs: the DistAttn arms trained 40 epochs
each from a base whose trainer had no such term and ended 15% apart in weight space,
tied on all six metrics (section 4); the Stream direction net trained 120,000 steps
from random init having never seen the spread term, tied on all six (section 1). Both
nulls now have measured power behind them, and neither forced a change to a shipped
model.

The stronger and better-evidenced claim is the one that emerged along the way:
**four auxiliary loss terms were added to these models on physical or statistical
reasoning, and not one of them improves the metric it was designed to improve.**
Three were never tested before this week.

And the method has teeth: applied to Stream's magnitude network (section 5) the same
test returns a 16% RMSE improvement with a confidence interval far from zero. Four
nulls from a test that can detect a real effect is a finding; four nulls from a test
that detects nothing is an artefact. This is the former.
