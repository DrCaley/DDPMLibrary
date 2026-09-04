# Runtime dials and inference cost, measured

`LOSS_TERM_ABLATIONS.md` covers the training-time terms. This covers the
inference-time dials, which had never been measured, and the per-model cost of a
field. Sections 1-2 are CorrDiff's `n_draws` and `sensor_noise`; section 3 is Stream's
ensemble size and 3b is DistAttn's -- the one default that came back wrong; section 4
is what each model costs to run; section 5 is a RePaint default that disagreed with its
own calibration. The measured sections are
40-case paired bootstraps on `ocean_bench_v1`, the same harness as the loss-term
work, with the conformal factor refit per arm (split conformal, fit on the first
20 frames, coverage verified on the other 20).

-> `benchmark/corrdiff_noise_and_draws.py`, `benchmark/corrdiff_noise_curve.py`
-> `benchmark/results_corrdiff_noise_draws.pt`

## 1. CorrDiff's ensemble size: the shipped 20 is right, and the factor is not portable

All four sizes are nested subsets of ONE 40-draw sampling run per case -- only
`ddim_sample_residual` is intercepted and cached by seed -- so the comparison is
exactly paired and costs a single 40-draw run.

| n_draws | RMSE | angle | vortRMSE | vortCorr | CRPScal | factor | covcal | width |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.06892 | 0.73425 | 0.01326 | 0.684 | 0.0328 | 5.053 | 0.909 | 0.2833 |
| 10 | 0.06784 | 0.73132 | 0.01304 | 0.692 | 0.0311 | 4.116 | 0.909 | 0.2632 |
| **20 (shipped)** | **0.06684** | **0.71926** | **0.01289** | **0.699** | **0.0304** | **3.805** | **0.912** | **0.2565** |
| 40 | 0.06677 | 0.71723 | 0.01285 | 0.701 | 0.0302 | 3.688 | 0.914 | 0.2509 |

```
10 minus 20   rmse +0.001003 [+0.000276, +0.001752]  SIGNIFICANT   (worse)
              angle +0.012056 [+0.002146, +0.023099] SIGNIFICANT   (worse)
              vort_rmse +0.000143 [+0.000071, +0.000219] SIGNIFICANT (worse)
              vort_corr -0.006666 [-0.010037, -0.003400] SIGNIFICANT (worse)
              crps_cal +0.000716 [+0.000387, +0.001065] SIGNIFICANT (worse)

40 minus 20   rmse -0.000066 [-0.000640, +0.000584]  TIED
              angle -0.002037 [-0.009973, +0.005960] TIED
              crps_cal -0.000205 [-0.000506, +0.000100] TIED
              vort_corr +0.001613 [-0.000544, +0.003756] TIED
              vort_rmse -0.000042 [-0.000083, -0.000002] SIGNIFICANT (marginal)
```

Halving to 10 is significantly worse on **all five** metrics; doubling to 40 buys
nothing but a marginal vorticity RMSE gain, for twice the sampling cost. **20 is
the right default** -- the first of these dials to be confirmed rather than
inherited.

**The conformal factor is not portable across ensemble sizes.** It runs 5.053 /
4.116 / 3.805 / 3.688 at n = 5 / 10 / 20 / 40: the raw ensemble std shrinks with
ensemble size, so a factor fitted at 20 and applied at 5 under-covers by about a
third. Nothing in the library said so, and `predict(n_draws=5, calibrate=True)`
would silently have returned intervals too narrow for their stated level. All four
predictors now raise a `RuntimeWarning` when `n_draws` differs from the size their
factor was fitted at (`calibration.py`). The shift is far larger than the Gaussian
small-sample correction predicts -- the ensemble mean improves with n as well, so
there is no closed-form fix, only refitting.

## 2. CorrDiff's sensor-noise channel: works as designed, and the effect is small

`sensor_noise` is a trained conditioning channel, not a post-hoc widening, so a
bare sweep on the noise-free benchmark would only measure the cost of lying to the
model. The question worth asking is whether the channel does its job when the
observations really are noisy. Arms at each level see byte-identical noisy
observations, so "told" and "not told" differ in exactly the channel.

| arm | obs error | channel | RMSE | vortCorr | CRPScal | factor | covcal | width |
|---|---|---|---|---|---|---|---|---|
| A | none | 0 | 0.06684 | 0.699 | 0.0304 | 3.805 | 0.912 | 0.2565 |
| B | 5% | 0 | 0.06678 | 0.699 | 0.0304 | 3.788 | 0.911 | 0.2553 |
| C | 5% | 0.05 | 0.06670 | 0.699 | 0.0300 | 3.631 | 0.912 | 0.2487 |
| D | 10% | 0 | 0.06719 | 0.696 | 0.0305 | 3.779 | 0.912 | 0.2558 |
| E | 10% | 0.10 | 0.06708 | 0.698 | 0.0297 | 3.467 | 0.912 | 0.2416 |

```
telling the model, at 5%    (C-B)   vort_rmse -0.000018  vort_corr +0.000890  crps_cal -0.000393   all SIGNIFICANT
telling the model, at 10%   (E-D)   vort_rmse -0.000036  vort_corr +0.001804  crps_cal -0.000794   all SIGNIFICANT
cost of the noise, at 5%    (B-A)   everything TIED
cost of the noise, at 10%   (D-A)   vort_rmse +0.000080  vort_corr -0.002966  SIGNIFICANT; rmse, angle, crps TIED
```

**The channel is real, and it is the cleanest dose-response in the project**: every
significant effect roughly doubles from 5% to 10% noise (CRPS -0.00039 to -0.00079,
vorticity correlation +0.00089 to +0.00180, vorticity RMSE -0.000018 to -0.000036).
That is what a correctly functioning noise-conditioning channel should do, and it is
the behaviour the four dead loss terms never showed.

**It is also small.** At the largest level the model supports, telling it the truth
about its sensors improves calibrated CRPS by 2.6% (0.0305 -> 0.0297) and narrows the
calibrated interval 5.6% (0.2558 -> 0.2416) at equal coverage. Keep the channel; do
not build an argument on it.

**CorrDiff is robust to observation noise at these levels.** 5% costs nothing
measurable; 10% costs a significant but tiny amount of vorticity fidelity and no
measurable RMSE or CRPS. The reason is scale: 10% of the field standard deviation is
0.0106 m/s, only 16% of CorrDiff's own 0.0668 m/s RMSE, so sensor error is well
inside the reconstruction error. That is a useful robustness statement for the
deployment case, and a limit on how much the noise channel could ever matter here.

## 3. Stream's ensemble size: 20 confirmed, same shape as CorrDiff

`STREAM_DEFAULT_N_DRAWS = 20` was justified in config only as "matches CorrDiff".
Swept the same way -- nested subsets of one 40-member ensemble per case, factor refit
per size:

| n_draws | RMSE | angle | divRMSE | vortRMSE | vortCorr | CRPScal | factor | covcal | width |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.08708 | 0.87303 | 0.00949 | 0.01678 | 0.565 | 0.0348 | 3.484 | 0.899 | 0.2164 |
| 10 | 0.08673 | 0.86732 | 0.00947 | 0.01664 | 0.569 | 0.0345 | 3.227 | 0.893 | 0.2067 |
| **20 (shipped)** | **0.08651** | **0.86630** | **0.00946** | **0.01654** | **0.571** | **0.0344** | **3.162** | **0.894** | **0.2068** |
| 40 | 0.08651 | 0.86658 | 0.00946 | 0.01651 | 0.572 | 0.0343 | 3.129 | 0.893 | 0.2061 |

```
5 minus 20    every metric SIGNIFICANTLY worse (rmse +0.000566, crps_cal +0.000466)
10 minus 20   rmse +0.000220, div_rmse, vort_rmse, vort_corr, crps_cal SIGNIFICANT (worse); angle TIED
40 minus 20   rmse -0.000002 TIED, crps_cal TIED, vort TIED; only div_rmse -0.000005 SIGNIFICANT
```

Identical shape to CorrDiff: below the default costs real accuracy, above it buys
nothing. RMSE at 40 matches 20 to five decimals, so 20 is the saturation point.
The factor moves 3.484 -> 3.129 across the sweep, the same non-portability as §1.

## 3b. DistAttn's ensemble size: 20 is better than the inherited 10 (not applied)

The one dial that came back wrong. `DISTATTN_DEFAULT_N_DRAWS` was 10, inherited from
the collaborator's evaluation rather than measured here. Same protocol as §1 and §3 --
each draw sampled once and reused across sizes, factor refit per size.

| n_draws | RMSE | angle | divRMSE | vortRMSE | vortCorr | CRPScal | factor | covcal | width |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.07572 | 0.81239 | 0.00835 | 0.01439 | 0.595 | 0.0305 | 2.003 | 0.914 | 0.2138 |
| **10 (shipped)** | 0.07336 | 0.79116 | 0.00809 | 0.01386 | 0.627 | 0.0288 | 1.598 | 0.918 | 0.1898 |
| **20** | **0.07180** | 0.77823 | 0.00798 | 0.01366 | **0.641** | **0.0279** | 1.359 | 0.891 | 0.1637 |
| 40 | 0.07304 | **0.77401** | **0.00795** | **0.01361** | 0.648 | 0.0282 | 1.340 | 0.889 | 0.1552 |

```
 5 minus 10   every metric SIGNIFICANTLY worse
20 minus 10   div_rmse  -0.000117  CI [-0.000164, -0.000070]  SIGNIFICANT (better)
              vort_rmse -0.000199  CI [-0.000353, -0.000047]  SIGNIFICANT (better)
              vort_corr +0.013469  CI [+0.004953, +0.022836]  SIGNIFICANT (better)
              crps_cal  -0.000930  CI [-0.001695, -0.000052]  SIGNIFICANT (better)
              rmse      -0.001557  CI [-0.003162, +0.000099]  TIED (at the edge)
              angle     -0.012926  CI [-0.031654, +0.005315]  TIED
40 minus 10   div_rmse, vort_rmse, vort_corr SIGNIFICANT (better); rmse, angle, crps_cal TIED
```

**Going from 10 to 20 draws significantly improves four of six metrics**, including
calibrated CRPS and all three divergence/vorticity metrics, and narrows the calibrated
interval 14% (0.1898 -> 0.1637). RMSE improves 2.1% but only reaches the edge of
significance. 40 is not better than 20 -- its RMSE is worse (0.07304 against 0.07180)
and its CRPS gain over 10 is no longer significant -- so 20, not more, is the target.

Read the coverage column alongside it: at n=10 the held-out coverage is 0.918, i.e.
**over-covering**, and at n=20 it is 0.891, just under nominal. Part of what 20 buys is
simply a less conservative interval, which is why the width drops as the CRPS improves.

**Measured, not applied.** The default is still 10 as of 2026-09-04, because raising it
is not a one-line change:

1. It **doubles the cost** -- 38.1 s/field to ~76 s, taking DistAttn from 15x CorrDiff
   to ~29x (§4). It is already the most expensive of the three paper models.
2. `DISTATTN_SIGMA_SCALE_TIMED = 1.621` was **fitted at n_draws=10**. Raising the default
   means refitting it *and* re-running the blind check on `ocean_bench_v1b`, or the
   paper's calibration claim ("the shipped factors hold blind", `PAPER_NUMBERS.md`)
   would describe a configuration that no longer ships. That is ~1.5 h of inference.
3. The collaborators' existing results were produced at 10, so raising it invalidates
   their numbers rather than merely shifting ours. That makes it a decision about their
   work, and it was put to them rather than taken unilaterally.

The model ranking does not change either way -- DistAttn stays third of the three.

**A design fix came out of investigating it.** The calibration guards initially compared
`n_draws` against each model's *default*, which conflates "the speed/quality choice we
ship" with "the ensemble size the factor was fitted at". Raising a default would then
have silenced the very warning that should fire. Those are now separate constants
(`X_DEFAULT_N_DRAWS` vs `X_FITTED_N_DRAWS`) and the guard reads the latter, so changing a
default without refitting warns instead of going quiet.

## 4. What each model costs per field

Measured on one device with a warmup field excluded, at the paper configurations
(`benchmark/timing_bench.py`, which now takes `--device` instead of hardcoding CUDA).
These are Apple MPS numbers, so they are comparable to each other but not to the
Titan Xp figures quoted for CorrDiff and RePaint in `PAPER_NUMBERS.md`. The harness
prints one decimal, so VCNN is reported as an upper bound rather than a value.

| model | s/field | relative to CorrDiff |
|---|---|---|
| VCNN | <0.05 | <0.02x |
| **Stream** (full field, 20 draws) | **0.2** | **0.08x** |
| GP (CPU) | 0.2 | 0.08x |
| **CorrDiff** (1 h, 20 draws) | **2.6** | **1x** |
| **DistAttn** (10 draws) | **38.1** | **15x** |
| RePaint (1 h, 10 draws, stride 5) | 133.6 | 51x |

Two things worth saying in the paper:

**Stream is the cheapest of the three, by 13x.** The hard-physics model is not only
the most constrained, it is the fastest -- two DPM-Solver++ steps against CorrDiff's
50. So the physics ladder has a cost axis as well as an accuracy axis: Stream is
capped on accuracy and 13x cheaper, CorrDiff wins accuracy at 13x the cost. That is
a real deployment trade-off, not just a loss.

**DistAttn is the expensive one**, 15x CorrDiff, because it cross-attends over up to
361 observation tokens at every one of its 100 sampling steps. Its published RMSE is
also the worst of the three, so it is dominated on both axes here.

The RePaint ratio holds up across devices: 51x CorrDiff on MPS against 41x measured
on the Titan Xp. The cost argument for excluding it does not depend on the machine.

## 5. RePaint's stride: the shipped default was not the calibrated one

Found while auditing the dials rather than by measurement, and worth recording because
it is the same failure as §1 in a different variable.

`REPAINT_SIGMA_SCALE_TIMED = 2.4513` was fitted at **stride 5** -- confirmed in
`results_repaint_calibration.pt`, whose metadata reads `{'n_draws': 10, 'stride': 5,
'cutoff_h': 1.0}`. So was the v1b blind calibration, and so was the 49.7 s/field cost.
But `REPAINT_STRIDE` was **1**, so a caller using the library defaults got a stride-1
chain with a factor fitted for stride 5, silently. Stride sets how much of the
1000-step chain is walked (1000 calls against 200), which changes the ensemble spread,
which is exactly what the factor rescales.

Fixed two ways: the default is now 5, the configuration every reported RePaint number
came from, and `resolve_sigma_scale` warns when either `n_draws` or `stride` differs
from the fitted value, listing every mismatch in one warning.

**What is still not known:** stride 1 has never been scored against stride 5 on the
benchmark. Stride 1 costs about 5x more -- roughly 7 h for the 40 cases -- so it was
not affordable, and more sampling steps would normally mean better samples. So stride 5
is defensible as *the calibrated setting*, and is not claimed to be optimal. If RePaint
ever moves from a footnote to a contribution, that sweep is the first thing to run.
