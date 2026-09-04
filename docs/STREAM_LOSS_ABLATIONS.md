# Two Stream loss terms: one that works and doesn't matter, one that costs

Both experiments are controlled fine-tunes from the shipped Stream direction
checkpoint, 1000 or 3000 steps, arms differing in exactly one loss term.
Determinism verified: step 1 is bit-identical across arms (recon 0.7851359844,
angle 0.1575206071, mag 0.0135164773), and a given arm reproduces bit-identically
across runs. Scored on the frozen benchmarks; the scoring code reproduces the
saved CUDA run to five decimals on all five metrics.

**Headline: the spread term is net negative and should be dropped on a retrain.
It fails at its own objective.**

**Note on sampler steps.** Every measurement in this document was taken with
`STREAM_DPMPP_STEPS = 6`, the default at the time. That default was itself
re-swept afterwards and is now 2 (see `PAPER_NUMBERS.md` §7c), which shifts all
Stream absolutes -- e.g. RMSE 0.0919 → 0.0865, conformal factor 2.909 → 3.304, and
then 3.165 once `STREAM_UNC_SMOOTH_SIGMA` moved 0.8 → 3.2 (the factor is coupled to
both).
The spread-term contrasts here are between arms measured at the same step count,
so they are unaffected; only the absolute values are superseded. The vorticity term is real but ~30x too weak to
matter.

---

## 0. The measurement that motivated all of this

`EDDY_METRIC_BIAS.md` recommends reporting **vorticity RMSE** instead of any
Okubo-Weiss eddy statistic, because vorticity needs no Helmholtz decomposition
and so carries no boundary-condition ambiguity. That number had never been
measured for Stream. It is not good.

40 cases, `ocean_bench_v1`, 3515 interior ocean cells, ensemble mean at each
model's reported configuration:

| model | vorticity RMSE | vs predicting zero vorticity | corr with truth |
|---|---|---|---|
| CorrDiff (1 h cutoff) | 0.01219 | -35.4% | 0.725 |
| DistAttn | 0.01385 | -26.7% | 0.634 |
| Stream (full field) | 0.01822 | **-3.5%** | 0.518 |

Predicting zero vorticity everywhere scores 0.01889. Stream beats that by 3.5%.
Paired bootstrap against CorrDiff: +0.00603, CI [+0.00508, +0.00699],
significant.

The model whose architectural premise is representing rotational structure
exactly is the worst of the three at reproducing it.

**It is not ensemble smearing.** The obvious explanation -- 20 draws with eddies
in different places averaging into mush -- is wrong:

| | vorticity RMSE | corr | amplitude ratio |
|---|---|---|---|
| n_draws = 1 | 0.03243 (+71.7% vs zero) | 0.318 | **1.892** |
| n_draws = 20 | 0.01822 (-3.5%) | 0.518 | 1.062 |

A single draw carries nearly **twice** the true vorticity amplitude. Averaging 20
cancels that to 1.06, which is essentially correct. So the amplitude is right and
the *pattern* is wrong. This is consistent with the grid-scale checkerboard modes
that `STREAM_UNC_SMOOTH_SIGMA` exists to suppress: central differences have a
Fourier symbol vanishing at Nyquist, so those modes are unconstrained by the
divergence-free structure, and vorticity is a derivative, which amplifies them.

→ `results_stream_vorticity_draws.pt`

---

## 1. A vorticity term on Stream: works, dose-responsive, too small to matter

Unlike CorrDiff -- where the same term was null on every metric -- there was a
real deficit here for it to attack. Three arms, 3000 steps, differing only in
`lambda_vort`.

| arm | vorticity RMSE | corr | RMSE | angle |
|---|---|---|---|---|
| lambda = 0 (control) | 0.01736 | 0.560 | 0.09181 | 0.8939 |
| lambda = 1 | 0.01709 | 0.565 | 0.09158 | 0.8950 |
| lambda = 5 | 0.01693 | 0.570 | 0.09162 | 0.8985 |

Isolated against the matched control, n_draws = 20:

```
lambda=1 - lambda=0   vort_rmse  -0.000272  CI [-0.000403, -0.000156]  SIGNIFICANT
                      vort_corr  +0.004829  CI [+0.002354, +0.007260]  SIGNIFICANT
                      rmse                                             TIED
                      angle                                            TIED
lambda=5 - lambda=0   vort_rmse  -0.000431  CI [-0.000666, -0.000229]  SIGNIFICANT
                      vort_corr  +0.010077  CI [+0.005146, +0.015733]  SIGNIFICANT
```

Significant, monotone in lambda, and replicated at n_draws = 1. The term
transfers to held-out fields, which is a cleaner outcome than the CorrDiff null.

**And it does not matter.** Vorticity correlation moves 0.560 → 0.570 across a
5x change in lambda. CorrDiff sits at 0.725. Closing that gap at +0.005 per unit
lambda would need lambda ~ 165. It buys nothing in RMSE or angle. During training
the term itself fell 8.4x (lambda=1) and 18x (lambda=5) -- it optimises its own
objective hard and almost none of that reaches the field.

→ `results_stream_vorticity.pt`

---

## 2. The spread term degrades the model, including at its own job

The control arm above improved vorticity far more than the vorticity term did
(+0.041 correlation at n_draws=20, +0.182 at n_draws=1) -- and the control has no
vorticity term at all. It differed from the shipped model in two ways: 3000 extra
steps, and the spread term removed. Resolving that confound is the main result.

**It is not the extra training.** A 1000-step arm *with* the spread term on lands
exactly where the shipped checkpoint already was:

| n_draws = 1, v1 | vorticity RMSE | corr | amplitude |
|---|---|---|---|
| shipped (spread on, converged) | 0.03243 | 0.318 | 1.892 |
| 1000 steps, spread **ON** | 0.03267 | 0.316 | 1.905 |
| 1000 steps, spread **OFF** | 0.01951 | 0.501 | 1.137 |

A step-count time course confirms the shape -- the gain accumulates over ~1000
steps and then plateaus exactly, rather than continuing:

| steps (spread off) | 0 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|
| vorticity RMSE | 0.03243 | 0.02910 | 0.02326 | 0.01951 | 0.01957 |
| corr | 0.318 | 0.351 | 0.425 | 0.501 | 0.501 |
| amplitude | 1.892 | 1.702 | 1.362 | 1.137 | 1.135 |

### The matched ablation, both draw counts, both benchmarks

Vorticity, spread OFF minus spread ON at 1000 steps:

| | vorticity RMSE | vorticity corr |
|---|---|---|
| v1, n_draws = 20 | -0.00101 CI [-0.00137, -0.00065] | +0.0436 CI [+0.0313, +0.0570] |
| v1b, n_draws = 20 | -0.00111 CI [-0.00142, -0.00083] | +0.0492 CI [+0.0368, +0.0637] |
| v1, n_draws = 1 | -0.01316 CI [-0.01536, -0.01105] | +0.1849 CI [+0.1596, +0.2115] |

All significant. The n_draws = 20 effect replicates almost exactly on the
independent set. The effect is ~4x larger at n_draws = 1, since ensemble
averaging already cancels much of the noise.

### Calibration: the term fails at the thing it was added for

The spread term exists to make the uncertainty map correlate with where the model
is actually wrong. Matched pair, n_draws = 20, `ocean_bench_v1`, same
split-conformal protocol as `uncertainty_final.py`:

| arm | CRPS raw | CRPS calibrated | cov raw | r(sigma, error) | factor | cov calibrated | width |
|---|---|---|---|---|---|---|---|
| shipped (spread on) | 0.0400 | 0.0376 | 0.5023 | 0.223 | 3.140 | 0.9032 | 0.2550 |
| 1000 steps, spread ON | 0.0400 | 0.0377 | 0.5023 | 0.225 | 3.151 | 0.9041 | 0.2563 |
| 1000 steps, spread OFF | 0.0407 | **0.0368** | 0.4560 | **0.289** | 3.222 | 0.9025 | **0.2351** |

```
spread OFF - spread ON (v1)     crps_cal  -0.000862  CI [-0.001272, -0.000473]  SIGNIFICANT
                                r_unc     +0.063716  CI [+0.035371, +0.092032]  SIGNIFICANT
spread OFF - spread ON (v1b)    crps_cal  -0.000358  CI [-0.000629, -0.000038]  SIGNIFICANT
                                r_unc     +0.045945  CI [+0.021345, +0.068925]  SIGNIFICANT
```

`r(sigma, error)` is **higher without the term** -- 0.289 against 0.225 on v1,
0.257 against 0.211 on v1b, both significant. The term's stated purpose is to
raise that number and it lowers it. Removing it also gives better calibrated
CRPS and **8% narrower intervals** at matched 90% coverage.

**A trap worth recording.** On *raw* sigma the CRPS comparison favours
spread-ON (+0.00066, significant). That is an artefact: both arms are badly
under-dispersed at ~0.50 raw coverage against a 0.90 target, so a wider raw
spread scores better for the wrong reason. On calibrated sigma the sign flips.
Report CRPS on calibrated sigma or this term looks beneficial when it is not.

### The natural experiment: the shipped model is a regression from its own predecessor

The strongest evidence needs no fine-tuning at all. The shipped direction
checkpoint is **bit-identical** to `Models/StreamFn_Cond_x0_mag_spread.pt`
(epoch 48), whose recorded `init` is
`best_streamfncond_minsnr5_mag0.2_ang1_lags13-25_div_free_cosine.pt`. And
`Models/StreamFn_Cond_x0_mag.pt` carries exactly that config -- same
`x0_streamfn_cond` architecture, `cond_ch` 10, `lambda_angle` 1.0, `lambda_mag`
0.2, `min_snr_gamma` 5.0, lags (13, 25), div-free noise, cosine schedule, and no
spread term -- at **epoch 78**.

So the shipped Stream model is a fully-trained no-spread model plus 48 epochs
with the spread term added. That history is the experiment, run on two real
training runs rather than short fine-tunes:

| n_draws = 20 | RMSE | vort RMSE | vort corr | CRPS cal | r(sigma, err) | factor | width |
|---|---|---|---|---|---|---|---|
| predecessor, 78 ep, **no spread** | 0.09194 | **0.01743** | **0.562** | **0.0368** | **0.296** | 2.909 | **0.2297** |
| shipped, +48 ep, **spread on** | **0.09082** | 0.01822 | 0.518 | 0.0376 | 0.223 | 3.140 | 0.2550 |

```
predecessor - shipped, n_draws=20   rmse       +0.001123  CI [-0.000869, +0.002977]  TIED
                                    vort_rmse  -0.000793  CI [-0.001476, -0.000070]  SIGNIFICANT
                                    vort_corr  +0.043627  CI [+0.024232, +0.063354]  SIGNIFICANT
                                    crps_cal   -0.000770  CI [-0.001447, -0.000129]  SIGNIFICANT
                                    r_unc      +0.072938  CI [+0.039213, +0.106800]  SIGNIFICANT
                     n_draws=1      vort_rmse  -0.012146  CI [-0.014524, -0.009769]  SIGNIFICANT
                                    vort_corr  +0.167406  CI [+0.139635, +0.196567]  SIGNIFICANT
```

The superseded model is better than the shipped one on vorticity fidelity,
calibrated CRPS, uncertainty-error correlation, and interval width (10% narrower,
and it needs less conformal inflation: 2.909 against 3.140). RMSE is tied.

**Replicated on the independent set, with one exception.** Same two checkpoints
scored on `ocean_bench_v1b`:

| n_draws = 20, v1b | RMSE | vort RMSE | vort corr | CRPS cal | r(sigma, err) | factor | width |
|---|---|---|---|---|---|---|---|
| predecessor, no spread | 0.08805 | **0.01686** | **0.528** | 0.0348 | **0.258** | 2.503 | **0.2294** |
| shipped, spread on | **0.08719** | 0.01779 | 0.480 | 0.0351 | 0.209 | 2.637 | 0.2432 |

```
predecessor - shipped, v1b, n=20   rmse       +0.000857  CI [-0.000475, +0.002257]  TIED
                                   vort_rmse  -0.000937  CI [-0.001467, -0.000385]  SIGNIFICANT
                                   vort_corr  +0.047576  CI [+0.026253, +0.070871]  SIGNIFICANT
                                   crps_cal   -0.000350  CI [-0.000916, +0.000282]  TIED
                                   r_unc      +0.049144  CI [+0.017776, +0.080151]  SIGNIFICANT
                            n=1    vort_rmse  -0.011793  CI [-0.013968, -0.009716]  SIGNIFICANT
                                   vort_corr  +0.155107  CI [+0.123502, +0.187586]  SIGNIFICANT
```

What survives on both benchmarks: **vorticity RMSE, vorticity correlation, and
r(sigma, error)**, all significant, plus a consistently narrower interval and
lower conformal factor. What does **not** replicate is calibrated CRPS -- on v1 it
was -0.00077 and significant, on v1b -0.00035 with the interval crossing zero.
Same direction, weaker. Do not report the CRPS gap as an established result.
RMSE is tied on both, with the predecessor nominally worse each time.

→ `results_stream_predecessor_v1b.pt`

**Convergent validity.** This independent run lands almost exactly where the
matched 1000-step spread-OFF fine-tune did, despite a 0.26 mean relative weight
distance between the two models:

| | predecessor (78 ep) | 1000-step spread-OFF fine-tune |
|---|---|---|
| vort RMSE, n=20 | 0.01743 | 0.01731 |
| vort corr, n=20 | 0.562 | 0.560 |
| CRPS calibrated | 0.0368 | 0.0368 |
| r(sigma, error) | 0.296 | 0.289 |
| vort corr, n=1 | 0.486 | 0.501 |
| vorticity amplitude, n=1 | 1.167 | 1.137 |

Two completely different routes to "no spread term" reach the same place, and
both differ from the spread-ON state by the same +0.0436 in vorticity
correlation. That is what closes the "you are just relaxing away from a converged
optimum" objection: the spread-ON state is itself the fine-tuned one, and the
spread-OFF state reproduces a model trained 78 epochs without the term.

**Caveat, stated plainly:** the spread run also changed `path_steps` (120-200 →
90) and `lr` (2e-4 → 5e-5), so this historical comparison is not
single-variable. It corroborates the matched fine-tune ablation, which is, rather
than replacing it.

→ `results_stream_predecessor.pt`

### The weight sweep: no beneficial value exists

If the term were merely mis-weighted, some smaller `lambda_spr` should be net
positive. None is. Four arms, 1000 steps, differing only in `lambda_spr`,
n_draws = 20, `ocean_bench_v1`:

| lambda_spr | RMSE | vort RMSE | vort corr | CRPS cal | r(sigma, err) | factor | width |
|---|---|---|---|---|---|---|---|
| **0.0 (off)** | 0.09187 | **0.01731** | **0.560** | **0.0368** | **0.289** | 3.222 | **0.2351** |
| 0.1 | 0.09137 | 0.01733 | 0.554 | 0.0369 | 0.269 | 3.186 | 0.2389 |
| 0.3 | 0.09128 | 0.01768 | 0.541 | 0.0372 | 0.254 | 3.166 | 0.2452 |
| 1.0 (shipped) | **0.09110** | 0.01832 | 0.516 | 0.0377 | 0.225 | 3.151 | 0.2563 |

Monotone in `lambda_spr` on every column, and monotonically the wrong way on all
four metrics the term is supposed to serve. Against `lambda_spr = 1.0`, all three
lower weights are significant on vorticity correlation, calibrated CRPS and
r(sigma, error); RMSE is tied at every comparison.

The term's own objective degrades fastest of all: r(sigma, error) falls 0.289 →
0.225 as its weight rises 0 → 1. The training logs show the same thing from the
other side -- at `lambda_spr = 0.3` the in-training `spread_r` diagnostic *fell*
over the run, 0.795 → 0.701 → 0.653 across 1000 steps, i.e. the quantity being
optimised got worse while optimising it. Whatever the K=8 mid-schedule estimate of
model spread is measuring, its gradient does not point at the evaluated
uncertainty-error correlation.

So the term is not mis-weighted. There is no weight in [0, 1] at which it pays.

**The one column that favours it** is RMSE, which improves monotonically with
`lambda_spr` (0.09187 → 0.09110) though tied at every pairwise test. Read with
v1b's significant +0.00062 RMSE cost for turning it off, the honest reading is
that the term buys a few tenths of a percent of RMSE and pays for it in vorticity
fidelity, interval sharpness, calibrated CRPS and its own target metric.

→ `results_stream_spread_sweep.pt`

### Persistence: unchanged at 20x the training length, on different hardware

The controlled arms above are 1000 steps. Rerunning the matched pair at **20,000
steps on a CUDA box** -- different device, different RNG stream, therefore entirely
different batch orderings -- reproduces the effect essentially exactly:

| n_draws = 20 | RMSE | vort RMSE | vort corr | CRPS cal | r(sigma, err) | factor | width |
|---|---|---|---|---|---|---|---|
| 20k steps, spread OFF | 0.09215 | **0.01755** | **0.558** | **0.0368** | **0.288** | 3.124 | **0.2303** |
| 20k steps, spread ON | **0.09081** | 0.01846 | 0.511 | 0.0377 | 0.220 | 3.141 | 0.2583 |

```
20k OFF - 20k ON, n=20   rmse       +0.001338  CI [-0.000289, +0.002869]  TIED
                         vort_rmse  -0.000915  CI [-0.001535, -0.000247]  SIGNIFICANT
                         vort_corr  +0.047897  CI [+0.030982, +0.064888]  SIGNIFICANT
                         crps_cal   -0.000905  CI [-0.001552, -0.000289]  SIGNIFICANT
                         r_unc      +0.068086  CI [+0.029467, +0.107084]  SIGNIFICANT
                  n=1    vort_rmse  -0.013717  CI [-0.016112, -0.011293]  SIGNIFICANT
                         vort_corr  +0.190707  CI [+0.162918, +0.221080]  SIGNIFICANT
```

Side by side with the 1000-step MPS arms:

| effect (OFF - ON) | 1000 steps, MPS | 20,000 steps, CUDA |
|---|---|---|
| vort corr, n=20 | +0.0436 | +0.0479 |
| vort RMSE, n=20 | -0.00101 | -0.00092 |
| r(sigma, error) | +0.0637 | +0.0681 |
| vort corr, n=1 | +0.185 | +0.191 |

Nothing decays or reverses. Combined with the predecessor comparison, the result
now holds across three independent routes -- a short fine-tune, a long fine-tune on
other hardware, and two separately-trained historical models -- and across two
benchmarks.

→ `results_stream_long20k.pt`

### Robustness: not an EMA artefact

The fine-tune saves EMA weights (decay 0.999), so at 1000 steps the EMA recursion
still places 0.999^1000 = 0.37 weight on the original shipped parameters. That
could in principle understate the spread-OFF effect. It does not, because the
parameters barely move: the mean relative EMA-vs-raw weight distance is 0.0072
(spread ON) and 0.0110 (spread OFF), so the two endpoints are within ~1% of each
other. Re-scoring both arms on the **raw, non-EMA** weights reproduces the result:

| | EMA weights | raw weights |
|---|---|---|
| vorticity RMSE, OFF - ON | -0.001005 | -0.001017 |
| vorticity corr, OFF - ON | +0.043632 | +0.043175 |

Both significant under either choice. The 1000-step and 3000-step spread-OFF arms
also agree (correlation 0.501 both) despite very different EMA composition, which
is the same conclusion from a second direction.

→ `results_stream_spread_raw_weights.pt`

### The one cost

RMSE. Tied on v1 (-0.00084, CI crossing zero) but significantly worse on v1b
(+0.00062, CI [+0.00011, +0.00111]) -- about 0.7% relative, with an inconsistent
sign. Call it a small possible RMSE cost against a consistent gain everywhere
else.

→ `results_stream_spread_ablation.pt`, `results_stream_spread_calibration.pt`,
`results_stream_spread_replication.pt`, `results_stream_spread_vort_n20.pt`,
`results_stream_vort_timecourse.pt`

---

## 2b. The uncertainty map's spatial structure barely earns its place

The smoothing sigma exists to remove a grid-scale checkerboard mode from the ensemble
spread. It was **0.8** when this was written; sweeping it -- raw sigma fetched once per
model, the smoothing applied offline, each row conformally re-fitted (fit on 20 cases,
verify on 20) -- showed 0.8 is not optimal for either model, and raised a harder
question. **It is now 3.2**, with the factor refit to 3.165.

| model | sigma | cov raw | factor | cov cal | width | CRPS cal | r(sigma, err) |
|---|---|---|---|---|---|---|---|
| predecessor | 0.0 | 0.4860 | 3.026 | 0.8925 | 0.2391 | 0.0373 | 0.268 |
| predecessor | **0.8** (shipped) | 0.4892 | 2.908 | 0.8926 | 0.2298 | 0.0368 | 0.295 |
| predecessor | 3.2 | 0.4976 | 2.748 | 0.8897 | 0.2176 | 0.0364 | **0.304** |
| predecessor | 6.4 | 0.5080 | 2.629 | 0.8900 | 0.2094 | 0.0362 | **0.304** |
| predecessor | 12.8 | 0.5266 | 2.513 | 0.8879 | **0.2039** | **0.0362** | 0.282 |
| predecessor | constant | 0.5299 | 2.650 | 0.8956 | 0.2094 | 0.0367 | 0.042 |
| shipped | 0.0 | 0.5004 | 3.312 | 0.9041 | 0.2691 | 0.0382 | 0.204 |
| shipped | **0.8** (shipped) | 0.5027 | 3.139 | 0.9032 | 0.2551 | 0.0376 | 0.222 |
| shipped | 6.4 | 0.5183 | 2.831 | 0.9051 | 0.2311 | 0.0366 | 0.228 |
| shipped | 12.8 | 0.5333 | 2.632 | 0.9019 | 0.2171 | 0.0362 | 0.231 |
| shipped | **constant** | 0.5394 | 2.623 | 0.9021 | **0.2131** | **0.0364** | 0.053 |

`constant` replaces sigma with its own per-case ocean mean -- the smoothing limit,
carrying zero spatial information.

**Two results.**

*The shipped smoothing value is far from optimal.* Both models improve
monotonically in width and calibrated CRPS out to sigma = 12.8, and 0.8 costs the
shipped model ~16% interval width against its own best setting. Nothing here is a
tuning of the model; it is an inference-time constant.

*For the shipped model, a single scalar beats the per-cell map.* Constant sigma
gives narrower intervals (0.2131 vs 0.2551) and better calibrated CRPS (0.0364 vs
0.0376) at equivalent calibrated coverage (0.9021 vs 0.9032). The spatial
structure of the shipped model's uncertainty map is not merely uninformative for
sharpness -- it is worse than not having it.

The predecessor is different: heavy blurring beats constant on both width (0.2039
vs 0.2094) and CRPS (0.0362 vs 0.0367), and its r(sigma, error) peaks at **0.304**
against the shipped model's 0.231. So it retains genuine spatial content where the
shipped model has little. That is a third, independent way the spread term hurts.

**Caveat that limits the cross-model width comparison.** The predecessor's rows sit
at 0.888-0.896 calibrated coverage while the shipped model's sit at 0.902-0.905, so
its narrower widths partly reflect slightly-under coverage. The within-model
conclusions (0.8 is not optimal; constant beats per-cell for the shipped model) are
unaffected, since those compare rows at matched coverage. A clean cross-model width
comparison would re-fit both to hit exactly 0.90.

**The failure is Stream-specific -- checked.** A per-cell uncertainty map a planner
can route on is the project's deliverable, so the same treatment was applied to all
three models at their reported configurations:

| model | best blur: width / CRPS cal | constant sigma: width / CRPS cal | r(sigma, err) | verdict |
|---|---|---|---|---|
| CorrDiff (1 h, 20 draws) | **0.1446 / 0.0236** | 0.1465 / 0.0243 | **0.447** | per-cell wins |
| DistAttn (10 draws) | **0.1766 / 0.0288** | 0.1954 / 0.0298 | **0.414** | per-cell wins |
| Stream (20 draws) | 0.2171 / 0.0362 | **0.2131 / 0.0364** | 0.231 | constant wins |

CorrDiff's map beats a scalar on width and calibrated coverage *simultaneously*
(0.1446 at 0.8959 against 0.1465 at 0.8843), so that comparison needs no coverage
caveat. DistAttn wins at matched coverage (sigma 3.2: 0.1837 at 0.9176 against
0.1954 at 0.9202). Both correlate with realised error at roughly **double** Stream's
rate.

So per-cell calibrated uncertainty is a real property of the headline model, and
Stream's inability to beat a scalar is a fourth strike against it rather than a
problem with the premise. Worth noting alongside: CorrDiff's raw coverage is
0.834-0.875 with a conformal factor of 1.09-1.31, against Stream's 0.50 and
2.6-3.3 -- it genuinely arrives near-calibrated.

→ `results_stream_smooth_sigma.pt`, `results_uncertainty_spatial_value.pt`

## 3. What to do with this

**Recommendation: retrain the Stream direction network without the spread term
and re-fit its conformal factor.** The weight sweep rules out the softer option:
no `lambda_spr` in [0, 1] is net positive. On current evidence the term costs vorticity
fidelity, calibrated CRPS, interval sharpness, and its own target metric, and
buys only a wider raw spread that the conformal factor removes anyway.

**Done, 2026-09-02.** Rather than retrain, the library now ships the model's own
no-spread predecessor (`StreamFn_Cond_x0_mag.pt`, 78 ep) as
`assets/stream_dir_weights.pt`, and `STREAM_SIGMA_SCALE_TIMED` moved 3.141 → 2.909
(fitted on v1; applied blind to all of v1b it gives coverage 0.912 against the 0.90
target, versus the old weights' 0.917 with 8% wider intervals). The factor has since
moved again to **3.304**, because the sampler default changed from 6 steps to 2 and
the two are coupled, and then to **3.165** when `STREAM_UNC_SMOOTH_SIGMA` moved 0.8 →
3.2 — the factor is coupled to the smoothing sigma as well. 3.165 is the shipped value;
see `PAPER_NUMBERS.md` §7c and `DEFAULTS_AND_DIALS.md`. The swap is a clean
drop-in: same architecture, `cond_ch` 10, and the library reproduces the measured
per-case values exactly. Every Stream number in the docs was regenerated.

Caveats to state with the result:

- No arm is a from-scratch retrain. The predecessor comparison supplies a
  fully-trained no-spread model and the 20,000-step arms supply a 20x-longer
  fine-tune on other hardware; both agree closely with the 1000-step arms, which
  mitigates but does not formally remove this.
- `lambda_spr` was swept over {0, 0.1, 0.3, 1.0} at 1000 steps and is monotone;
  values above 1.0 were not tested, and the sweep was not replicated on v1b.
- The vorticity term arms had the spread term off in both control and treatment,
  so they are not the shipped configuration either -- but the contrast is clean,
  since both arms lost it equally.
- The two loss variants cannot be combined: the spread loss
  (`training_loss_streamfn_spread`) has no vorticity term, which is also why the
  shipped model's recorded `lambda_vort: 0.0` is an argparse default rather than
  a switched-off term.

## Reproducing

```bash
# arms (each ~7 min on an M-series GPU; the spread arm ~20 min at K=8)
python "Conditional DDPM/train_stream_vort.py" --init <shipped> --lambda_vort 0.0 --steps 3000 --out Models/stream_vort/lam00.pt
python "Conditional DDPM/train_stream_vort.py" --init <shipped> --lambda_vort 1.0 --steps 3000 --out Models/stream_vort/lam10.pt
python "Conditional DDPM/train_stream_vort.py" --init <shipped> --lambda_vort 5.0 --steps 3000 --out Models/stream_vort/lam50.pt
python "Conditional DDPM/testing/precompute_spread_targets.py" --pickle Datasets/pickles/data_divfree_chrono.pickle --split 0 --lags 13,25 --path_steps 90 --n_emp 80 --out "Conditional DDPM/spread_targets_train.npy"
python "Conditional DDPM/train_stream_vort.py" --init <shipped> --lambda_vort 0.0 --lambda_spread 1.0 --spread_targets "Conditional DDPM/spread_targets_train.npy" --steps 1000 --out Models/stream_vort/spread_s1000.pt

# evaluation
python benchmark/stream_vorticity.py
python benchmark/stream_spread_calibration.py
python benchmark/stream_spread_replication.py
```

Add `--verify_determinism 2` to any arm to print per-step losses and confirm the
arms see identical batches.

### From scratch, never having seen the spread term

The natural experiment above compares two real checkpoints, but they differ in more
than the term: the shipped model is epoch 126 of a lineage whose predecessor stopped at
epoch 78, and the spread run also changed `path_steps` (120-200 -> 90) and `lr`
(2e-4 -> 5e-5). So "no spread term" stayed confounded with "less training", and every
fine-tune ablation started from a model that had already absorbed the term.

This removes both confounds: a direction net trained from **random init** for 120,000
steps with `--lambda_vort 0 --lambda_angle 1 --lambda_mag 0.2 --min_snr_gamma 5`, no
spread term at any point, scored against the shipped net with the magnitude network,
sampler, seeds and masking held identical.

| arm | RMSE | angle | divRMSE | vortRMSE | vortCorr | CRPScal | factor | covcal | width |
|---|---|---|---|---|---|---|---|---|---|
| shipped (epoch 78, no spread) | 0.08651 | 0.86630 | 0.00946 | 0.01654 | 0.571 | 0.0344 | 3.162 | 0.894 | 0.2068 |
| from scratch, never saw it | 0.08714 | 0.87249 | 0.00946 | 0.01670 | 0.565 | 0.0345 | 3.125 | 0.892 | 0.2086 |

```
from scratch - shipped, paired over 40 cases:
   rmse       +0.000633  CI [-0.000328, +0.001735]  TIED
   angle      +0.006194  CI [-0.009359, +0.022279]  TIED
   div_rmse   -0.000008  CI [-0.000028, +0.000012]  TIED
   vort_rmse  +0.000157  CI [-0.000126, +0.000418]  TIED
   vort_corr  -0.006759  CI [-0.015674, +0.002208]  TIED
   crps_cal   +0.000144  CI [-0.000230, +0.000531]  TIED
```

**Tied on all six.** Training the direction net from scratch without the spread term
reproduces the shipped checkpoint, so the decision to drop the term does not rest on a
checkpoint that merely stopped early. The from-scratch net is nominally worse on five
of six, which is why it is *not* shipped -- there is no evidence it is better, and
swapping the asset would force a refit of the conformal factor and a recompute of every
Stream number for no measured gain.

Training loss fell from 3.797 at step 1 to 0.341, with the angle term 1.028 -> 0.081 and
the magnitude term 0.850 -> 0.005, so this is a converged run rather than an undertrained
one. -> `results_stream_scratch.pt`, `benchmark/stream_scratch_eval.py`
