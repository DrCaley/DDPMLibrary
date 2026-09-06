# Methods

> **Being regenerated (2026-09-05).** Every DistAttn and RePaint number below
> predates the fix for their per-draw seed aliasing, and every Stream number
> predates the `helmholtz_project` symbol fix. CorrDiff, VCNN and GP are
> unaffected. Do not quote the affected rows until this note is gone.

Data, models, training, and inference for CorrDiff, Stream, DistAttn and RePaint.
Every value is read from the dataset files, the `.mat` export, the trained
checkpoints, or `config.py`. None are script defaults.

## 1. Data

### Source

Hourly surface currents at 5 m depth off Ram Head, St. John, U.S. Virgin Islands,
delivered as `ramhead_dataset.mat` (exported from
`stjohn_hourly_5m_velocity_ramhead_v2.mat`).

| | |
|---|---|
| Arrays | `u`, `v` (94 x 44 x 17040), `lat`, `lon`, `ocean_time` |
| Grid | 94 x 44 cells at 50.7 m, covering 4.73 x 2.18 km |
| Extent | 18.2900-18.3096 N, 64.7248-64.6801 W |
| Cells | 4,136 total: 3,787 ocean, 349 land |
| Time range | 01:00 on 2021-01-13 to 00:00 on 2022-12-24 |
| Cadence | exactly 1 h, 17,040 frames, no gaps |
| Span | 709 days 23 hours |

`ocean_time` is a MATLAB datenum in days. The time zone is not recorded. The file
carries no model name, configuration or DOI, so the dataset cannot be cited yet.

### Files

| file | size | contents | `data_std` |
|---|---|---|---|
| `data.pickle` | 868 MB | three float64 arrays (94, 44, 2, N): 9,180 / 1,965 / 1,965 | n/a |
| `data_divfree.pickle` | 434 MB | same shapes, float32, Helmholtz-projected | n/a |
| `data_raw_chrono.pickle` | 564 MB | dict; `fields` (17040, 2, 94, 44) float32; splits 11,639 / 1,680 / 1,680 | 0.114603 |
| `data_divfree_chrono.pickle` | 564 MB | same, Helmholtz-projected | 0.106275 |
| `data_interp.pickle` | 434 MB | PCHIP-interpolated to 30 s (120 sub-steps per hour), built from `data.pickle` plus the `.mat`; splits stored as 132 / 131 / 131 contiguous segments | n/a |

`data_divfree.pickle` is not referenced by any shipped checkpoint and can be
deleted. `data.pickle` must stay: RePaint's unconditional variant was trained on it.

### Split scheme A: the original 70/15/15

Three separate arrays, 9,180 / 1,965 / 1,965 frames, carved into contiguous
time-scattered segments. Measured from `data_interp.pickle`, which preserves the
segment boundaries: train has 132 segments up to 70 frames long (mean 69.5), while
val and test have 131 segments each, every one exactly 15 frames.

Fifteen frames is shorter than the 25 h conditioning look-back, so the defect is
total rather than partial:

| split | frames | with a valid 25 h prior | with a valid 13 h prior |
|---|---|---|---|
| train | 9,180 | 5,895 (64.2%) | 7,467 (81.3%) |
| val | 1,965 | 0 (0.0%) | 262 (13.3%) |
| test | 1,965 | 0 (0.0%) | 262 (13.3%) |

Not one validation or test target in the original split has a real 25 h prior, and
only 13% have a real 13 h one. That is the defect the chronological rebuild fixes.
(The build script's docstring says "~21 frames" and "~80%"; both are wrong, and the
measured figures above supersede them.)

### Split scheme B: chronological

One continuous array, splits stored as lists of target frame indices rather than as
separate arrays, so a prior is always the genuine earlier field. Frames are grouped
into contiguous 336-hour (14-day) blocks, 51 of them with the last at 240 h,
labelled by a repeating seven-block pattern:

```
train  train  train  train  train  val  test
```

Two removals follow: the first 25 frames of the record, since every target needs
its 13 h and 25 h priors, and 48 frames at each block boundary.

| split | targets | blocks | share |
|---|---|---|---|
| train | 11,639 | 37 | 78% |
| val | 1,680 | 7 | 11% |
| test | 1,680 | 7 | 11% |
| dropped | 2,041 frames | | 12% of record |

Index overlap between the three is zero. The guard is 48 frames for two reasons:
adjacent hourly frames correlate about 0.95, so touching blocks would place
near-duplicates on both sides of a split; and 48 > 25, so no target's prior can
reach backwards across a boundary into another split.

Rotating blocks rather than one chronological cut puts all three splits in both
years and every tide phase:

| # | validation | test |
|---|---|---|
| 1 | 2021-03-24 01:00 - 2021-04-07 00:00 | 2021-04-07 01:00 - 2021-04-21 00:00 |
| 2 | 2021-06-30 01:00 - 2021-07-14 00:00 | 2021-07-14 01:00 - 2021-07-28 00:00 |
| 3 | 2021-10-06 01:00 - 2021-10-20 00:00 | 2021-10-20 01:00 - 2021-11-03 00:00 |
| 4 | 2022-01-12 01:00 - 2022-01-26 00:00 | 2022-01-26 01:00 - 2022-02-09 00:00 |
| 5 | 2022-04-20 01:00 - 2022-05-04 00:00 | 2022-05-04 01:00 - 2022-05-18 00:00 |
| 6 | 2022-07-27 01:00 - 2022-08-10 00:00 | 2022-08-10 01:00 - 2022-08-24 00:00 |
| 7 | 2022-11-02 01:00 - 2022-11-16 00:00 | 2022-11-16 01:00 - 2022-11-30 00:00 |

The 48-frame guard is removed from each end, so usable targets sit inside these
windows.

Normalisation is std-only: the mean is 0 and is not subtracted, which preserves
vector direction. Priors are not materialised; the 13 h and 25 h priors for target
`i` are `fields[i-13]` and `fields[i-25]` in the same array.

### Which model trains on what

| model | pickle | scheme | inputs |
|---|---|---|---|
| CorrDiff diffusion | `data_raw_chrono.pickle` | B | z-scored |
| CorrDiff V-CNN mean | not recorded in checkpoint | | z-scored per channel |
| Stream direction | `data_divfree_chrono.pickle` | B | z-scored |
| Stream magnitude | `data_divfree_chrono.pickle` | B | z-scored |
| DistAttn | `data_interp.pickle` | A | raw m/s |
| RePaint time-conditioned | `data_chrono_raw.pickle` (remote copy of `data_raw_chrono`) | B | raw m/s |
| RePaint unconditional | `data.pickle` | A | raw m/s |

Two notes on this table:

CorrDiff normalises raw data with `data_std = 0.10628`, which is
`data_divfree_chrono`'s value rather than `data_raw_chrono`'s 0.114603. This is not
a bug. `corrdiff_predict.py:103` reads the same constant back from the checkpoint,
so training and inference agree and it acts as a consistent global scale about 7%
off nominal.

The V-CNN's dataset is not recorded, but its per-channel normalisation
(-0.0691, -0.0321) and (0.1357, 0.0886) matches raw train ocean cells
(-0.0714, -0.0346) and (0.1362, 0.0887); the divergence-free equivalent is
(0.1245, 0.0840). It is trained on raw data.

### Fair evaluation pool

Because two split schemes are in use, either pickle's test set is training data for
roughly half the models. All evaluation therefore draws only from
`scripts/fair_eval_frames.json`: 2,460 frames held out of training by both schemes,
each at index >= 25 so the lags exist. Every benchmark number we report is on this
pool.

## 2. Benchmark

`benchmark/ocean_bench_v1.npz`, md5 `44b866540296490c615be13bacb4242e`.

| | |
|---|---|
| Cases | 40, drawn from the fair pool, seed 20260829 |
| Vehicle | Dubins-style track, 1.06 m/s, 102.1 m turn radius, 50.8 m cells |
| Collection | 2.00 h, 200 readings, each sampled from the field at the moment its cell was visited |
| Scored cells | 3,749, the intersection of every model's own ocean mask |
| Contents | `observations` (40, 200, 5), `truth` (40, 44, 94, 2), `priors` (40, 2, 44, 94, 2), plus mask, lat/lon, frame indices |

`ocean_bench_v1b.npz` (md5 `08f1a69fbbc6186b4bda90dfe2e79280`) is an independent
replication set: same protocol, seed 20260901, zero frame overlap with v1.

Metrics report both RMSE conventions. Vector magnitude, `sqrt(mean(du^2 + dv^2))`,
is used throughout the results; the per-component convention is smaller by exactly
`sqrt(2)`. Confusing the two cost the project a week with a collaborator, so state
which one is meant.

Uncertainty is calibrated by split conformal: fit the factor on 20 cases, verify
coverage on the held-out 20. CRPS is computed from (mean, sigma) under a Gaussian
assumption, identically for every model, rather than from raw ensembles whose
member counts differ.

## 3. CorrDiff

Deterministic mean plus residual diffusion, after Mardani et al.
(arXiv:2309.15214). A V-CNN predicts the field; the diffusion model learns only the
residual. No physics term anywhere in the loss.

| | |
|---|---|
| Parameters | 16.90 M: 14.97 M diffusion, 1.93 M V-CNN |
| Conditioning | 11 channels: 4 observation (u, v, mask, distance-to-path), 4 prior (lags 13, 25 h), 3 geometry. 15 total with the noised field. |
| Parameterisation | v-prediction |
| Schedule | cosine, T = 1000 |
| Training | 200 epochs, best at epoch 199; lr 2e-4, batch 16, EMA 0.999, base_ch 64, time_dim 256 |
| Sensor-noise dial | trained on sigma ~ U(0, 0.10) as a fraction of field std; `predict()` rejects values outside that range |
| Inference | 20 draws, 50 DDIM steps |
| Measured cost | 1.2 s per field on a Titan Xp |

Loss, one term:

```
L = min(SNR_t, 5)/(SNR_t + 1) * ||v_hat - v||^2_omega
v = sqrt(a_t)*eps - sqrt(1 - a_t)*x0
```

Verified across all git history: no CorrDiff trainer ever carried a structural term.

Calibration. The raw ensemble is under-dispersed, a known property of conditional
diffusion models that the original paper reports too. `CORRDIFF_SIGMA_SCALE = 1.6787`
was fitted for simultaneous observations; `CORRDIFF_SIGMA_SCALE_TIMED = 2.1801` for
2 h time-varying collection. Using the first on the second task gives intervals 2.3x
too narrow, which was a live correctness bug.

Sampling steps matter more for the spread than the mean. At 16 steps the mean RMSE
is within about 2% but the distribution degrades, so 50 stays.

## 4. Stream

Two networks. A diffusion model predicts direction as a scalar stream function; a
separate heteroscedastic U-Net predicts speed; the two are fused and reprojected.

Why the pieces fit together: divergence-free fields form a linear subspace and the
forward diffusion process is linear, so if the target and the noise are both
divergence-free, every mixture is too.

```
div(x0) = 0  and  div(eps) = 0
  =>  div( sqrt(a_t)*x0 + sqrt(1 - a_t)*eps ) = 0   for every t
```

Nothing is projected inside the loop. Three pieces make it hold: targets come from
`data_divfree_chrono.pickle`, the noise is divergence-free by construction, and the
network's output is `curl(psi)`, divergence-free identically. The reverse trajectory
is therefore divergence-free at every step, not only at the end.

### Divergence-free noise

Draw two independent Gaussian fields, FFT both, project each Fourier mode onto the
direction orthogonal to `k`:

```
u' = u - kx * (kx*u + ky*v) / |k|^2        (the k = 0 mode is left alone)
```

then inverse FFT and divide by a single scalar std across both channels. A
per-channel scale would break the property.

`kx` and `ky` are the central-difference symbols `sin(2*pi*f)`, not the spectral
symbols `2*pi*f`. The network's curl, the curl/divergence loss term, and the
divergence metric all use the `[-1,0,1]/2` stencil. Projecting onto the spectral
symbol leaves a central-difference divergence of about 0.4, the same order as the
field itself, which leaks into every sampled field.

### The two networks

| | direction (diffusion) | magnitude (regression) |
|---|---|---|
| predicts | scalar psi; output is `curl(psi)` | speed mu and sigma per cell |
| parameters | 14.96 M | 13.93 M |
| conditioning | 10 channels: 3 observation, 4 prior (lags 13, 25 h), 3 geometry | 10 channels |
| parameterisation | x0 | n/a |
| T / schedule / noise | 1000 / cosine / divergence-free | n/a |
| training | 300 epochs configured, best at epoch 78; lr 2e-4, batch 6, EMA 0.999 | 25 epochs configured, best at epoch 14; lr 2e-3, batch 16, backbone frozen |
| sampler | DPM-Solver++(2M), 2 steps, 20 draws | n/a |

The direction network never predicts an angle. Direction is scored by the
`1 - cos theta` loss term, separately from speed. Incompressibility is structural,
not penalised.

DPM-Solver++ replaced the DDPM ancestral sampler after a head-to-head: it wins on
every calibration and accuracy metric and is about 24x faster. The ancestral
sampler at 100 steps is kept for reproducing published numbers.

### Losses

Direction network, three terms:

```
L = w_t*||x0_hat - x0||^2_omega
  + 1.0*(1 - cos theta)_omega
  + 0.2*(rms(x0_hat)/rms(x0) - 1)^2

w_t = min(SNR_t, 5) / mean_t[min(SNR_t, 5)]
```

| term | purpose |
|---|---|
| Min-SNR weighted squared error | caps easy timesteps so they stop soaking up the gradient |
| `1 - cos theta` | direction error, scored separately from speed |
| rms ratio | penalises amplitude shrinkage; squared error rewards hedging toward the mean, which flattens the field |

**A fourth term was dropped on 2026-09-02.** `1.0*(1 - rho(sigma_model,
sigma_empirical))` correlated the model's directional spread across draws against
a precomputed empirical spread map, and was intended to make predicted uncertainty
track actual error. Measured against a matched control it did the opposite --
r(sigma, error) 0.223 with it against 0.296 without -- while also degrading
vorticity fidelity and widening intervals ~10% at matched coverage, for a
statistically tied RMSE. No weight in [0, 1] was net positive. The shipped
direction weights are now `StreamFn_Cond_x0_mag.pt`, the 78-epoch no-spread
predecessor of the previous 48-epoch spread-term checkpoint. See
`STREAM_LOSS_ABLATIONS.md`.

A vorticity term also exists in other stream loss variants in the codebase and was
not used. Tested separately: it transfers on Stream, unlike on CorrDiff, but moves
vorticity correlation only 0.560 to 0.570 across a 5x change in weight.

Magnitude network:

```
L = 0.5*( log(s^2) + (y - mu)^2 / s^2 ) + 0.05*TV(log s^2)
```

Gaussian NLL, so confident-and-wrong is punished by the second half and
uncertain-everywhere by the first, plus a total-variation term keeping the
uncertainty map smooth rather than speckled. The 0.05 is the script default; the
checkpoint does not record it.

### Fusing the two outputs

`coupled_magnitude`, then `helmholtz_project`:

1. Draw 20 fields from the direction network.
2. Take each draw's own speed and z-score it across the 20 draws.
3. Rescale that z to the magnitude network's per-cell `mu(x) + sigma(x)*z`.
4. Keep each draw's direction, replace its speed.
5. Reproject each field to divergence-free.

Step 2 is what makes it work. Reusing each draw's own speed anomaly rather than
injecting fresh noise keeps ensemble members spatially coherent instead of speckled.

### Two defaults that were wrong

`full_field=False` returned an exactly divergence-free output, costing 12.5% RMSE,
and `n_draws=1` returned a single noisy draw as the mean, costing 3.8%. Together
18.8%. Both were fixed on 2026-08-31; any Stream number from before that date
understates the model.

The ensemble size itself was swept on CorrDiff and the shipped 20 confirmed: 10 is
significantly worse on all five metrics, 40 buys nothing for twice the cost. The
conformal factor does not transfer across ensemble sizes, so changing `n_draws` now
warns. See `DEFAULTS_AND_DIALS.md`.

The uncertainty map also needs a nan-aware Gaussian smooth
(`STREAM_UNC_SMOOTH_SIGMA = 3.2`; it was 0.8 until it was swept on 2026-09-04, which
also refit the conformal factor to 3.165). Central differences have a Fourier symbol that
vanishes at Nyquist, so grid-scale checkerboard modes are unconstrained by the
divergence-free structure and appear in the ensemble spread. Smoothing removes them
and raises calibration correlations by about 0.025.

## 5. DistAttn

Sam's distance-aware attention base with Lin's time conditioning. Observations enter
as cross-attention tokens rather than as raster channels, penalised by physical
distance and by observation age.

| | |
|---|---|
| Parameters | 17.78 M |
| Observation token | 5 dims: `[x_norm, y_norm, u, v, age_norm]` |
| Attention | `attn = q k^T / sqrt(d) - alpha*dist - beta*age`, 4 heads |
| Parameterisation | eps-prediction |
| Schedule | linear, T = 1000 |
| Training | 150 epochs from a warm start on Sam's base, best at epoch 142; lr 2e-4, batch 16, 4,096 samples per epoch |
| Transects | 5 min to 3 h, so ages much past 3 h are out of distribution and `predict()` warns |
| Age scaling | `age_norm = (t_end - t_obs)/3600`, i.e. hours |
| Inputs | raw m/s, not z-scored |
| Inference | 10 draws, strided DDPM reverse chain, stride 10, so 100 network calls |
| Measured cost | 16.1 s per field |

It carries no temporal priors. It also uses its own stricter ocean mask, 3,749 cells
against the shared grid's 3,787; sampling zeroes land at every step, so substituting
the shared mask would diverge from how it was trained.

Loss, three terms:

```
L = ||eps_hat - eps||^2_omega
  + 0.002*|| Phi(x0_hat) - Phi(x0) ||_omega
  + 1.0 * sum_{i in P} (x0_hat - x0)^2 / (2*N_P)

Phi(x) = [ curl(x), div(x) ]
```

The second term is a physics check on the shape of the field, comparing curl and
divergence against the truth. The third forces agreement with the readings at the
cells the vehicle actually drove through, averaged over visited cells only;
averaging over the whole grid would dilute it to nothing, since the track touches
under 5% of the domain.

The observation term was designed to be weightable by reading age, but the shipped
checkpoint predates that option, so every reading was weighted equally. The
age-aware version was never trained.

Calibration factor fitted here: `DISTATTN_SIGMA_SCALE_TIMED = 1.3592`, refit on
2026-09-04 when the default ensemble size went 10 -> 20 (it was 1.621 at 10).

## 6. RePaint

Joseph's model. Despite the name it does not use RePaint's mask-and-resample
inpainting. It is an unconditional (or prior-conditioned) DDPM whose observations
are imposed at sampling time by DPS, diffusion posterior sampling: at each reverse
step the sample is nudged by the gradient of the observation likelihood.

Two variants share one architecture:

| | time-conditioned | unconditional |
|---|---|---|
| conditioning channels | 4: prev 13 h (u, v), prev 25 h (u, v) | 0 |
| trained on | `data_chrono_raw.pickle` (scheme B) | `data.pickle` (scheme A) |
| parameters | 14.96 M | 14.96 M |
| epoch / val loss | 120 / 2.13e-4 | 147 / 1.95e-4 |

Shared settings: linear schedule, T = 1000, base_ch 64, time_dim 256, 150 epochs,
raw m/s inputs (not z-scored), `curl_div_weight = 0.002` in the training loss.

Sampling: DPS with step size 0.04 over the full 1000-step chain, 10 draws. DPS
marginally beat MCG in the published numbers. Stride > 1 subsamples the chain for
speed. Measured cost 49.7 s per field, 41x CorrDiff.

## 7. Results

Frozen benchmark, 40 cases, each model at its own best configuration.

| model | RMSE | angle RMS (rad) | CRPS | raw coverage | own conformal factor | interval width |
|---|---|---|---|---|---|---|
| CorrDiff, 1 h cutoff | 0.0618 | 0.6842 | 0.0242 | 0.908 | 1.009 | 0.174 |
| DistAttn, full track | 0.0718 | 0.7782 | 0.0282 | 0.801 | 1.3592 | 0.164 |
| Stream, full field | 0.0865 | 0.8663 | 0.0389 | 0.435 | 3.162 | 0.207 |

All three pairwise RMSE gaps and all three CRPS gaps are significant under a paired
bootstrap over cases.

CorrDiff arrives calibrated: raw coverage 0.908 against a 0.90 target, and its
residual conformal factor is 1.009, meaning the shipped timed factor is already
right to within 1%. DistAttn's intervals need inflating 1.6x and Stream's 3.1x
before they are honest, and even then Stream's are 46% wider than CorrDiff's.
DistAttn calibrates to a respectable width; its problem is accuracy, not spread
shape.

Everything replicates on the disjoint v1b set with the cutoff and all calibration
factors fixed in advance:

| model | RMSE v1 -> v1b | CRPS | coverage with shipped factor applied blind |
|---|---|---|---|
| CorrDiff | 0.0618 -> 0.0566 | 0.0228 | 0.9171 |
| DistAttn | 0.0718 -> 0.0710 | 0.0274 | 0.8826 |
| Stream | 0.0865 -> 0.0834 | 0.0329 | 0.9101 |

Same ranking, same significance, and the cutoff-selected configuration got better on
unseen cases, which is the opposite of selection inflation.

RePaint ties CorrDiff on accuracy for the third time on a third independent set
(v1b: -0.0021, CI [-0.0079, +0.0028]), and ties on CRPS. What separates them is
calibration: granted its own conformal fit within v1b, RePaint covers 0.863, the
worst of the four, while CorrDiff covers 0.917 using a factor fitted on different
data. RePaint's spread has the wrong shape to calibrate well. Combined with 41x the
cost, that is the reason CorrDiff is the headline model and RePaint is related work.

## 8. Settled questions

Discarding observations older than 1 h improves CorrDiff by 8.2% RMSE and narrows
intervals 34% at matched coverage. It is a genuine interior optimum: 0.75 h and
1.25 h are both worse. Only models carrying an independent temporal prior benefit;
DistAttn gets 15.5% worse and GP 10.9% worse, so each model runs at its own optimum.
Controlled for spatial extent with matched-count controls.

A vorticity loss term on CorrDiff does nothing. Fine-tuned on byte-identical batches
against a lambda = 0 control, it is tied on every metric including vorticity RMSE
itself. Training loss improved 9% and did not transfer. Without the control this
would have been misreported as a significant eddy-recall win.

The divergence-free constraint has a provable floor. Helmholtz projection is
orthogonal, so no divergence-free field is closer to the truth than the projection of
the truth: 0.0613 RMSE on held-out frames, against CorrDiff's 0.0665. A perfect
Stream would win by 8%. The floor is 59% of Stream's RMSE but only 35% of its squared
error, so it is the ceiling, not the whole explanation of the gap.

The physics backs the diagnosis. Geostrophic flow is divergence-free, ageostrophic
flow is not, and Ekman transport produces genuine surface convergence. The divergent
part of this field carries 14% of its energy; Fablet et al. (2024, arXiv:2211.13059)
independently report the ageostrophic component at about 47% for satellite surface
currents, matching our 48% derivative ratio.

The Okubo-Weiss eddy metric is biased by divergence, and the obvious correction is
boundary-condition dependent. Adding a curl-free field cannot change vorticity but
drives the raw metric from 0.458 to 0. The projection-based correction gives opposite
model rankings under periodic and Neumann boundary conditions, so no eddy ranking is
reported. Report vorticity RMSE instead; it needs no decomposition. This applies
equally to the collaborator's eddy-IoU.

Model blending is closed. At each model's best configuration CorrDiff plus DistAttn
gains nothing and CorrDiff plus RePaint is borderline with an interval crossing zero.

## 9. What the three models are for

CorrDiff, no physics, learns everything and leans on temporal priors. DistAttn, soft
physics, a curl/divergence penalty at lambda = 0.002. Stream, hard physics, exactly
divergence-free by construction.

The finding is that hard physics caps you, soft physics does nothing measurable, and
temporal priors win. Stream's two-network complexity is then evidence about the
constraint rather than engineering to defend: the magnitude network exists only to
restore the speed information the constraint removed.

That last clause is now measured rather than asserted. Ablating only the fusion step
costs 16% RMSE, drops vorticity correlation 0.571 to 0.503, and leaves a raw ensemble
that needs an 11.5x conformal inflation to reach 90% coverage against 3.16x with the
network in place -- the speeds really do collapse, and by a measurable factor. It is
also the power check on the loss-term nulls: the same paired test returns three
significant effects here and TIED four times there. See `LOSS_TERM_ABLATIONS.md` 5.
