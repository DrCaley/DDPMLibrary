# Data and Stream

## 1. Data

### The dataset

The raw dataset is **94 × 44 × 17,040** — a 94 × 44 grid of hourly (u, v) surface-current
frames, running from **01:00 on 2021-01-13** to **00:00 on 2022-12-24**. That is 709 days
23 hours; every frame is exactly 3600 s from the last, with no gaps.

| | |
|---|---|
| Source file | `ramhead_dataset.mat`, exported from `stjohn_hourly_5m_velocity_ramhead_v2.mat` |
| Location | Ram Head, St. John, U.S. Virgin Islands, 5 m depth |
| Grid | 94 × 44 cells at 50.7 m, covering 4.73 × 2.18 km |
| Extent | 18.2900–18.3096 N, 64.7248–64.6801 W |
| Cells | 4,136 total — 3,787 ocean, 349 land |
| Variables | `u`, `v` (94 × 44 × 17040), `lat`, `lon`, `ocean_time` |
| `ocean_time` | MATLAB datenum, in days. Time zone not recorded in the file. |

The file carries no model name, configuration, or DOI, so it cannot be cited yet — that
has to come from the provider.

### First split: `data.pickle`

Sliced into three separate arrays of shape (94, 44, 2, N):

| split | frames | share |
|---|---|---|
| train | 9,180 | 70% |
| val | 1,965 | 15% |
| test | 1,965 | 15% |

Those were carved into short, time-scattered blocks averaging ~21 frames — shorter than
the 25 h conditioning look-back, so ~80% of val/test targets had no real prior: the prior
fell across a block boundary into an unrelated time.

### Chronological rebuild: `data_raw_chrono.pickle`

One continuous array of shape **(17040, 2, 94, 44)**, with the splits stored as lists of
target frame indices instead of separate arrays. Every target's priors are therefore
always the genuine earlier field.

Frames are grouped into contiguous **336-hour (14-day) blocks** — 51 of them, the last
240 h — labelled by a repeating 7-block pattern:

```
train  train  train  train  train  val  test
```

Then dropped: the **first 25 frames** of the record (a target needs its 13 h and 25 h
priors), and **48 frames at each block boundary**.

| split | targets | blocks |
|---|---|---|
| train | 11,639 | 37 |
| val | 1,680 | 7 |
| test | 1,680 | 7 |
| dropped | 2,041 frames (12%) | — |

Index overlap between the three splits is zero.

The guard is 48 frames for two reasons: adjacent hourly frames correlate ≈0.95, so
touching blocks would put near-duplicates on both sides of a split; and 48 > 25, so no
target's 13/25 h prior can reach across a boundary into another split. Rotating blocks
rather than one chronological cut puts all three splits in both years and every tide
phase.

### Validation and test blocks

| # | validation | test |
|---|---|---|
| 1 | 2021-03-24 01:00 – 2021-04-07 00:00 | 2021-04-07 01:00 – 2021-04-21 00:00 |
| 2 | 2021-06-30 01:00 – 2021-07-14 00:00 | 2021-07-14 01:00 – 2021-07-28 00:00 |
| 3 | 2021-10-06 01:00 – 2021-10-20 00:00 | 2021-10-20 01:00 – 2021-11-03 00:00 |
| 4 | 2022-01-12 01:00 – 2022-01-26 00:00 | 2022-01-26 01:00 – 2022-02-09 00:00 |
| 5 | 2022-04-20 01:00 – 2022-05-04 00:00 | 2022-05-04 01:00 – 2022-05-18 00:00 |
| 6 | 2022-07-27 01:00 – 2022-08-10 00:00 | 2022-08-10 01:00 – 2022-08-24 00:00 |
| 7 | 2022-11-02 01:00 – 2022-11-16 00:00 | 2022-11-16 01:00 – 2022-11-30 00:00 |

The 48-frame guard is removed from each end, so usable targets sit inside these windows.

### Normalisation and priors

Mean is 0 and is not subtracted, so vector direction is preserved. Std is
**0.114603 m/s**, computed on train-split ocean cells only. Priors are not stored: the
13 h and 25 h priors for target `i` are `fields[i-13]` and `fields[i-25]` in the same
array.

Stream trains on `data_divfree_chrono.pickle` — the same array, Helmholtz-projected,
which removes 99% of the divergence and moves vorticity by 1%.

---

## 2. Stream

### Noise combinations on divergence-free data

Divergence-free fields are a linear subspace and the forward process is linear, so every
mixture stays in the subspace:

```
div(x0) = 0  and  div(eps) = 0
  =>  div( sqrt(a_t)*x0 + sqrt(1-a_t)*eps ) = 0   for every t
```

Nothing is projected inside the diffusion loop. Three pieces make it hold: the targets are
Helmholtz-projected, the noise is divergence-free by construction, and the network's output
is `curl(ψ)`, which is divergence-free identically. So the reverse trajectory is
divergence-free at every step, not just at the end.

The noise is built by drawing two independent Gaussian fields, FFT'ing both, projecting
each Fourier mode onto the direction orthogonal to `k`
(`u' = u - kx*(kx*u + ky*v)/|k|^2`, with `k = 0` left alone), inverse-FFT'ing, and dividing
by a single scalar std across both channels — a per-channel scale would break the property.

`kx` and `ky` are the **central-difference** symbols `sin(2πf)`, not the spectral symbols
`2πf`. The network's curl, the curl/divergence loss term, and the divergence metric all use
the `[-1,0,1]/2` stencil; projecting onto the spectral symbol leaves a central-difference
divergence of ≈0.4, the same order as the field itself.

### Training the direction network

It never predicts an angle. It predicts **one scalar stream function ψ** and outputs its
curl, so incompressibility is structural rather than penalised. Direction is scored by the
`1 − cos θ` loss term, separately from speed.

| | Direction (diffusion) | Magnitude (regression) |
|---|---|---|
| predicts | scalar ψ; output = `curl(ψ)` | speed μ and σ, per cell |
| parameters | 14.96 M | 13.93 M |
| training data | `data_divfree_chrono.pickle` | same, speeds |
| parameterisation | x₀ | — |
| T / schedule / noise | 1000 / cosine / divergence-free | — |
| epochs | 80 | 25 |
| lr / batch | 5e-5 / 6 | 2e-3 / 16 |
| EMA | 0.999 | none; backbone frozen |

### Combining angle and magnitude

`coupled_magnitude`, then `helmholtz_project`:

1. Draw 20 fields from the direction network.
2. Take each draw's **own** speed and z-score it across the 20 draws.
3. Rescale that z to the magnitude network's per-cell `μ(x) + σ(x)·z`.
4. Keep each draw's direction, replace its speed.
5. Reproject each field to divergence-free.

Step 2 is what makes it work: reusing each draw's own speed anomaly instead of fresh noise
keeps ensemble members spatially coherent rather than speckled.

### Loss functions

**Direction network** — three terms:

```
L = w_t*||x0_hat - x0||^2  +  1.0*(1 - cos θ)  +  0.2*(rms(x0_hat)/rms(x0) - 1)^2

w_t = min(SNR_t, 5) / mean_t[min(SNR_t, 5)]
```

| term | what it does |
|---|---|
| `w_t ||x0_hat - x0||^2` | squared error, weighted by min(SNR, 5) so easy timesteps stop soaking up the gradient |
| `1.0 (1 - cos θ)` | direction error, scored separately from speed |
| `0.2 (rms ratio - 1)^2` | penalises amplitude shrinkage — squared error rewards hedging toward the mean, which flattens the field |

A fourth term, `1.0 (1 - ρ(σ_model, σ_emp))`, was **dropped on 2026-09-02**. It
correlated the model's directional spread across draws against an empirical spread
map, and existed to make predicted uncertainty track actual error. Measured against
a matched control it did the reverse — r(σ, error) 0.223 with it, 0.296 without —
while also degrading vorticity fidelity and widening intervals ~10% at matched
coverage, for a tied RMSE.

A vorticity term exists in the code and was left switched off (λ = 0).

**Magnitude network:**

```
L = 0.5*( log(s^2) + (y - mu)^2 / s^2 )  +  0.05*TV(log s^2)
```

Gaussian NLL — confident-and-wrong is punished by the second half, uncertain-everywhere by
the first — plus a total-variation term that keeps the uncertainty map smooth rather than
speckled.

Full equations, weights and citations for all four losses: `docs/loss_doc/loss_functions.docx`.

### Reasonable argument for why we did this

The original argument was that incompressibility is the right prior for coherent eddies: a
hard constraint costs nothing at inference and cannot be violated.

The measurements contradict it for this variable. Surface currents are not
divergence-free — geostrophic flow is, ageostrophic flow is not, and Ekman transport
produces real convergence. The divergent part carries **14% of this field's energy**.
Because Helmholtz projection is orthogonal, no divergence-free field is closer to the truth
than the projection of the truth itself: **0.0613 RMSE** on held-out frames. A perfect
Stream would beat CorrDiff's 0.0665 by 8%. That is a provable ceiling, not a tuning result.
The magnitude network exists only to restore the speed information the constraint removed.

The argument that survives is comparative: Stream is the hard-constraint arm of a
controlled physics comparison — none (CorrDiff), soft (DistAttn's curl/divergence penalty),
hard (Stream). Hard physics caps you, soft physics does nothing measurable, and the
temporal priors are what win.
