# DDPMLibrary — the models

> Which model is whose: **CorrDiff, StreamDDPM, VCNN, DDPM** (Henry) · **RePaint, RePaintUncond** (Joseph) · **DistAttn** (Sam's distance attention, Lin's time conditioning) · **GP** (classical baseline).

Every model returns the same thing, so you can swap between them without
changing your code:

```python
mean, uncertainty = Model(device="auto").predict(observations)     # or (observations, priors)
```

- `observations` — list of `(lat, lon, unix_time, u, v)`, u/v in m/s
- `mean`, `uncertainty` — both `(44, 94, 2)` float32 in m/s, indexed
  `[i_lat, j_lon, component]`, component 0 = u, 1 = v. Land is 0.
- `priors` — for the three models that need it, a list of two `(44, 94, 2)`
  fields: the ocean state 13 h and 25 h before the target time.

All weights live in **`src/ddpm_library/assets/`**. Run `git lfs pull` after
cloning or they arrive as 130-byte stubs and loading fails.

---

## 1. CorrDiff

A fast CNN makes a first guess at the field, then a diffusion model paints in the
correction on top of it. Running that correction many times gives an ensemble
whose spread is a trustworthy error bar — it is the only model here whose
uncertainty is calibrated.

```python
from ddpm_library import CorrDiff

model = CorrDiff(device="auto")
mean, uncertainty = model.predict(observations, priors, n_draws=20)
mean, uncertainty = model.predict(observations, priors, sensor_noise=0.05)  # 5% sensor error
```

**Weights:** `assets/corrdiff_weights.pt` (114 MB), `assets/corrdiff_grid.npz`,
and `assets/vcnn_weights.pt` (7.4 MB — used as the first guess).

---

## 2. DistAttn — Sam's attention model with Lin's time conditioning

*aka the time-conditioned DDPM / distance-aware time-conditioning model.*
*Sam wrote the distance-attention base; Lin added the time awareness.*

Each observation is fed in as a separate item the network can attend to, tagged
with where it was taken and **how long ago**. It is the only model that reads
your timestamps, so a track collected over two hours is treated as such rather
than as an instant snapshot.

```python
from ddpm_library import DistAttn

model = DistAttn(device="auto")
mean, uncertainty = model.predict(observations, n_draws=10)
```

No priors needed. Pass your real observation times — only the gaps between them
matter, not the absolute clock. Trained on runs of 5 min to 3 h.

**Weights:** `assets/distattn_weights.pt` (204 MB),
`assets/distattn_ocean_mask.npy`.

---

## 3. RePaint — Joseph's time-conditioned model

*conditioned on 13 h/25 h PRIOR FIELDS, not on observation age — different sense of "time-conditioned" from DistAttn above*

A diffusion model that has learned what ocean fields look like in general, then
nudged during generation so its output passes through your measurements. It also
sees the field 13 h and 25 h earlier.

```python
from ddpm_library import RePaint

model = RePaint(device="auto")
mean, uncertainty = model.predict(observations, priors, n_draws=10)
mean, uncertainty = model.predict(observations, priors, n_draws=10, stride=10)  # ~10x faster
```

Slow: about 4 minutes per field on a GPU at the default settings.

**Weights:** `assets/repaint_timecond_weights.pt` (171 MB).

---

## 4. RePaintUncond

The same model and the same nudging as above, but trained without any history, so
your measurements are all it has to go on. Useful when you don't have the earlier
fields.

```python
from ddpm_library import RePaintUncond

mean, uncertainty = RePaintUncond(device="auto").predict(observations, n_draws=10)
```

**Weights:** `assets/repaint_uncond_weights.pt` (171 MB).

---

## 5. StreamDDPM

Instead of predicting the currents directly, it predicts a surface whose slopes
*are* the currents, which forces the result to conserve water automatically. A
second network predicts the current speed and the two are combined.

```python
from ddpm_library import StreamDDPM

mean, uncertainty = StreamDDPM(device="auto").predict(observations, priors, n_draws=20)
```

**Weights:** `assets/stream_dir_weights.pt` (114 MB),
`assets/stream_mag_weights.pt` (53 MB), `assets/stream_grid.npz`.

---

## 6. VCNN

Fills the space between your measurements by giving every grid cell the value of
its nearest observation, then cleans that up with a single pass through a CNN.
No diffusion, essentially instant, and a strong baseline.

```python
from ddpm_library import VCNN

mean, _ = VCNN(device="auto").predict(observations)      # uncertainty is zeros
```

**Weights:** `assets/vcnn_weights.pt` (7.4 MB).

---

## 7. DDPM

The original diffusion model for this project. By default it takes a single
shortcut step (~40 ms) instead of the full slow generation loop.

```python
from ddpm_library import DDPM

mean, _ = DDPM(device="auto").predict(observations)      # uncertainty is zeros
mean, _ = DDPM(device="auto").predict(observations, single_step=False)   # full chain
```

**Weights:** `assets/weights.pt` (110 MB).

---

## 8. GP — the classical baseline

Not anyone's model: standard Gaussian-process regression (Matern kriging), the
textbook way to reconstruct a field from scattered measurements. It fits a
smooth surface through the observations and reports how uncertain it is as you
move away from them, with u and v fitted independently.

It is here for two reasons. It is the method a reader will ask why we didn't
just use, and it is the honest reference for any *calibration* claim: a GP
produces its error bars from the model itself, with nothing fitted afterwards to
make the coverage come out right.

```python
from ddpm_library import GP

mean, uncertainty = GP().predict(observations)      # no priors, no device needed
```

**Weights:** none — there is nothing to download. It fits its own
hyperparameters to your observations on every call, runs on CPU (scikit-learn),
and takes well under a second. `device` is accepted and ignored.

Cost grows as the cube of the *number of observations*, so a track of a few
thousand points would need a sparse approximation; at the ~100 observations used
here that is irrelevant.

### How it works

One assumption, and nothing else: nearby places have similar currents. It is told
nothing about oceans.

To predict a cell you did not measure, it takes a **weighted average of the cells
you did** — nearby readings count for more. Close to a measurement that
measurement dominates, so the reconstruction passes through your observations;
far from everything the weights all shrink and the answer drifts back toward the
average of your track. Each of the ~3800 output cells gets its own set of weights,
computed from its distances to the ~90 observations.

The important thing to be clear about: **there is no training set.** The weighted
average is over the observations of the single field being reconstructed. The GP
never sees another ocean field. That is exactly why it is the baseline worth
reporting — it is the score you get with no learned prior at all, so everything
the diffusion models earn above it is attributable to having learned what these
fields look like.

The weights are cleverer than plain distance-weighting in two ways. Illustration
from a 1-D version of the same kernel, nine observations clustered on the left
and a wide unmeasured gap on the right (indicative, not output of this module):

```
query near the observations          query out in the gap
  distance   weight                    distance   weight
     0.03    +0.779                       0.23    +0.068
     0.04    +0.406                       0.23    +0.063
     0.09    -0.214                       0.30    -0.043
     0.15    +0.080                       0.36    +0.019
     0.20    -0.032                       0.42    -0.008
  weights sum to 1.03                  weights sum to 0.07
```

**Weights go negative**, so a cluster of nearby readings does not get its vote
counted several times over and neighbours can oppose each other to sharpen a
gradient — which is why kriging does not oversmooth the way inverse-distance
averaging does. And **the weights sum to about 1 where you have data and to about
0 where you do not**: in the gap the prediction is 7% your measurements and 93%
"revert to the mean", which is the model correctly reporting that it has nothing
to go on.

Two numbers are fitted per call, by asking which pair makes your own observations
least surprising: how far "nearby" reaches (the length scale) and how much to
trust an individual reading (the noise level). That is the entire fit, it takes
milliseconds, and it is why there is no checkpoint.

### Why its uncertainty is unreliable

The error bars come out of the same algebra, with no calibration fitted
afterwards, which is what makes GP the honest reference for a calibration claim.
But read them with care:

**The variance depends on where you measured, not on what you measured.** Move
the vehicle along the same track through a violent eddy field and through uniform
flow and the GP returns *identical* error bars. In fact the whole weight matrix
can be computed before a single current value is read off the vehicle. So an eddy
sitting in water the vehicle drove around is invisible twice over: not predicted,
and not admitted to as a possibility.

(Strictly, that is exact for fixed hyperparameters. Ours are refitted per call
and `normalize_y=True` rescales by the observations' spread, so the values do
reach the variance — but only through those two scalars, never through *where*
the structure actually is.)

That is the mechanism behind its coverage of **0.395 against a 0.90 target** in
the results table below — the worst in the library. Do not let "it is a GP so its
sigma is principled" stand unqualified; principled here does not mean calibrated.

Two smaller caveats worth a sentence if we report GP as the baseline:

- **u and v are fitted as two independent problems**, so nothing couples them and
  the result is not required to conserve water the way the real currents are.
- Both grid axes are normalised to [0, 1] while the grid is 94x44, so a single
  isotropic length scale spans about 2.1x more physical distance along longitude
  than along latitude. Inherited from the research implementation and kept so the
  numbers match the published run; flagged in a comment in `gp_predict.py`.

---

# Results

All 8 models, 40 frames held out of **every** model's training data, identical
observations (90-cell track), scored on the common ocean mask (3749 cells).
CRPS and RMSE in m/s.

| model | priors? | CRPS ↓ | RMSE ↓ | angle° ↓ | spread–skill (→1) | coverage@90 (→0.90) |
|---|---|---|---|---|---|---|
| **corrdiff** | yes | **0.0256** | 0.0477 | **28.0** | **1.19** | **0.878** |
| repaint | yes | 0.0261 | **0.0466** | 27.9 | 0.70 | 0.676 |
| repaint_uncond | no | 0.0349 | 0.0639 | 35.0 | 0.72 | 0.693 |
| distattn | no | 0.0401 | 0.0666 | 42.9 | 0.62 | 0.609 |
| vcnn | no | 0.0446 | 0.0602 | 35.6 | — | — |
| stream | yes | 0.0451 | 0.0723 | 42.4 | 0.50 | 0.531 |
| gp | no | 0.0629 | 0.0950 | 48.5 | 0.30 | 0.395 |
| ddpm | no | 0.0701 | 0.0899 | 54.4 | — | — |

VCNN and DDPM report no uncertainty, so spread–skill and coverage are undefined.
CRPS reduces to mean absolute error in that case, so every row is still directly
comparable.

### How to read this

**Compare within the priors groups.** The models given the 13 h/25 h history have
far more information than the rest. Joseph's two models isolate the effect
exactly — same architecture, same training, priors removed: RMSE 0.0466 → 0.0639,
**37% worse**. That gap is larger than any difference between architectures here.
Among the observations-only models, **VCNN has the best RMSE** (0.0602 vs 0.0639-0.0950) but **DistAttn has the better CRPS** (0.0401 vs 0.0446), because CRPS credits DistAttn's real uncertainty and VCNN reports none. Which one is 'best' depends on whether you need error bars.

**corrdiff and repaint are TIED on accuracy.** Measured directly on 40 frames
with a per-frame bootstrap: CRPS difference −0.0023, 95% CI [−0.0056, +0.0010];
RMSE, angle error and SSIM likewise all cross zero. The one significant
difference goes the other way — **RePaint has better eddy recall** (0.487 vs
0.442, CI [−0.085, −0.006]). Do not claim CorrDiff is more accurate.

**CorrDiff's real advantages are calibration and speed.** It is the only model
whose error bars mean what they say (spread–skill 1.19, coverage 0.878 against a
0.90 target); everything else is over-confident, including the Gaussian process,
which is right 39% of the time while claiming 90%. And it runs in ~15 s per field
versus RePaint's ~4 min.

**All of this assumes observations are simultaneous, and they are not.** A real
vehicle takes over an hour to collect a 90-cell track, which costs every model
about 20% CRPS and drops CorrDiff's coverage to 0.80. If your measurements span
time, pass `sigma_scale=CORRDIFF_SIGMA_SCALE_TIMED` to restore calibration. See
`docs/STALENESS_FINDINGS.md`.

## Reproducing this

```bash
python scripts/compare_models.py --pickle /path/to/data_raw_chrono.pickle \
    --frames-file scripts/fair_eval_frames.json \
    --n-frames 40 --models vcnn corrdiff repaint repaint_uncond distattn
```

`--frames-file` matters: the checkpoints come from two dataset pickles whose
train/test splits disagree, so either pickle's own test set is training data for
about half the models. That file lists the 2,460 frames held out by both.
