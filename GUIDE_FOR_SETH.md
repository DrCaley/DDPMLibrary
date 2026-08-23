# DDPMLibrary — the models

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

## 2. DistAttn

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

## 3. RePaint

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

# Results

40 frames held out of every model's training data, identical observations
(90 cells per frame), scored on the common ocean mask (3,749 cells). CRPS and all
RMSE columns are in m/s.

### Accuracy

| model | priors? | CRPS ↓ | RMSE ↓ | RMSE @observed ↓ | RMSE @unobserved ↓ | angle error° ↓ |
|---|---|---|---|---|---|---|
| corrdiff | yes | **0.0256** | 0.0470 | 0.0059 | 0.0476 | 28.21 |
| repaint | yes | 0.0261 | **0.0466** | **0.0008** | **0.0472** | **27.93** |
| repaint_uncond | no | 0.0349 | 0.0639 | 0.0009 | 0.0646 | 35.02 |
| distattn | no | 0.0400 | 0.0665 | 0.0202 | 0.0672 | 42.93 |
| vcnn | no | 0.0446 | 0.0602 | 0.0076 | 0.0609 | 35.63 |
| stream | yes | *not yet measured* | | | | |
| ddpm | no | *not yet measured* | | | | |

### Uncertainty and structure

| model | spread–skill (→1.0) | coverage @90% (→0.90) | small-scale energy (→1.0) | skilful scale ↓ | eddy recall ↑ | SSIM ↑ | anomaly corr ↑ |
|---|---|---|---|---|---|---|---|
| corrdiff | **1.19** | **0.87** | 0.71 | 6.04 | 0.394 | 0.555 | **0.770** |
| repaint | 0.70 | 0.68 | **0.73** | **4.74** | **0.438** | **0.574** | 0.764 |
| repaint_uncond | 0.72 | 0.69 | 1.13 | 4.62 | 0.386 | 0.492 | 0.610 |
| distattn | 0.66 | 0.62 | 0.56 | 9.20 | 0.333 | 0.419 | 0.538 |
| vcnn | — | — | 0.63 | 6.40 | 0.381 | 0.461 | 0.600 |
| stream | *not yet measured* | | | | | | |
| ddpm | *not yet measured* | | | | | | |

VCNN and DDPM report no uncertainty, so spread–skill and coverage are undefined
for them. CRPS reduces to mean absolute error in that case, so every row is
still directly comparable.

### How to read this

**Compare within the priors groups, not across them.** The three models that get
the 13 h/25 h history have far more information than the four that don't.
Joseph's two models isolate the effect exactly — same architecture, same
training, priors removed: RMSE 0.0466 → 0.0639, i.e. **37% worse**. That gap is
bigger than any difference between architectures here.

Among the models that use **only observations**, VCNN is the most accurate
(0.0602 vs 0.0639 and 0.0665).

**Only CorrDiff's uncertainty is calibrated.** Its spread–skill of 1.19 and 87%
coverage are close to the ideal 1.0 and 90%; every other model sits at 0.66–0.72
and 62–69%, meaning their error bars are too narrow. For those models
`mean ± 1.645σ` is *not* a 90% interval.

**Two caveats worth stating.** The CorrDiff/RePaint CRPS gap is 0.0005 m/s on 40
frames with one seed — treat them as tied on accuracy; CorrDiff's real advantages
are calibration and being ~15× faster. And this benchmark gave every observation
the same timestamp, so DistAttn's age mechanism was inactive; its numbers here
are a floor, and it should do better on real staggered vehicle data.

---

## Reproducing this

```bash
python scripts/compare_models.py --pickle /path/to/data_raw_chrono.pickle \
    --frames-file scripts/fair_eval_frames.json \
    --n-frames 40 --models vcnn corrdiff repaint repaint_uncond distattn
```

`--frames-file` matters: the checkpoints come from two dataset pickles whose
train/test splits disagree, so either pickle's own test set is training data for
about half the models. That file lists the 2,460 frames held out by both.
