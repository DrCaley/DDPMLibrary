# DDPMLibrary

Inference-only Python library for ocean-surface-velocity inpainting with a
denoising-diffusion model (DDPM). Install it with `pip`, `import ddpm_library`,
and call `predict` on a list of sparse `(lat, lon, unix_time, u, v)`
observations to get back a dense velocity field over the St. John Rams Head
domain.

This library wraps the best "split-head" Helmholtz-decomposition model from
our research repo, trained on the small single-location (Rams Head) dataset
**without** bathymetry.

---

## Quick start

```python
from ddpm_library import DDPM

ddpm = DDPM(device="auto")          # "cpu", "cuda", "mps", or "auto"

observations = [
    # (lat, lon, unix_time_seconds, u_m_per_s, v_m_per_s)
    (18.295, -64.710, 1_700_000_000.0,  0.12, -0.04),
    (18.302, -64.690, 1_700_000_000.0, -0.08,  0.05),
    # ...
]

mean, uncertainty = ddpm.predict(observations)
# mean.shape        == (44, 94, 2)   # last axis: (u, v) in m/s
# uncertainty.shape == (44, 94, 2)   # currently all zeros (see Limitations)
```

Or use the stateless one-liner (lazy singleton, same API):

```python
from ddpm_library import predict
mean, uncertainty = predict(observations)
```

See `example.py` for a complete runnable example.

---

## Installation

Requirements: Python ≥ 3.9, PyTorch ≥ 2.0, NumPy, and **Git LFS** (the
weights file is ~110 MB and is stored via LFS).

### 1. Install Git LFS (one-time, per machine)

```bash
# macOS
brew install git-lfs
# Ubuntu / Debian
sudo apt install git-lfs
# Windows
# download the installer from https://git-lfs.com

git lfs install
```

If you skip this step, `assets/weights.pt` will clone as a
~130-byte pointer file and model loading will fail with an `unpickling`
error.

### 2. Clone the repo

```bash
git clone https://github.com/DrCaley/DDPMLibrary.git
cd DDPMLibrary
```

Verify the weights downloaded correctly:

```bash
ls -lh assets/weights.pt   # should be ~110M, not ~130B
```

If it looks like a pointer file, run `git lfs pull` to fetch the real bytes.

### 3. Install the package

```bash
pip install .
```

We recommend doing this in a fresh virtual environment (`python -m venv
.venv && source .venv/bin/activate`).

For editable development installs:

```bash
pip install -e ".[dev]"
```

### 4. Smoke-test the install

```bash
python scripts/example.py
```

You should see a device announcement (`cuda`, `mps`, or `cpu`) and a
printed `mean` shape of `(44, 94, 2)`. On CPU this takes ~1 minute with
`t_start=250`; on CUDA/MPS it's a few seconds.

### 5. Use it from your own project

Then just:
```python
from ddpm_library import DDPM
```

---

## API

### `DDPM(device="auto", weights_path=None)`

Loads the model once and keeps it resident. Reuse a single instance across
many calls for best performance.

| Argument | Default | Description |
| --- | --- | --- |
| `device` | `"auto"` | `"auto"` picks CUDA → MPS → CPU. Or pass `"cpu"`, `"cuda"`, `"mps"`, or a `torch.device`. |
| `weights_path` | `None` | Override the bundled `weights.pt` path. |

### `DDPM.predict(observations, *, single_step=True, t_start=249, resample_steps=3, seed=None) -> (mean, uncertainty)`

| Argument | Default | Description |
| --- | --- | --- |
| `observations` | *required* | Iterable of `(lat, lon, unix_time, u, v)` tuples. Co-located obs are averaged. |
| `single_step` | `True` | If `True`, run exactly one UNet forward pass and return the direct `x0` estimate (fast, ~40 ms on MPS/CUDA). If `False`, run the full iterative RePaint chain. |
| `t_start` | `249` | Single-step mode: the timestep the UNet is queried at (higher ⇒ more noise in the `x_t` input, model leans harder on observations). Iterative mode: reverse-chain starting step. |
| `resample_steps` | `3` | RePaint resample count per step (iterative mode only). |
| `seed` | `None` | If set, uses a deterministic noise RNG. |

Returns:
- `mean`: `np.ndarray` shape `(44, 94, 2)`, `float32`, `(u, v)` in m/s.
- `uncertainty`: `np.ndarray` shape `(44, 94, 2)`, `float32`. **Currently always zero** — see Limitations.

### `ddpm_library.predict(observations, **kwargs)`

Convenience module-level function backed by a lazily-created singleton
`DDPM(device="auto")`.

### `ddpm_library.geo.grid_arrays() -> (lat, lon)`

Returns the native 1-D lat (44,) and lon (94,) arrays for the model grid.
Useful if you want to place observations on exact grid nodes or convert
model output back to (lat, lon, u, v) tuples.

---

## Coverage area

The model operates on a fixed 44 × 94 grid over the **St. John Rams Head**
region:

- latitude:  `[18.290007, 18.309564]`
- longitude: `[-64.724760, -64.680048]`

Passing an observation outside this bounding box raises `ValueError`. There is
no extrapolation — this model has only ever seen this specific piece of
ocean.

---

## Usage patterns

### Reading back (lat, lon, u, v) tuples

The output is an array indexed `[i_lat, j_lon, component]`. To convert it
back to a flat list of tuples:

```python
import numpy as np
from ddpm_library import DDPM
from ddpm_library.geo import grid_arrays

lat, lon = grid_arrays()
mean, _ = DDPM(device="auto").predict(observations)

predictions = [
    (float(lat[i]), float(lon[j]), float(mean[i, j, 0]), float(mean[i, j, 1]))
    for i in range(mean.shape[0])
    for j in range(mean.shape[1])
]
```

### Snapping arbitrary points to the grid

If you want to feed the planner values at specific lat/lon queries, sample
them from `mean` with a nearest-neighbour lookup:

```python
from ddpm_library.geo import lat_lon_to_index

i, j = lat_lon_to_index(18.301, -64.705)   # raises ValueError if out of bounds
u, v = mean[i, j]
```

### Reusing one model across many calls

Model construction loads ~110 MB of weights and allocates GPU memory.
**Create one `DDPM` instance and reuse it** — don't re-instantiate per call:

```python
ddpm = DDPM(device="auto")             # load once
for planning_step in range(100):
    mean, _ = ddpm.predict(obs_so_far) # fast
```

The module-level `predict()` helper does this for you automatically via a
lazy singleton, but you give up explicit control of the device.

### Speed vs. quality trade-off

**Default is `single_step=True`**: one UNet forward pass, returns the
direct x0 estimate. On MPS/CUDA this is ~40 ms per call. Observations are
preserved exactly at their cells; everything else is filled in by the
model.

If you want the full iterative RePaint chain (slower, can produce
smoother fields in some regimes), pass `single_step=False`:

| Mode | Speed (MPS) | Notes |
| --- | --- | --- |
| `single_step=True` *(default)* | ~40 ms | Best for planners — one network call. |
| `single_step=False, t_start=249, resample_steps=3` | ~30 s | Full RePaint quality. |
| `single_step=False, t_start=100, resample_steps=1` | ~5 s | Reasonable middle ground. |

### Reproducibility

Pass `seed=<int>` to make a call deterministic:

```python
m1, _ = ddpm.predict(obs, seed=42)
m2, _ = ddpm.predict(obs, seed=42)
# m1 and m2 are bit-identical on the same device.
```

---

## Limitations

1. **No time awareness.** `unix_time` is accepted and ignored. The model was
   trained as a time-invariant prior over the current patterns in this
   region; different seasons / tidal states all collapse to the same prior.
2. **No uncertainty.** `uncertainty` is returned as zeros. A proper
   ensemble-based uncertainty estimate would require running `predict`
   multiple times with different seeds and taking the per-cell standard
   deviation; we may add this as an opt-in flag later.
3. **Fixed domain.** The bounding box and grid are baked in. Out-of-bounds
   observations throw.
4. **No bathymetry channel.** This library uses the model variant trained
   without bathymetry, so you do not need to supply any topographic data.
5. **Inference only.** Training lives in the research repo.

---

## Repository layout

```
DDPMLibrary/
├── assets/
│   ├── weights.pt            # ~110 MB, tracked with git-lfs
│   └── lat_lon_grid.npz      # native grid coordinates
├── src/
│   └── ddpm_library/
│       ├── __init__.py       # exports DDPM, predict
│       ├── predict.py        # public API
│       ├── inference.py      # reverse-diffusion loop
│       ├── geo.py            # lat/lon ↔ grid-index
│       ├── rasterize.py      # obs list → sparse channels
│       ├── standardize.py    # z-score helpers
│       ├── config.py         # bounding box, grid, checkpoint metadata
│       └── model/            # vendored UNet + schedule (no research-repo deps)
├── tests/
├── scripts/
│   └── example.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## License

MIT. See `LICENSE`.

---

## Troubleshooting

**`UnpicklingError` / `RuntimeError: PytorchStreamReader failed`** when
constructing `DDPM()`.
→ `weights.pt` is still a Git LFS pointer. Run `git lfs install && git lfs
pull` in the repo root, then `ls -lh assets/weights.pt` — the
file should be ~110 MB.

**`ValueError: Observation (lat, lon) is outside the model's covered region`**
→ One of your input tuples has a lat/lon outside the bounding box printed
earlier in this README. The model has no coverage outside Rams Head; clip
or drop those observations before calling `predict`.

**`RuntimeError: MPS backend out of memory`** on Apple Silicon.
→ Fall back with `DDPM(device="cpu")`. The CPU path is slower but fully
supported.

**Predictions look different every call.**
→ Expected — each call draws fresh diffusion noise. Pass `seed=<int>` for
determinism.

**Predictions ignore my observations entirely.**
→ Check that your observations' `u`/`v` are in **m/s** and that their
lat/lon snap to distinct grid cells (`lat_lon_to_index` tells you the
index). Two observations at the same grid cell get averaged.

---

## Changelog

- **0.2.0** — Single-step inference is now the default (`single_step=True`).
  One UNet forward pass instead of the full 250-step RePaint chain —
  ~800× faster (~40 ms on MPS/CUDA). Pass `single_step=False` to get the
  old iterative behaviour.
- **0.1.0** — Initial release. Split-head Helmholtz multi-res model, no-
  bathymetry Rams Head checkpoint. Uncertainty stub.
