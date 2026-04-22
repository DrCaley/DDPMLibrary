# DDPMLibrary

Inference-only Python library for ocean-surface-velocity inpainting with a
denoising-diffusion model (DDPM). Drop the folder into your project, `import
ddpm_library`, and call `predict` on a list of sparse `(lat, lon, unix_time,
u, v)` observations to get back a dense velocity field over the St. John Rams
Head domain.

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

```bash
git clone https://github.com/DrCaley/DDPMLibrary.git
cd DDPMLibrary
pip install -r requirements.txt
```

Then either copy the `ddpm_library/` folder into your project, or add this
repo to your `PYTHONPATH`.

**Note on weights.** The bundled `ddpm_library/assets/weights.pt` is ~110 MB
and is tracked with Git LFS. Make sure you have `git-lfs` installed before
cloning, or the file will be a tiny pointer and loading will fail.

Requirements: Python ≥ 3.9, PyTorch ≥ 2.0, NumPy.

---

## API

### `DDPM(device="auto", weights_path=None)`

Loads the model once and keeps it resident. Reuse a single instance across
many calls for best performance.

| Argument | Default | Description |
| --- | --- | --- |
| `device` | `"auto"` | `"auto"` picks CUDA → MPS → CPU. Or pass `"cpu"`, `"cuda"`, `"mps"`, or a `torch.device`. |
| `weights_path` | `None` | Override the bundled `weights.pt` path. |

### `DDPM.predict(observations, *, t_start=250, resample_steps=3, seed=None) -> (mean, uncertainty)`

| Argument | Default | Description |
| --- | --- | --- |
| `observations` | *required* | Iterable of `(lat, lon, unix_time, u, v)` tuples. Co-located obs are averaged. |
| `t_start` | `250` | Reverse-diffusion starting step (1–250). Lower = faster but noisier. |
| `resample_steps` | `3` | RePaint resample count per step (higher = better blending with observations). |
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
├── ddpm_library/
│   ├── __init__.py           # exports DDPM, predict
│   ├── predict.py            # public API
│   ├── inference.py          # reverse-diffusion loop
│   ├── geo.py                # lat/lon ↔ grid-index
│   ├── rasterize.py          # obs list → sparse channels
│   ├── standardize.py        # z-score helpers
│   ├── config.py             # bounding box, grid, checkpoint metadata
│   ├── model/                # vendored UNet + schedule (no research-repo deps)
│   └── assets/
│       ├── weights.pt        # ~110 MB, tracked with git-lfs
│       └── lat_lon_grid.npz  # native grid coordinates
├── tests/
├── example.py
├── requirements.txt
└── README.md
```

---

## License

MIT. See `LICENSE`.
