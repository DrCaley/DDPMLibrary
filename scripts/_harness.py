"""Shared pieces for the evaluation scripts in this directory.

Every script here needs the same four things: load the chronological dataset,
choose frames no model was trained on, lay a vehicle track over cells every model
calls ocean, and put a confidence interval on a paired difference. Those were
copy-pasted across several scripts; this is the single copy.

Not part of the installed package -- ``pyproject.toml`` ships ``src/`` only. These
are harness utilities, not inference code.

IMPORTANT: :func:`make_track` consumes the passed RNG in a fixed order, and every
published number in this project depends on that order. Changing it silently
re-rolls every track and invalidates past results. ``tests/test_harness.py`` pins
the behaviour with a fingerprint.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

# The collaborator's measured rate for the real vehicle. One walk step advances
# one cell, so a step costs 30 / 0.628 seconds.
ROBOT_CELLS_PER_30S = 0.628
SEC_PER_CELL = 30.0 / ROBOT_CELLS_PER_30S      # ~47.8 s

#: Temporal-prior lags used by every prior-conditioned model here, in hours.
LAGS = (13, 25)


def load_fields(pickle_path: str | Path) -> tuple[np.ndarray, int]:
    """Return ``(fields, block_size)`` with fields as (T, 44, 94, 2) float32 m/s.

    Accepts the chronological format (a dict with a (T, 2, 94, 44) ``fields``
    array), a dict of named splits, or a bare (T, H, W, 2) array. ``block_size``
    is the length of a contiguous hourly run; frames whose lookback would cross a
    block boundary must be skipped, since neighbouring blocks are unrelated in
    time.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "fields" in data:
        arr = np.transpose(np.nan_to_num(np.asarray(data["fields"], np.float32)),
                           (0, 3, 2, 1))                  # (T,2,94,44) -> (T,44,94,2)
        block = int(data.get("meta", {}).get("block_size") or 10 ** 9)
        return arr, block
    arr = np.asarray(data["test"] if "test" in data else data, np.float32)
    if arr.ndim != 4 or arr.shape[3] != 2:
        raise ValueError(f"expected (T, H, W, 2); got {arr.shape}")
    if arr.shape[1:3] == (94, 44):
        arr = np.transpose(arr, (0, 2, 1, 3))
    return np.nan_to_num(arr), 10 ** 9


def frame_pool(frames_file: str | Path | None, n_frames: int, arr_len: int,
               rng: np.random.Generator, lo: int | None = None) -> list[int]:
    """Choose ``n_frames`` target frames, sorted.

    ``frames_file`` is a JSON file with a ``frames`` list -- pass
    ``fair_eval_frames.json`` to restrict to frames held out of training by BOTH
    dataset splits in use across the group's checkpoints. Without it, either
    pickle's own test set is training data for roughly half the models.
    """
    lo = max(LAGS) if lo is None else lo
    if frames_file:
        spec = json.load(open(frames_file))
        if "ranges" in spec:            # compact form: inclusive [start, end] pairs
            pool = np.concatenate([np.arange(a, b + 1) for a, b in spec["ranges"]])
        else:                           # legacy flat list
            pool = np.asarray(spec["frames"], dtype=int)
    else:
        pool = np.arange(lo, arr_len)
    pool = pool[(pool >= lo) & (pool < arr_len)]
    if pool.size == 0:
        raise ValueError(f"no frame satisfies {lo} <= index < {arr_len}")
    return sorted(int(x) for x in rng.choice(
        pool, size=min(n_frames, pool.size), replace=False))


def make_track(ocean_mask: np.ndarray, n_obs: int,
               rng: np.random.Generator, return_steps: bool = False,
               straight_bias: float = 0.0):
    """A contiguous vehicle track of ``n_obs`` distinct cells, in VISIT ORDER.

    Confined to ``ocean_mask`` -- pass the mask every model agrees is ocean.
    Without that the walk can place observations on land, and each model then
    drops them against its own mask, so the model with the strictest mask
    silently gets fewer observations than the others.

    With ``return_steps`` also returns, per returned cell, the walk STEP at which
    it was first reached.

    AGE ACCOUNTING -- read before timing anything with this. The returned list is
    the sequence of FIRST visits, but the walk keeps moving over cells it has
    already seen, so cell ``k`` was generally reached at step ``s_k > k``. This
    walk needs ~199 steps to collect 90 distinct cells, so real elapsed time is
    about 2.2x ``len(cells) * SEC_PER_CELL``. Deriving ages from list position
    (what :func:`track_ages` does) therefore UNDERSTATES staleness for this walk.
    Use the returned steps when the timing has to be right. See
    ``docs/STALENESS_FINDINGS.md``.

    ``straight_bias`` > 0 gives the walk directional persistence, matching the
    generator the models were TRAINED with (0.75). That produces a transect-like
    path with few revisits, so positional ages are nearly correct and the eval
    track resembles the training track. ``0.0`` (the default) is the unbiased
    walk the published numbers used -- kept so they reproduce.
    """
    ok = np.asarray(ocean_mask, bool)
    valid = np.argwhere(ok)
    if len(valid) < n_obs:
        raise ValueError(f"only {len(valid)} usable cells for {n_obs} observations")
    r, c = (int(x) for x in valid[rng.integers(len(valid))])
    H, W = ok.shape
    cur = _DIRS[rng.integers(4)] if straight_bias > 0.0 else (0, 0)
    cells: list[tuple[int, int]] = []
    steps: list[int] = []
    seen: set[tuple[int, int]] = set()
    stuck = step = 0
    while len(cells) < n_obs:
        if (r, c) not in seen:
            seen.add((r, c)); cells.append((r, c)); steps.append(step); stuck = 0
        else:
            stuck += 1
        if stuck > 200:                 # walk boxed in; restart elsewhere
            r, c = (int(x) for x in valid[rng.integers(len(valid))])
            stuck = 0; step += 1
            continue
        if straight_bias > 0.0:
            r, c, cur = _persistent_step(ok, r, c, cur, straight_bias, rng)
        else:
            nr = int(np.clip(r + rng.integers(-1, 2), 0, H - 1))
            nc = int(np.clip(c + rng.integers(-1, 2), 0, W - 1))
            if ok[nr, nc]:
                r, c = nr, nc
        step += 1
    return (cells, steps) if return_steps else cells


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _persistent_step(ok, r, c, cur, straight_bias, rng):
    """One 4-connected step that prefers to keep going straight.

    Mirrors ``Utils/paths.biased_walk_path`` in the research repo, which is what
    generated the training paths: continuing straight is weighted
    ``straight_bias``, turning ``(1 - bias) / 2``, reversing 0.01, each scaled
    down by how often the target cell has been seen.
    """
    H, W = ok.shape
    valid = [(dr, dc) for dr, dc in _DIRS
             if 0 <= r + dr < H and 0 <= c + dc < W and ok[r + dr, c + dc]]
    if not valid:
        return r, c, cur
    side = (1.0 - straight_bias) / 2.0
    w = []
    for dr, dc in valid:
        dot = dr * cur[0] + dc * cur[1]
        w.append(straight_bias if dot > 0 else (side if dot == 0 else 0.01))
    w = np.asarray(w, float); w /= w.sum()
    dr, dc = valid[rng.choice(len(valid), p=w)]
    return r + dr, c + dc, (dr, dc)


def track_ages(n_cells: int, steps=None) -> np.ndarray:
    """Age in HOURS of each cell of a track, in visit order.

    Pass ``steps`` from ``make_track(..., return_steps=True)`` for the correct
    answer. Without it, ages are derived from list position, which assumes the
    walk never revisited a cell -- true for a planned transect, but this walk
    revisits enough to make the real span about 2.2x longer. The published
    staleness numbers used the positional form and so understate the effect;
    see ``docs/STALENESS_FINDINGS.md``.
    """
    if steps is not None:
        steps = np.asarray(steps, float)
        return (steps.max() - steps) * SEC_PER_CELL / 3600.0
    return np.array([(n_cells - 1 - k) * SEC_PER_CELL / 3600.0
                     for k in range(n_cells)])


def interp_at(arr: np.ndarray, t_frac: float) -> np.ndarray:
    """The field linearly interpolated to a fractional frame index.

    Never reads past ``floor(t_frac)`` when the fraction is zero, so a caller
    asking for exactly frame ``t`` cannot pull in ``t + 1`` and leak the future.
    """
    i0 = int(np.floor(t_frac))
    w = float(t_frac - i0)
    if w < 1e-9:
        return arr[i0]
    return arr[i0] * (1.0 - w) + arr[i0 + 1] * w


def lookback_ok(t: int, oldest_hours: float, block: int) -> bool:
    """True if reading back ``oldest_hours`` from frame ``t`` stays in one run."""
    back = int(np.ceil(oldest_hours)) + 1
    return t - back >= 0 and (t // block) == ((t - back) // block)


def common_ocean_mask(models, shape=(44, 94)) -> np.ndarray:
    """Intersection of every model's own ocean mask.

    Scoring on the intersection stops any model being credited for cells another
    calls land. The masks genuinely differ -- 3749 to 3796 cells across the
    library -- so this is not a formality.
    """
    common = np.ones(shape, bool)
    for m in models:
        mask = getattr(m, "ocean_mask", None)
        if mask is not None:
            common &= np.asarray(mask) > 0.5
    return common


def bootstrap_ci(x, level: float = 0.95, n: int = 20000, seed: int = 0):
    """``(mean, lo, hi)`` for a paired difference, by bootstrap over frames.

    Resamples FRAMES, not cells: cells within a frame are strongly correlated, so
    treating them as independent would give indefensibly narrow intervals.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n)])
    tail = (1.0 - level) / 2.0 * 100.0
    return float(x.mean()), float(np.percentile(draws, tail)), \
        float(np.percentile(draws, 100.0 - tail))
