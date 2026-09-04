"""Build ocean_bench_v1.npz -- a frozen, shareable reconstruction benchmark.

Everyone in this group is currently scoring a different task: different vehicle
tracks, different observation timing, different cell masks, different RMSE and
angle conventions. Two independent reimplementations of "the same" experiment
agree on model ORDERING and disagree on absolute numbers by 15-50%, which makes
cross-checking results impossible.

This file removes every one of those choices. It fixes the observations, their
timestamps, the target fields, the temporal priors, the scored cell set and the
metric code. Run your model on these inputs, score with score.py, and any
remaining disagreement is genuinely in the model rather than in the harness.

The observation process is the realistic one: a Dubins-style vehicle (constant
speed, bounded turn rate, random walk in the steering command) drives for 2 h
from a random start, and each reading is the field AT THE MOMENT the vehicle
reached that cell -- not the target frame's value. The target is the field at the
END of the run. Scoring against instantaneous observations flatters models that
were trained that way and measures a task no vehicle can perform.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _paths import PICKLES_DIR  # noqa: E402
from ddpm_library.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX   # noqa: E402
from _harness import load_fields, frame_pool, interp_at, lookback_ok  # noqa: E402

PICKLE = str(PICKLES_DIR / "data_raw_chrono.pickle")
# Replication support: --seed/--out/--exclude let a second, independent set be
# generated with the same protocol. --exclude takes an existing benchmark whose
# target frames are removed from the pool, so the two sets share no frames.
FRAMES_FILE = Path(__file__).resolve().parents[1] / "scripts" / "fair_eval_frames.json"
N_CASES, SEED = 40, 20260829
N_READINGS = 200
CELL_M, SPEED_MS, DUR_SEC, STEP_M, TURN_RADIUS_CELLS = 50.8, 1.06, 7200.0, 5.0, 2.01
LAGS = (13, 25)


def dubins_path(ok, rng):
    """Simulated at 5 m resolution; subsampled afterwards so the trajectory is
    independent of how often it is read."""
    ok = np.asarray(ok, bool); H, W = ok.shape
    dt = STEP_M / SPEED_MS
    n = int(DUR_SEC / dt) + 1
    speed_c, om_max = STEP_M / CELL_M, (STEP_M / CELL_M) / TURN_RADIUS_CELLS
    om_sig = 0.13 * np.sqrt(dt / 36.0)
    valid = np.argwhere(ok)
    y, x = (float(v) for v in valid[rng.integers(len(valid))])
    th, om = rng.uniform(0, 2 * np.pi), 0.0
    cells, times = [], []
    for s in range(n):
        cells.append((int(round(y)), int(round(x)))); times.append(s * dt)
        om = float(np.clip(om + rng.normal(0, om_sig), -om_max, om_max))
        for attempt in range(64):
            th_try = th + om
            ny, nx = y + speed_c * np.sin(th_try), x + speed_c * np.cos(th_try)
            iy, ix = int(round(ny)), int(round(nx))
            if 0 <= iy < H and 0 <= ix < W and ok[iy, ix]:
                th, y, x = th_try, ny, nx; break
            om = om_max if attempt % 2 == 0 else -om_max
        else:
            th += np.pi; om = 0.0
    idx = np.unique(np.linspace(0, len(cells) - 1, N_READINGS).round().astype(int))
    return [cells[i] for i in idx], np.asarray(times)[idx]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", default=None,
                    help="existing benchmark .npz whose frames to exclude")
    args = ap.parse_args()
    seed = args.seed

    arr, block = load_fields(PICKLE)
    # Score the INTERSECTION of every model's own ocean mask. Shipping one
    # model's mask would charge the others for cells they call land, and the
    # masks genuinely differ (3749 vs 3787 across this library).
    A = Path(__file__).resolve().parents[1] / "src" / "ddpm_library" / "assets"
    ocean = (~np.asarray(np.load(A / "corrdiff_grid.npz")["land_mask"]).astype(bool)).T
    ocean &= (~np.asarray(np.load(A / "stream_grid.npz")["land_mask"]).astype(bool)).T
    da = np.load(A / "distattn_ocean_mask.npy")
    ocean &= (da.T if da.shape != ocean.shape else da) > 0.5         # (44, 94)
    lats = np.linspace(LAT_MIN, LAT_MAX, 44)
    lons = np.linspace(LON_MIN, LON_MAX, 94)

    rng = np.random.default_rng(seed)
    picks = frame_pool(FRAMES_FILE, N_CASES * 3, arr.shape[0], rng)
    if args.exclude:
        banned = set(int(x) for x in np.load(args.exclude)["frame_index"])
        picks = [t for t in picks if t not in banned]

    obs_all, truth_all, priors_all, frames, spans, drifts = [], [], [], [], [], []
    for t in picks:
        if len(frames) >= N_CASES:
            break
        cells, times = dubins_path(ocean, rng)
        if not lookback_ok(t, float(max(LAGS)), block):
            continue
        ages_h = (times[-1] - times) / 3600.0
        t_end = 1_700_000_000.0 + t * 3600.0
        truth = arr[t]
        rows = []
        for k, (r, c) in enumerate(cells):
            u, v = interp_at(arr, t - ages_h[k])[r, c]     # value WHEN VISITED
            rows.append((float(lats[r]), float(lons[c]),
                         t_end - float(ages_h[k]) * 3600.0, float(u), float(v)))
        rows = np.asarray(rows, np.float64)
        drift = float(np.abs(rows[:, 3:] - np.array([truth[r, c] for r, c in cells])).mean())
        if drift < 1e-4:
            raise RuntimeError(f"frame {t}: readings are not time-varying "
                               f"(drift {drift:.2e}) -- check the age units")
        obs_all.append(rows); truth_all.append(truth)
        priors_all.append(np.stack([arr[t - L] for L in LAGS]))
        frames.append(t); spans.append(float(ages_h.max())); drifts.append(drift)

    if len(frames) < N_CASES:
        raise RuntimeError(f"only {len(frames)} usable cases; widen the frame pool")

    out = (Path(args.out) if args.out
           else Path(__file__).resolve().parent / "ocean_bench_v1.npz")
    np.savez_compressed(
        out,
        observations=np.asarray(obs_all, np.float64),      # (C, N, 5)
        truth=np.asarray(truth_all, np.float32),           # (C, 44, 94, 2)
        priors=np.asarray(priors_all, np.float32),         # (C, 2, 44, 94, 2)
        ocean_mask=ocean,                                  # (44, 94) bool
        lats=lats, lons=lons, frame_index=np.asarray(frames),
        prior_lags_hours=np.asarray(LAGS),
        collection_span_hours=np.asarray(spans),
        reading_drift_ms=np.asarray(drifts),
        seed=seed, n_readings=N_READINGS, cell_metres=CELL_M,
        vehicle_speed_ms=SPEED_MS, turn_radius_metres=TURN_RADIUS_CELLS * CELL_M,
    )
    spd = np.linalg.norm(np.asarray(truth_all)[:, ocean], axis=-1)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(frames)} cases x {N_READINGS} readings")
    print(f"  collection span   {np.mean(spans):.2f} h")
    print(f"  scored cells      {int(ocean.sum())}")
    print(f"  RMS current speed {np.sqrt((spd ** 2).mean()):.4f} m/s")
    print(f"  reading drift     {np.mean(drifts):.4f} m/s "
          f"(mean |reading - target-frame value|)")


if __name__ == "__main__":
    main()
