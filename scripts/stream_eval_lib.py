"""Shared helper for the stream-model evaluation scripts.

Loads the conditioned-priors chrono dataset and builds, for a chosen frame,
everything the uncertainty-map / multidraw scripts need — all in the model's
native (2, 94, 44) orientation, reusing the vendored pipeline so results match
the research probes.

This is script-support code (needs the full dataset); it is NOT part of the
importable library and is never used by ``ddpm_library.predict``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from ddpm_library import config as C
from ddpm_library.stream import (
    DDPM, StreamFunctionUNet, geometry_channels, build_conditioning,
    load_hetero_magnitude_model,
)
from ddpm_library.stream.paths import biased_walk_path


class StreamEvalContext:
    """Loads the model + dataset once; serves per-frame conditioning + neighbours."""

    def __init__(self, pickle_path=None, device="cpu",
                 dir_weights=None, mag_weights=None, path_steps=90):
        self.device = device
        self.path_steps = path_steps

        pk = pickle_path or C.STREAM_DATASET_PATH
        if pk is None or not Path(pk).exists():
            raise FileNotFoundError(
                "Conditioned-priors dataset (ramhead_chrono.pickle) not found. "
                "It is large (~540 MB) and not bundled with the library. "
                "Pass --pickle /path/to/ramhead_chrono.pickle, or set the "
                f"STREAM_DATASET environment variable. (resolved path: {pk})")
        with open(Path(pk), "rb") as f:
            data = pickle.load(f)
        self.lags = tuple(int(x) for x in data.get("lags", (13, 25)))
        self.data_std = float(data["data_std"])
        self.data_mean = float(data.get("data_mean", 0.0))
        self.land_np = np.asarray(data["land_mask"]).astype(bool)   # (94,44)
        self.ocean_np = ~self.land_np
        self.n_ocean = int(self.ocean_np.sum())

        fields = np.nan_to_num(np.asarray(data["fields"], dtype=np.float32))
        fields = (fields - self.data_mean) / max(self.data_std, 1e-8)
        fields[:, :, self.land_np] = 0.0
        self.fields = fields                                        # (N,2,94,44)
        self.N = fields.shape[0]
        self.valid = np.asarray(data["splits"]["test"], dtype=np.int64)
        self.max_lag = max(self.lags)

        self.geom = geometry_channels(self.land_np)

        # --- models ---
        dckpt = torch.load(dir_weights or C.STREAM_DIR_WEIGHTS_PATH,
                           map_location=device, weights_only=False)
        self.pred_type = dckpt.get("pred_type", C.STREAM_PRED_TYPE)
        self.cond_ch = int(dckpt.get("cond_ch", C.STREAM_COND_CH))
        self._legacy_obs = self.cond_ch <= 10
        ca = dckpt.get("args", {})
        self.model = StreamFunctionUNet(
            in_ch=2, base_ch=ca.get("base_ch", 64),
            time_dim=ca.get("time_dim", 256), cond_ch=self.cond_ch).to(device)
        self.model.load_state_dict(dckpt["model"]); self.model.eval()
        self.diffusion = DDPM(
            T=ca.get("T", 1000), beta_schedule=ca.get("schedule", "cosine"),
            device=device, noise_type=ca.get("noise_type", "div_free"),
            spectral_filter=dckpt.get("spectral_filter", None))
        (self.het_net, self.hsm, self.hss,
         self.het_clip) = load_hetero_magnitude_model(
            mag_weights or C.STREAM_MAG_WEIGHTS_PATH, device)

    def resolve_frame(self, frame):
        """Map a FRAME id (value in valid) or split index to a split index."""
        hits = np.where(self.valid == frame)[0]
        return int(hits[0]) if len(hits) else int(frame)

    def frame_conditioning(self, src_idx, seed=None):
        """Build (cond, target, path_mask, coverage%) for a split index."""
        if seed is None:
            seed = src_idx
        f = int(self.valid[src_idx])
        target = self.fields[f]                                     # (2,94,44)
        priors = np.concatenate([self.fields[f - L] for L in self.lags], axis=0)
        path_mask = biased_walk_path(self.land_np, n_steps=self.path_steps,
                                     seed=seed, straight_bias=0.75)
        cond = build_conditioning(target, path_mask, priors, self.land_np,
                                  self.geom, legacy_obs=self._legacy_obs)
        pm_ocean = path_mask & self.ocean_np
        cov = 100.0 * pm_ocean.sum() / self.ocean_np.sum()
        return cond, target, path_mask, float(cov), f

    def empirical_neighbours(self, src_idx, path_mask, n_emp=40,
                             guard=48, min_sep=12, prior_weight=1.0):
        """Nearest real frames by observed-path + temporal-prior distance.

        Returns a list of (2,94,44) fields (empirical posterior), incl. the
        source frame first. Mirrors _probe_calib_all's matching.
        """
        f = int(self.valid[src_idx])
        src = self.fields[f]
        pm_ocean = path_mask & self.ocean_np
        obs_src = src[:, pm_ocean]
        obs_all = self.fields[:, :, pm_ocean]
        npath = max(int(pm_ocean.sum()), 1)
        dist = ((obs_all - obs_src[None]) ** 2).sum(axis=(1, 2)) / (2 * npath)

        src_priors = np.concatenate([self.fields[f - L] for L in self.lags], axis=0)
        src_p_ocean = src_priors[:, self.ocean_np]
        prior_dist = np.full(self.N, np.inf)
        f_idx = np.arange(self.max_lag, self.N)
        acc = np.zeros(f_idx.shape[0]); c = 0
        for li, L in enumerate(self.lags):
            cand = self.fields[f_idx - L][:, :, self.ocean_np]
            ref = src_p_ocean[2 * li:2 * li + 2]
            acc += ((cand - ref[None]) ** 2).sum(axis=(1, 2)); c += 2
        prior_dist[f_idx] = acc / (c * self.n_ocean)
        dist = dist + prior_weight * prior_dist

        order = np.argsort(dist); picks = []
        for cand_f in order:
            cand_f = int(cand_f)
            if not np.isfinite(dist[cand_f]) or abs(cand_f - f) <= guard:
                continue
            if any(abs(cand_f - p) < min_sep for p in picks):
                continue
            picks.append(cand_f)
            if len(picks) == n_emp - 1:
                break
        return [src] + [self.fields[p] for p in picks]
