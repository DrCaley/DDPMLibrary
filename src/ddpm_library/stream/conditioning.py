"""Conditioning assembly, magnitude fusion, and spread metrics.

Vendored faithfully from the research repo so the library reproduces the
best "old pipeline" exactly:

  * conditioning channels  — from Utils/cond_dataset.py
  * magnitude fusion       — from Conditional DDPM/testing/_probe_multidraw.py
  * Helmholtz reprojection — from Conditional DDPM/testing/_probe_calib_mag.py
  * spread (uncertainty) metrics — same three definitions used by the maps

All arrays are in the model's native (2, 94, 44) orientation and standardized
units unless noted; the predictor handles transpose + (de)standardization.
"""

from __future__ import annotations

import numpy as np
import torch

from .mag_model import HeteroMagnitudeUNet

EPS = 1e-8


# ===========================================================================
# Conditioning channels (from Utils/cond_dataset.py)
# ===========================================================================

def geometry_channels(land_mask: np.ndarray,
                      bathy: np.ndarray | None = None) -> torch.Tensor:
    """Static geometry channels [coord_x, coord_y, dist_coast] (3, H, W)."""
    from scipy import ndimage

    land = np.asarray(land_mask).astype(bool)
    H, W = land.shape
    ocean = ~land

    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)

    dist = ndimage.distance_transform_edt(ocean).astype(np.float32)
    dmax = float(dist.max())
    if dmax > 0:
        dist = dist / dmax
    dist[land] = 0.0

    channels = [xs, ys, dist]
    if bathy is not None:
        b = np.asarray(bathy, dtype=np.float32).copy()
        b[land] = 0.0
        bmax = float(b[ocean].max()) if ocean.any() else 1.0
        if bmax > 0:
            b = b / bmax
        channels.append(b)

    geom = np.stack(channels, axis=0)
    return torch.from_numpy(geom)


def observation_channels(field: torch.Tensor,
                         path_mask: np.ndarray,
                         land_np: np.ndarray,
                         legacy: bool = False) -> torch.Tensor:
    """Soft-observation channels by revealing ``field`` on a path.

    legacy=True → [obs_u, obs_v, path_mask] (3ch, the old cond_ch=10 pipeline).
    legacy=False → adds dist_to_path (4ch).
    """
    from scipy import ndimage

    pm = torch.from_numpy(np.asarray(path_mask, dtype=bool))
    obs = torch.zeros_like(field)
    obs[:, pm] = field[:, pm]
    mask = pm.float()[None]

    pm_np = pm.numpy()
    ocean_np = ~land_np
    dist = ndimage.distance_transform_edt(~pm_np).astype(np.float32)
    dist[land_np] = 0.0
    dmax = float(dist[ocean_np].max()) if ocean_np.any() and dist[ocean_np].max() > 0 else 1.0
    dist = dist / dmax
    dist[land_np] = 0.0
    dist_t = torch.from_numpy(dist)[None]

    if legacy:
        return torch.cat([obs, mask], dim=0)          # (3, H, W)
    return torch.cat([obs, mask, dist_t], dim=0)      # (4, H, W)


def assemble_cond(obs: torch.Tensor, priors: torch.Tensor,
                  geom: torch.Tensor) -> torch.Tensor:
    """Canonical channel order: [ obs | priors | geom ]."""
    return torch.cat([obs, priors, geom], dim=0)


def build_conditioning(obs_field_std: np.ndarray,
                       path_mask: np.ndarray,
                       priors_std: np.ndarray,
                       land_np: np.ndarray,
                       geom: torch.Tensor,
                       *,
                       legacy_obs: bool) -> torch.Tensor:
    """Assemble the (C, H, W) conditioning tensor from standardized arrays.

    Parameters
    ----------
    obs_field_std : (2, H, W) standardized sparse field (zeros off-path).
    path_mask     : (H, W) bool, True where observed.
    priors_std    : (2*n_lags, H, W) standardized prior fields.
    land_np       : (H, W) bool, True = land.
    geom          : (n_geom, H, W) static geometry (precomputed).
    legacy_obs    : True for the old cond_ch=10 model (3 obs channels).
    """
    obs = observation_channels(torch.from_numpy(obs_field_std.astype(np.float32)),
                               path_mask, land_np, legacy=legacy_obs)
    priors_t = torch.from_numpy(priors_std.astype(np.float32))
    return assemble_cond(obs, priors_t, geom)


# ===========================================================================
# Magnitude model + fusion (from _probe_multidraw.py)
# ===========================================================================

def load_hetero_magnitude_model(checkpoint, device):
    """Load a HeteroMagnitudeUNet -> (net, speed_mean, speed_std, logvar_clip)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    base_ch = ckpt.get("args", {}).get("base_ch", 64)
    in_ch = int(ckpt["model"]["enc0.conv1.weight"].shape[1])
    sd = ckpt["model"]
    head_hidden = (int(sd["logvar_head.0.weight"].shape[0])
                   if any(k.startswith("logvar_head") for k in sd) else 0)
    net = HeteroMagnitudeUNet(in_ch=in_ch, base_ch=base_ch,
                              head_hidden=head_hidden).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    net.in_ch = in_ch
    return (net, float(ckpt["speed_mean"]), float(ckpt["speed_std"]),
            tuple(ckpt.get("logvar_clip", (-8.0, 4.0))))


@torch.no_grad()
def predict_speed_mean_sigma(net, speed_mean, speed_std, land_mask, data_std,
                             device, cond, logvar_clip):
    """Per-cell (mu_norm, sigma_norm) speed from the hetero UNet.

    ``land_mask`` is the LAND mask (True = land). Returns two (H, W) float32
    arrays in normalized (÷data_std) units.
    """
    c = cond if torch.is_tensor(cond) else torch.from_numpy(np.asarray(cond))
    mean, logvar = net(c.unsqueeze(0).to(device).float())
    mu = mean[0, 0].cpu().numpy()
    lv = logvar[0, 0].clamp(*logvar_clip).cpu().numpy()
    mu_phys = np.clip(mu * speed_std + speed_mean, 0.0, None)
    sigma_phys = np.exp(0.5 * lv) * speed_std
    mu_phys[land_mask] = 0.0
    sigma_phys[land_mask] = 0.0
    return (mu_phys / data_std).astype(np.float32), (sigma_phys / data_std).astype(np.float32)


def coupled_magnitude(members, speed_mu, speed_sigma, ocean_np):
    """Direction-coupled magnitude calibration (no white noise).

    Reuse each diffusion draw's own magnitude anomaly, standardize it across
    the ensemble, and rescale to the hetero UNet's per-cell mu(x)/sigma(x).
    """
    arr = np.stack(members, axis=0).astype(np.float64)        # (K, 2, H, W)
    mag = np.sqrt((arr ** 2).sum(axis=1))                     # (K, H, W)
    mbar = mag.mean(axis=0)
    s = mag.std(axis=0)
    z = (mag - mbar[None]) / (s[None] + EPS)
    out = []
    for k, m in enumerate(members):
        spd = np.clip(speed_mu + speed_sigma * z[k], 0.0, None)
        u, v = m[0], m[1]
        d = np.sqrt(u ** 2 + v ** 2) + EPS
        fu = (u / d * spd).astype(np.float32)
        fv = (v / d * spd).astype(np.float32)
        fu[~ocean_np] = 0.0
        fv[~ocean_np] = 0.0
        out.append(np.stack([fu, fv], axis=0))
    return out


def helmholtz_project(field, ocean_mask, max_iters=5, tol=1e-4, symbol="discrete"):
    """Iterative FFT Helmholtz projection to remove the divergent component.

    ``symbol`` selects the Fourier derivative symbol:

      "continuous"            k = 2*pi*f      -- what Stream numbers before
                                                 2026-09-05 were produced with
      "discrete" (default)    k = sin(2*pi*f) -- the central-difference symbol,
                                                 matching both the divergence
                                                 operator these fields are scored
                                                 with and `leray_project`, which
                                                 built the training data

    "discrete" is the mathematically consistent choice: it matches the operator the
    fields are scored with, and on real prior fields leaves 1.2e-04 divergence
    against "continuous"'s 1.1e-03. It costs a little accuracy. Paired over the 40
    benchmark cases, discrete minus continuous:

        rmse       +0.000646  CI [+0.000363, +0.000991]  worse
        angle      +0.005361  CI [+0.001702, +0.009281]  worse
        vort_rmse  +0.000346  CI [+0.000277, +0.000426]  worse
        vort_corr  -0.010692  CI [-0.013427, -0.008259]  worse
        div_rmse   -0.000473  CI [-0.000577, -0.000374]  better
        crps_cal   +0.000085  CI [-0.000004, +0.000184]  tied

    The reason is that the truth is not divergence-free. A partial projection leaves
    residual divergence that happens to sit closer to the real thing than a clean
    projection plus the VCNN's estimated divergent component does. Consistency with
    the training-data projection was judged worth that cost.
    STREAM_SIGMA_SCALE_TIMED should be refit for this setting (measured 3.147).
    """
    land = ~ocean_mask
    ux = field[0].copy().astype(np.float64); ux[land] = 0.0
    uy = field[1].copy().astype(np.float64); uy[land] = 0.0
    H, W = ux.shape
    if symbol == "discrete":
        kx = np.sin(2 * np.pi * np.fft.fftfreq(H, d=1.0))[:, None]
        ky = np.sin(2 * np.pi * np.fft.rfftfreq(W, d=1.0))[None, :]
    elif symbol == "continuous":
        kx = np.fft.fftfreq(H, d=1.0 / (2 * np.pi))[:, None]
        ky = np.fft.rfftfreq(W, d=1.0 / (2 * np.pi))[None, :]
    else:
        raise ValueError(f"symbol must be 'continuous' or 'discrete'; got {symbol!r}")
    k2 = kx ** 2 + ky ** 2
    k2_safe = np.where(k2 > 0.0, k2, 1.0)      # DC and Nyquist: sin() vanishes
    interior = np.zeros((H, W), bool); interior[1:-1, 1:-1] = True
    check = interior & ocean_mask
    prev = np.inf
    for _ in range(max_iters):
        Ux = np.fft.rfft2(ux); Uy = np.fft.rfft2(uy)
        Phi = -(1j * kx * Ux + 1j * ky * Uy) / k2_safe
        Phi[k2 == 0.0] = 0.0
        ux = np.fft.irfft2(Ux - 1j * kx * Phi, s=(H, W)); ux[land] = 0.0
        uy = np.fft.irfft2(Uy - 1j * ky * Phi, s=(H, W)); uy[land] = 0.0
        # divergence is d(ux)/d0 + d(uy)/d1, matching how kx/ky are paired above.
        # This previously used the transposed pairing, so the early-exit tolerance
        # fired on a quantity that was not the divergence.
        dux = np.zeros_like(ux); dux[1:-1, :] = (ux[2:, :] - ux[:-2, :]) / 2
        duy = np.zeros_like(uy); duy[:, 1:-1] = (uy[:, 2:] - uy[:, :-2]) / 2
        cur = float(np.abs(dux + duy)[check].mean())
        if abs(prev - cur) / (prev + 1e-12) < tol:
            break
        prev = cur
    return np.stack([ux.astype(np.float32), uy.astype(np.float32)], axis=0)


def fuse_coupled(members, cond, land_np, het_net, hsm, hss, het_clip,
                 data_std, device):
    """Full coupled fuse used by the best pipeline: hetero mu/sigma → coupled
    magnitude → Helmholtz reprojection. Returns a list of (2, H, W) arrays."""
    ocean_np = ~land_np
    mu_n, sig_n = predict_speed_mean_sigma(
        het_net, hsm, hss, land_np, data_std, device, cond, het_clip)
    fixed = coupled_magnitude(members, mu_n, sig_n, ocean_np)
    return [helmholtz_project(d, ocean_np) for d in fixed]


# ===========================================================================
# Spread (uncertainty) metrics — the three used by the uncertainty maps
# ===========================================================================

def unit_normalize(field_np, ocean_np, eps=1e-8):
    """Unit-normalize each vector of a (2, H, W) field; land/near-zero -> 0."""
    u, v = field_np[0], field_np[1]
    mag = np.sqrt(u ** 2 + v ** 2)
    safe = mag > eps
    uh = np.zeros_like(u); vh = np.zeros_like(v)
    uh[safe] = u[safe] / mag[safe]
    vh[safe] = v[safe] / mag[safe]
    uh[~ocean_np] = 0.0; vh[~ocean_np] = 0.0
    return uh, vh, mag


def directional_spread(members, ocean_np, eps=1e-8):
    """Per-cell circular spread (1 - resultant length). NaN at land."""
    us, vs = [], []
    for m in members:
        uh, vh, _ = unit_normalize(m, ocean_np, eps)
        us.append(uh); vs.append(vh)
    mean_u = np.mean(us, axis=0); mean_v = np.mean(vs, axis=0)
    R = np.sqrt(mean_u ** 2 + mean_v ** 2)
    spread = 1.0 - R
    spread[~ocean_np] = np.nan
    return spread.astype(np.float32)


def vector_spread(members, ocean_np, mag_norm="abs"):
    """Full-vector RMS dispersion across the ensemble. NaN at land."""
    arr = np.stack(members, axis=0).astype(np.float64)
    mean = arr.mean(axis=0)
    dev = arr - mean[None]
    disp = np.sqrt((dev ** 2).sum(axis=1).mean(axis=0))
    if mag_norm == "cov":
        mean_mag = np.sqrt((arr ** 2).sum(axis=1)).mean(axis=0)
        disp = disp / (mean_mag + EPS)
    out = disp.astype(np.float32)
    out[~ocean_np] = np.nan
    return out


def magnitude_spread(members, ocean_np, mag_norm="abs"):
    """Speed-only dispersion across the ensemble. NaN at land."""
    arr = np.stack(members, axis=0).astype(np.float64)
    speed = np.sqrt((arr ** 2).sum(axis=1))
    disp = speed.std(axis=0)
    if mag_norm == "cov":
        disp = disp / (speed.mean(axis=0) + EPS)
    out = disp.astype(np.float32)
    out[~ocean_np] = np.nan
    return out


def pcorr(a, b, eps=1e-12):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))
