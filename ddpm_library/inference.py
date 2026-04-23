"""RePaint-style reverse diffusion inference for the split-head model.

Pipeline (matches scripts/benchmark_inference_timing.py reverse_chain):

    1. Start from x_T ~ q(x_T | known_standardized)
    2. For t = T-1 .. 0, repeat `resample_steps` times per step:
         a. Predict x0 from the UNet
         b. Denoise to x_{t-1} via the split posterior
         c. Splice: known region gets q_sample(known, t-1), missing gets
            the denoised value
         d. (If resampling and t > 0) re-noise x back to t and repeat
    3. Return the final x0 cropped to the ocean region, inverse-standardized.

The `known_standardized` input is zero in unobserved cells. The x_t input
to the network has the known region overwritten with fresh Gaussian noise
(`mask_xt=True` training convention).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import (
    DEFAULT_RESAMPLE_STEPS, DEFAULT_SINGLE_STEP_T, DEFAULT_T_START,
    FULL_H, FULL_W, IRR_SPEED,
    MAX_BETA, MIN_BETA, N_STEPS, OCEAN_H, OCEAN_W, WEIGHTS_PATH,
)
from .model import HelmholtzSplitSchedule, MyUNet_Helmholtz_Split_FiLM_MultiRes


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_network(
    weights_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> MyUNet_Helmholtz_Split_FiLM_MultiRes:
    """Load the UNet with bundled EMA weights."""
    path = Path(weights_path) if weights_path is not None else WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {path}. Ensure the assets/ directory "
            f"is present (Git LFS may need to be pulled)."
        )

    net = MyUNet_Helmholtz_Split_FiLM_MultiRes(
        n_steps=N_STEPS,
        time_emb_dim=256,
        in_channels=5,
        detach_heads=False,
        use_distance_field=False,
        use_bathymetry=False,
    )

    # Research-repo checkpoints wrap the UNet inside a GaussianDDPM module,
    # so state-dict keys are prefixed with "network.". Strip that prefix and
    # drop any schedule buffers the wrapper adds.
    state = torch.load(path, map_location="cpu", weights_only=False)
    net_state = {}
    for k, v in state.items():
        if k.startswith("network."):
            net_state[k[len("network."):]] = v
        elif k in {"betas", "alphas", "alpha_bars"}:
            # GaussianDDPM wrapper buffers — unused by the bare UNet
            continue
        else:
            net_state[k] = v

    missing, unexpected = net.load_state_dict(net_state, strict=False)
    # Silently allow missing non-parameter diagnostic attributes.
    hard_missing = [k for k in missing if not k.startswith("last_")]
    if hard_missing:
        raise RuntimeError(
            f"Missing keys when loading checkpoint: {hard_missing[:5]}..."
        )
    if unexpected:
        # Not fatal — older checkpoints may carry extra buffers.
        pass

    if device is not None:
        net = net.to(device)
    net.eval()
    return net


def _pad_ocean_to_full(
    arr: np.ndarray,
) -> np.ndarray:
    """Zero-pad a (C, 44, 94) or (44, 94) ocean-region array into (C, 64, 128)."""
    if arr.ndim == 2:
        out = np.zeros((FULL_H, FULL_W), dtype=arr.dtype)
        out[:OCEAN_H, :OCEAN_W] = arr
        return out
    if arr.ndim == 3:
        c = arr.shape[0]
        out = np.zeros((c, FULL_H, FULL_W), dtype=arr.dtype)
        out[:, :OCEAN_H, :OCEAN_W] = arr
        return out
    raise ValueError(f"Unsupported ndim: {arr.ndim}")


def _crop_full_to_ocean(arr: torch.Tensor) -> torch.Tensor:
    """Crop (..., 64, 128) → (..., 44, 94)."""
    return arr[..., :OCEAN_H, :OCEAN_W]


def _voronoi_fill_2ch(
    known_std: torch.Tensor, known_mask: torch.Tensor,
) -> torch.Tensor:
    """Nearest-neighbour (Voronoi) fill of sparse observations.

    Matches scripts/eval_helmholtz_split.py::voronoi_fill, restricted to
    the OCEAN_H×OCEAN_W ocean region. Used as the dense base field that
    is forward-noised by `q_sample` so that at moderate t the network
    sees a signal everywhere, not just at the <1% of observed cells.
    """
    from scipy.spatial import cKDTree

    B, C, H, W = known_std.shape
    assert B == 1 and C == 2
    km = known_mask[0, 0, :OCEAN_H, :OCEAN_W].detach().cpu().numpy()
    ky, kx = np.where(km > 0.5)
    out = torch.zeros_like(known_std)
    if len(ky) == 0:
        return out
    u = known_std[0, 0, :OCEAN_H, :OCEAN_W].detach().cpu().numpy()
    v = known_std[0, 1, :OCEAN_H, :OCEAN_W].detach().cpu().numpy()
    tree = cKDTree(np.stack([ky, kx], axis=1).astype(np.float64))
    gy, gx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
    _, idx = tree.query(np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64))
    idx = idx.reshape(OCEAN_H, OCEAN_W)
    filled = np.stack([u[ky, kx][idx], v[ky, kx][idx]], axis=0)
    out[0, :, :OCEAN_H, :OCEAN_W] = torch.from_numpy(filled).to(known_std)
    return out


def _build_model_input(
    x_t: torch.Tensor,
    miss_mask: torch.Tensor,
    known_mask: torch.Tensor,
    known_std: torch.Tensor,
) -> torch.Tensor:
    """Assemble the 5-channel input expected by the network.

    Channels: [x_t_masked(2), miss(1), sparse_obs_u(1), sparse_obs_v(1)].
    Replaces the known region of x_t with fresh Gaussian noise
    (the `mask_xt=True` training convention) so the model only sees
    observations through the sparse-obs conditioning channels.
    """
    noise_replace = torch.randn_like(x_t)
    x_t_in = x_t * miss_mask + noise_replace * known_mask
    miss_ch = miss_mask[:, :1]
    cond_field = known_std * known_mask
    return torch.cat([x_t_in, miss_ch, cond_field], dim=1)


def run_single_step(
    net: MyUNet_Helmholtz_Split_FiLM_MultiRes,
    schedule: HelmholtzSplitSchedule,
    known_std: torch.Tensor,
    miss_mask: torch.Tensor,
    *,
    t: int = N_STEPS - 1,
    voronoi: bool = False,
) -> torch.Tensor:
    """Single forward pass: one UNet call at step `t`, directly return x0.

    The model was trained with `prediction_target=x0`, so the network
    output at any timestep is a direct estimate of the clean field.
    This skips the entire iterative reverse chain — the fastest possible
    inference path. Quality is lower than full RePaint but still sensible
    for sparse-obs inpainting because the x0-prediction objective is
    well-conditioned on the observation channels.

    Returns (1, 2, 64, 128) standardized velocity prediction. The known
    region is spliced back from the input observations (not the UNet
    output) so observed cells are exactly preserved.
    """
    device = known_std.device
    known_mask = 1.0 - miss_mask
    if voronoi:
        vor_std = _voronoi_fill_2ch(known_std, known_mask)
        base = known_std * known_mask + vor_std * miss_mask
    else:
        base = known_std
    t_b = torch.tensor([t], device=device)
    x_t, _, _ = schedule.q_sample(base, t_b)
    t_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
    x0_pred = net(
        _build_model_input(x_t, miss_mask, known_mask, known_std),
        t_tensor,
    )
    # Splice: keep observations exactly, use UNet output elsewhere.
    return known_std * known_mask + x0_pred * miss_mask


def run_repaint(
    net: MyUNet_Helmholtz_Split_FiLM_MultiRes,
    schedule: HelmholtzSplitSchedule,
    known_std: torch.Tensor,
    miss_mask: torch.Tensor,
    *,
    t_start: int = DEFAULT_T_START,
    resample_steps: int = DEFAULT_RESAMPLE_STEPS,
    voronoi: bool = False,
) -> torch.Tensor:
    """RePaint-style reverse diffusion in standardized space.

    Parameters
    ----------
    known_std, miss_mask : (1, C, 64, 128)
        `known_std` holds standardized observed values (zero elsewhere) and
        `miss_mask` is 1 where unobserved, 0 at observations/land-outside-ocean.
    t_start : int
        Highest timestep to start from (typically N_STEPS-1 for full chain).
    resample_steps : int
        RePaint repaint iterations per step (3 matches the research repo).

    Returns
    -------
    (1, 2, 64, 128) standardized velocity prediction.
    """
    device = known_std.device
    known_mask = 1.0 - miss_mask
    if voronoi:
        vor_std = _voronoi_fill_2ch(known_std, known_mask)
        base = known_std * known_mask + vor_std * miss_mask
    else:
        base = known_std

    t0 = torch.tensor([t_start], device=device)
    x, _, _ = schedule.q_sample(base, t0)

    for t in range(t_start, -1, -1):
        n_resample = resample_steps if t > 0 else 1
        for r in range(n_resample):
            t_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
            x0_pred = net(_build_model_input(x, miss_mask, known_mask, known_std),
                          t_tensor)

            if t > 0:
                t_b = torch.tensor([t], device=device)
                x_denoised = schedule.p_step(x, x0_pred, t_b)
                t_prev_b = torch.tensor([t - 1], device=device)
                x_known_t, _, _ = schedule.q_sample(known_std, t_prev_b)
                x = x_known_t * known_mask + x_denoised * miss_mask
            else:
                x = known_std * known_mask + x0_pred * miss_mask

            if r < n_resample - 1 and t > 0:
                t_b = torch.tensor([t], device=device)
                x, _, _ = schedule.q_sample(x, t_b)

    return x


def make_schedule(device: torch.device) -> HelmholtzSplitSchedule:
    schedule = HelmholtzSplitSchedule(
        n_steps=N_STEPS, min_beta=MIN_BETA, max_beta=MAX_BETA,
        irr_speed=IRR_SPEED, device=device,
    )
    # Ensure all schedule tensors live on the correct device.
    for attr in ("betas_sol", "alphas_sol", "alpha_bars_sol",
                 "betas_irr", "alphas_irr", "alpha_bars_irr"):
        setattr(schedule, attr, getattr(schedule, attr).to(device))
    schedule.device = device
    return schedule


def inpaint(
    sparse_u: np.ndarray,
    sparse_v: np.ndarray,
    missing_mask: np.ndarray,
    *,
    net: MyUNet_Helmholtz_Split_FiLM_MultiRes,
    schedule: HelmholtzSplitSchedule,
    device: torch.device,
    single_step: bool = True,
    t_start: Optional[int] = None,
    resample_steps: int = DEFAULT_RESAMPLE_STEPS,
    voronoi: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Run one inference sample.

    Inputs are (44, 94) ocean-region arrays in raw m/s units (velocities)
    and {0,1} for the mask (1 = unobserved). Output is (44, 94, 2) in m/s.

    If `single_step=True` (default), runs exactly one UNet forward pass at
    step `t_start` (default `DEFAULT_SINGLE_STEP_T=50`) and returns the
    direct x0 prediction. This is dramatically faster than the iterative
    RePaint chain.

    If `single_step=False`, runs the iterative reverse chain from `t_start`
    (default `DEFAULT_T_START=75`) with `resample_steps` RePaint
    resamplings per step. Defaults match scripts/eval_helmholtz_split.py.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if t_start is None:
        t_start = DEFAULT_SINGLE_STEP_T if single_step else DEFAULT_T_START

    # Pad into the 64×128 UNet working grid.
    # Standardize observed values; unobserved cells stay zero — they're
    # ignored via known_mask anyway.
    from .standardize import standardize, inverse_standardize
    uv_raw = np.stack([sparse_u, sparse_v], axis=0)  # (2, 44, 94)
    uv_std = standardize(uv_raw).astype(np.float32)
    known_mask_44x94 = (1.0 - missing_mask).astype(np.float32)
    uv_std = uv_std * known_mask_44x94[None]  # zero out unobserved cells

    uv_full = _pad_ocean_to_full(uv_std)                  # (2, 64, 128)
    miss_full = np.ones((1, FULL_H, FULL_W), dtype=np.float32)
    miss_full[0, :OCEAN_H, :OCEAN_W] = missing_mask

    known_std_t = torch.from_numpy(uv_full).unsqueeze(0).to(device)
    miss_mask_t = torch.from_numpy(miss_full).unsqueeze(0).to(device)

    with torch.no_grad():
        if single_step:
            x_final = run_single_step(
                net, schedule, known_std_t, miss_mask_t, t=t_start,
                voronoi=voronoi,
            )
        else:
            x_final = run_repaint(
                net, schedule, known_std_t, miss_mask_t,
                t_start=t_start, resample_steps=resample_steps,
                voronoi=voronoi,
            )

    ocean = _crop_full_to_ocean(x_final).squeeze(0).cpu().numpy()  # (2, 44, 94)
    ocean = inverse_standardize(ocean)
    # Return as (44, 94, 2) — (lat, lon, {u, v})
    return np.transpose(ocean, (1, 2, 0)).astype(np.float32)
