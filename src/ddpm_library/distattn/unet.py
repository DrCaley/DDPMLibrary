"""
Time-and-Distance-Aware Attention conditioned UNet for ocean current inpainting.

The time-aware extension of Conditioning/distattn_model.py (copied, not
imported — the Conditioning package is not used by anything in this folder).
DistAttnUNet penalizes attention by the physical distance between a query
pixel and an observation; this model additionally penalizes by the
observation's AGE — how long ago the robot measured that cell:

    attn = (q @ k^T) / sqrt(d)  -  alpha * distance(query_xy, obs_xy)
                                -  beta  * age(obs)

Observation tokens grow from 4 to 5 dims: [x_norm, y_norm, u, v, age_norm],
where age_norm = (t_end - t_obs) / 3600 s (hours since the observation, so 0
for the newest observation and up to ~3 for the oldest on a 3 h path). The
age also enters obs_proj, so the network can modulate the value/key content
by staleness, not just down-weight it.

Both alpha and beta are learned scalars per attention instance,
zero-initialized. Combined with zero-padding the new age column of obs_proj
during warm-start, a freshly warm-started TimeCondUNet is bit-for-bit
identical to the trained DistAttnUNet checkpoint it loaded — training decides
how much distance and staleness matter.

Architecture otherwise identical to DistAttnUNet: same depth, channel widths,
and (94, 44) -> (96, 48) padding.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Token layout and the age normalization constant (seconds -> token units).
OBS_DIM       = 5
AGE_SCALE_SEC = 3600.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


def _num_groups(channels: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


def _normalized_grid(H: int, W: int, device) -> torch.Tensor:
    """
    (H*W, 2) grid of [x_norm, y_norm] in [0, 1], row-major flattened to match
    x.flatten(2)'s ordering (flat index i -> h = i // W, w = i % W).

    Approximation: this treats each resolution's feature map as spanning [0,1]
    uniformly, ignoring the exact (2,2,1,1) pad offset applied before the UNet's
    first downsample. A soft attention bias only needs to be monotonic with true
    physical distance, not pixel-exact, so this is a deliberate simplification —
    not a bug.
    """
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (H*W, 2)


# ---------------------------------------------------------------------------
# Time-and-distance-aware cross-attention
# ---------------------------------------------------------------------------

class TimeDistAwareAttention(nn.Module):
    """
    Same query/key/value structure as DistAwareAttention, plus a learned age
    penalty on the raw attention score. Both `alpha` (distance) and `beta`
    (age) start at 0, so a warm-started module is bit-for-bit identical to
    the distattn checkpoint until training moves them.
    """

    def __init__(self, ch: int, obs_dim: int = OBS_DIM, n_heads: int = 4):
        super().__init__()
        assert ch % n_heads == 0, f"ch={ch} not divisible by n_heads={n_heads}"
        self.n_heads  = n_heads
        self.head_dim = ch // n_heads

        self.norm     = nn.GroupNorm(_num_groups(ch), ch)
        self.obs_proj = nn.Linear(obs_dim, ch)
        self.to_q     = nn.Linear(ch, ch)
        self.to_k     = nn.Linear(ch, ch)
        self.to_v     = nn.Linear(ch, ch)
        self.to_out   = nn.Linear(ch, ch)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # Learned penalties, zero-init: neither distance nor staleness
        # contributes anything until training decides otherwise.
        self.alpha = nn.Parameter(torch.tensor(0.0))   # distance penalty
        self.beta  = nn.Parameter(torch.tensor(0.0))   # age penalty (NEW)

    def _attn_weights(self, x: torch.Tensor, obs_tokens: torch.Tensor,
                       obs_mask: torch.Tensor):
        """Returns (attn, v, (B, C, H, W)) — post-softmax weights, exposed as a
        seam for testing the distance/age biases in isolation."""
        B, C, H, W = x.shape
        nh, dh = self.n_heads, self.head_dim

        h = self.norm(x).flatten(2).transpose(1, 2)   # (B, HW, C)
        q = self.to_q(h)
        obs_emb = self.obs_proj(obs_tokens)            # (B, N, C)
        k = self.to_k(obs_emb)
        v = self.to_v(obs_emb)

        q = q.view(B, H * W, nh, dh).transpose(1, 2)   # (B, nh, HW, dh)
        k = k.view(B, -1,    nh, dh).transpose(1, 2)   # (B, nh, N,  dh)
        v = v.view(B, -1,    nh, dh).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (dh ** 0.5)  # (B, nh, HW, N)

        query_xy = _normalized_grid(H, W, x.device)              # (HW, 2)
        obs_xy   = obs_tokens[:, :, :2]                           # (B, N, 2)
        dist     = torch.cdist(query_xy.unsqueeze(0).expand(B, -1, -1), obs_xy)
        attn     = attn - self.alpha * dist[:, None, :, :]        # broadcast over heads

        # NEW vs. DistAwareAttention: penalize stale observations. The age is
        # per-token, so it broadcasts over both heads and query pixels.
        age  = obs_tokens[:, :, 4]                                # (B, N)
        attn = attn - self.beta * age[:, None, None, :]

        mask = obs_mask[:, None, None, :]                         # (B, 1, 1, N)
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        return attn, v, (B, C, H, W)

    def forward(self, x: torch.Tensor, obs_tokens: torch.Tensor,
                obs_mask: torch.Tensor) -> torch.Tensor:
        attn, v, (B, C, H, W) = self._attn_weights(x, obs_tokens, obs_mask)
        out = attn @ v                                  # (B, nh, HW, dh)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.to_out(out)
        return out.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# ResBlock with time-and-distance-aware attention conditioning
# ---------------------------------------------------------------------------

class TimeCondResBlock(nn.Module):
    """
    Same backbone/naming as DistAttnResBlock (norm1/conv1/time_fc/norm2/conv2/
    skip/cross_attn — the attribute name `cross_attn` is kept unchanged so a
    trained distattn checkpoint's weights load directly into this model).
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 obs_dim: int = OBS_DIM, n_heads: int = 4):
        super().__init__()
        self.norm1   = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1   = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_fc = nn.Linear(time_dim, out_ch)
        self.norm2   = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip    = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act     = nn.SiLU()

        # Attribute name matches DistAttnResBlock.cross_attn exactly (warm-start).
        self.cross_attn = TimeDistAwareAttention(out_ch, obs_dim=obs_dim, n_heads=n_heads)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                obs_tokens: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        h = h + self.time_fc(self.act(t_emb))[:, :, None, None]
        h = h + self.cross_attn(h, obs_tokens, obs_mask)

        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Time-Conditioned UNet
# ---------------------------------------------------------------------------

class TimeCondUNet(nn.Module):
    """
    Structurally identical to DistAttnUNet (same submodule names: enc0..enc3,
    mid, dec3..dec0, out_conv, time_mlp, down, up) so a trained distattn
    checkpoint's weights load directly, leaving only the new `beta` scalars
    (one per ResBlock) at their zero initialization; obs_proj's new 5th input
    column (age) is zero-padded during warm-start.
    """

    _PAD  = (2, 2, 1, 1)
    _UPAD = (2, 2, 1, 1)

    def __init__(self, in_ch: int = 2, base_ch: int = 64, time_dim: int = 256,
                 obs_dim: int = OBS_DIM, n_heads: int = 4):
        super().__init__()
        self.time_dim = time_dim
        c = base_ch

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        def block(i, o):
            return TimeCondResBlock(i, o, time_dim, obs_dim=obs_dim, n_heads=n_heads)

        self.enc0 = block(in_ch,    c)       # 96×48
        self.enc1 = block(c,       c*2)      # 48×24
        self.enc2 = block(c*2,     c*4)      # 24×12
        self.enc3 = block(c*4,     c*8)      # 12×6

        self.mid  = block(c*8,     c*8)      # 6×3

        self.dec3 = block(c*8+c*8, c*4)
        self.dec2 = block(c*4+c*4, c*2)
        self.dec1 = block(c*2+c*2, c)
        self.dec0 = block(c  +c,   c)

        self.out_conv = nn.Conv2d(c, in_ch, 1)
        self.down     = nn.MaxPool2d(2)
        self.up       = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                obs_tokens: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (B, 2, 94, 44) noisy field
            t:          (B,) integer timesteps
            obs_tokens: (B, N_obs, 5) [x_norm, y_norm, u, v, age_norm] per
                        observation, in visit order; age_norm =
                        (t_end - t_obs) / 3600 s (0 = newest observation)
            obs_mask:   (B, N_obs) bool, True = real observation (vs. padding)
        Returns:
            predicted noise: (B, 2, 94, 44)
        """
        x = F.pad(x, self._PAD)

        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        a = (obs_tokens, obs_mask)
        e0 = self.enc0(x,             t_emb, *a)
        e1 = self.enc1(self.down(e0), t_emb, *a)
        e2 = self.enc2(self.down(e1), t_emb, *a)
        e3 = self.enc3(self.down(e2), t_emb, *a)

        h  = self.mid(self.down(e3),  t_emb, *a)

        h = self.dec3(torch.cat([self.up(h), e3], dim=1), t_emb, *a)
        h = self.dec2(torch.cat([self.up(h), e2], dim=1), t_emb, *a)
        h = self.dec1(torch.cat([self.up(h), e1], dim=1), t_emb, *a)
        h = self.dec0(torch.cat([self.up(h), e0], dim=1), t_emb, *a)

        h = self.out_conv(h)

        left, right, top, bottom = self._UPAD
        return h[:, :, top:h.shape[2]-bottom, left:h.shape[3]-right]

    # -----------------------------------------------------------------
    # Warm-start from a trained Distance-Aware Attention checkpoint
    # -----------------------------------------------------------------

    def load_from_distattn(self, distattn_ckpt_path: str, device: str = "cpu"):
        """
        Copy every weight from a trained DistAttnUNet checkpoint. Two things
        differ from a plain load:

          * obs_proj.weight grew from (ch, 4) to (ch, 5) — the checkpoint
            tensor is zero-padded with a 5th (age) input column, so the age
            feature is initially invisible to the projection.
          * each attention gained one `beta` scalar — left at zero-init.

        Together with beta = 0 this makes the warm-started model's output
        bit-for-bit identical to the trained DistAttnUNet for ANY age values
        (verified by the init-equivalence smoke test).

        Raises on any unexpected checkpoint key (naming drift) and on any
        missing key that is not a `.beta` scalar.
        """
        ckpt  = torch.load(distattn_ckpt_path, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt

        sched = ckpt.get("schedule", "linear") if isinstance(ckpt, dict) else "linear"
        if sched != "linear":
            raise RuntimeError(
                f"Warm-start checkpoint was trained with beta schedule "
                f"'{sched}', but this pipeline's DDPM (root ddpm.py) is "
                f"linear-only."
            )

        state = dict(state)
        n_padded = 0
        for key, w in list(state.items()):
            if key.endswith("cross_attn.obs_proj.weight") and w.shape[1] == 4:
                state[key] = torch.cat(
                    [w, torch.zeros(w.shape[0], 1, dtype=w.dtype, device=w.device)],
                    dim=1,
                )
                n_padded += 1

        result = self.load_state_dict(state, strict=False)

        unexpected = list(result.unexpected_keys)
        if unexpected:
            raise RuntimeError(
                f"Warm-start failed: {len(unexpected)} checkpoint keys have no "
                f"matching parameter in TimeCondUNet (naming drift?): "
                f"{unexpected[:5]}..."
            )

        missing = list(result.missing_keys)
        non_beta_missing = [k for k in missing if not k.endswith(".beta")]
        if non_beta_missing:
            raise RuntimeError(
                f"Warm-start failed: {len(non_beta_missing)} non-beta params "
                f"were left uninitialized (expected only 'beta' scalars to be "
                f"new): {non_beta_missing[:5]}..."
            )

        print(f"Warm-start from {distattn_ckpt_path}:")
        print(f"  Loaded tensors                     : {len(state) - len(unexpected)}")
        print(f"  obs_proj weights padded 4 -> 5 dims: {n_padded}")
        print(f"  New (zero-init) beta params        : {len(missing)}")
        return result
