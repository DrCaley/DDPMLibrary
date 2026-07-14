"""Multi-draw visualization for the StreamDDPM pipeline.

For ONE frame, draws N plausible fields from the SAME conditioning and renders:
    [ ground truth | ensemble mean | directional spread ]
    [ draw 1 | draw 2 | ... ]
All fused (coupled magnitude) on a shared colour scale, so the draws agree where
the path/priors constrain the flow and diverge where it is genuinely uncertain.

Usage:
    python scripts/multidraw.py --frame 4476 --n_draws 6 --out_dir out/
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_eval_lib import StreamEvalContext  # noqa: E402

from ddpm_library.stream import (  # noqa: E402
    ensemble_infer, dpmpp_ensemble, fuse_coupled, directional_spread,
)


def plot_field(ax, u, v, land_d, title, vmax, step=2):
    import matplotlib.colors as mcolors
    H, W = u.shape
    spd = np.sqrt(u ** 2 + v ** 2)
    im = ax.imshow(spd, origin="lower", cmap="cool", vmin=0, vmax=vmax, aspect="auto")
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    ax.quiver(xs, ys, u[::step, ::step], v[::step, ::step],
              color="black", scale=vmax * 30, width=0.003)
    ax.imshow(land_d, origin="lower",
              cmap=mcolors.ListedColormap([(0, 0, 0, 0), "black"]),
              aspect="auto", zorder=2)
    ax.set_title(title, fontsize=10)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default=None)
    ap.add_argument("--frame", type=int, default=-1,
                    help="frame id (value in valid) or split index; -1 → random")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n_draws", type=int, default=6)
    ap.add_argument("--sampler", choices=["dpmpp", "ddpm"], default="dpmpp")
    ap.add_argument("--inference_steps", type=int, default=None,
                    help="default: 6 for dpmpp, 100 for ddpm")
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_dir", default="out/multidraw")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    from ddpm_library.inference import resolve_device
    device = str(resolve_device(args.device))
    os.makedirs(args.out_dir, exist_ok=True)

    ctx = StreamEvalContext(pickle_path=args.pickle, device=device,
                            path_steps=args.path_steps)

    if args.frame >= 0:
        src_idx = ctx.resolve_frame(args.frame)
    else:
        rng = np.random.default_rng(args.seed)
        src_idx = int(rng.integers(0, len(ctx.valid)))

    cond, target, path_mask, cov, src_f = ctx.frame_conditioning(src_idx)
    steps = args.inference_steps if args.inference_steps is not None else (
        6 if args.sampler == "dpmpp" else 100)
    if args.sampler == "dpmpp":
        members = dpmpp_ensemble(ctx.model, ctx.diffusion, cond, ctx.land_np,
                                 n_members=args.n_draws, inference_steps=steps,
                                 device=device, seed=src_idx,
                                 pred_type=ctx.pred_type)
    else:
        members = ensemble_infer(ctx.model, ctx.diffusion, cond, ctx.land_np,
                                 n_members=args.n_draws, inference_steps=steps,
                                 device=device, base_seed=src_idx,
                                 pred_type=ctx.pred_type)
    fused = fuse_coupled(members, cond, ctx.land_np, ctx.het_net, ctx.hsm,
                         ctx.hss, ctx.het_clip, ctx.data_std, device)
    fused_mean = np.mean(fused, axis=0).astype(np.float32)
    spread = directional_spread(members, ctx.ocean_np)

    s = ctx.data_std
    land_d = ctx.land_np.T
    ocean_d = ~land_d
    tspd = np.sqrt((target[0] * s) ** 2 + (target[1] * s) ** 2).T
    vmax = float(np.nanpercentile(tspd[ocean_d], 98)) if ocean_d.any() else 1.0

    n = args.n_draws
    ncol = max(3, int(np.ceil(n / 2)))
    fig, axes = plt.subplots(3, ncol, figsize=(5.6 * ncol, 14), dpi=90)
    ax = axes.flatten()
    for a in ax:
        a.axis("off")

    ax[0].axis("on")
    plot_field(ax[0], target[0].T * s, target[1].T * s, land_d, "Ground truth", vmax)
    ax[1].axis("on")
    plot_field(ax[1], fused_mean[0].T * s, fused_mean[1].T * s, land_d,
               "Ensemble mean", vmax)
    ax[2].axis("on")
    im = ax[2].imshow(spread.T, origin="lower", cmap="magma", vmin=0, vmax=1,
                      aspect="auto")
    ax[2].imshow(land_d, origin="lower",
                 cmap=mcolors.ListedColormap([(0, 0, 0, 0), "black"]),
                 aspect="auto", zorder=2)
    plt.colorbar(im, ax=ax[2], label="1 - R", shrink=0.7)
    ax[2].set_title("Directional spread (uncertainty)", fontsize=10)

    for k in range(n):
        a = ax[ncol + k]
        a.axis("on")
        plot_field(a, fused[k][0].T * s, fused[k][1].T * s, land_d,
                   f"Plausible draw {k + 1}", vmax)

    plt.suptitle(f"StreamDDPM - {n} plausible fields from the SAME conditioning "
                 f"(frame {src_f}, coverage {cov:.1f}%)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out_dir, f"multidraw_frame{src_f}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"frame {src_f}  coverage {cov:.1f}%  draws {n}\nsaved: {out}")


if __name__ == "__main__":
    main()
