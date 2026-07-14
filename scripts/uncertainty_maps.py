"""Uncertainty-map calibration for the StreamDDPM pipeline.

For each frame: draw a model ensemble, fuse (coupled magnitude), and compare the
model's per-pixel spread maps against the EMPIRICAL neighbour-posterior spread
maps. Reports the three calibration correlations plus accuracy, and renders a
3x2 (angle / magnitude / overall) x (empirical / model) panel per frame.

    r_angle      corr of directional-spread maps  (1 - |mean unit vec|)
    r_magnitude  corr of speed-spread maps
    r_overall    corr of full-vector-spread maps
    RMSE, angle  accuracy of the fused ensemble mean vs the true field

Usage:
    python scripts/uncertainty_maps.py --n_frames 6 --n_model 40 --out_dir out/
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_eval_lib import StreamEvalContext  # noqa: E402

from ddpm_library.stream import (  # noqa: E402
    ensemble_infer, dpmpp_ensemble, fuse_coupled, directional_spread,
    magnitude_spread, vector_spread, pcorr,
)


def draw_members(ctx, cond, si, sampler, steps, n_model, device):
    if sampler == "dpmpp":
        return dpmpp_ensemble(ctx.model, ctx.diffusion, cond, ctx.land_np,
                              n_members=n_model, inference_steps=steps,
                              device=device, seed=si, pred_type=ctx.pred_type)
    return ensemble_infer(ctx.model, ctx.diffusion, cond, ctx.land_np,
                          n_members=n_model, inference_steps=steps,
                          device=device, base_seed=si, pred_type=ctx.pred_type)

EPS = 1e-8


def accuracy(pred, true, ocean, data_std):
    pu, pv = pred[0][ocean], pred[1][ocean]
    tu, tv = true[0][ocean], true[1][ocean]
    ps = np.sqrt(pu ** 2 + pv ** 2); ts = np.sqrt(tu ** 2 + tv ** 2)
    rmse = float(np.sqrt((((pu - tu) ** 2 + (pv - tv) ** 2) * data_std ** 2).mean()))
    puh, pvh = pu / (ps + EPS), pv / (ps + EPS)
    tuh, tvh = tu / (ts + EPS), tv / (ts + EPS)
    ang = float(np.degrees(np.arccos(np.clip(puh * tuh + pvh * tvh, -1, 1))).mean())
    return rmse, ang


def render(maps, land_np, src_f, cov, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    land_d = land_np.T
    rows = [("angle", "1 - |mean unit vec|"), ("magnitude", "std speed"),
            ("overall", "RMS vector disp")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 14), dpi=90)
    for ri, (key, ylabel) in enumerate(rows):
        emp, mod, r = maps[key]
        both = np.concatenate([emp[np.isfinite(emp)], mod[np.isfinite(mod)]])
        vmax = float(np.nanpercentile(both, 98)) if both.size else 1.0
        for ci, (arr, title) in enumerate([(emp, "Empirical"),
                                           (mod, f"Model  (r={r:+.3f})")]):
            ax = axes[ri, ci]
            im = ax.imshow(arr.T, origin="lower", cmap="magma", vmin=0, vmax=vmax,
                           aspect="auto")
            ax.imshow(land_d, origin="lower",
                      cmap=mcolors.ListedColormap([(0, 0, 0, 0), "black"]),
                      aspect="auto", zorder=2)
            ax.set_title(f"{title} - r_{key}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.7)
    plt.suptitle(f"Uncertainty maps - empirical vs model  (frame {src_f}, "
                 f"cov {cov:.1f}%)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default=None)
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--n_model", type=int, default=40)
    ap.add_argument("--n_emp", type=int, default=40)
    ap.add_argument("--sampler", choices=["dpmpp", "ddpm"], default="dpmpp")
    ap.add_argument("--inference_steps", type=int, default=None,
                    help="default: 6 for dpmpp, 100 for ddpm")
    ap.add_argument("--path_steps", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_dir", default="out/uncertainty_maps")
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="explicit frame ids; else random from --seed")
    args = ap.parse_args()

    from ddpm_library.inference import resolve_device
    device = str(resolve_device(args.device))
    os.makedirs(args.out_dir, exist_ok=True)

    ctx = StreamEvalContext(pickle_path=args.pickle, device=device,
                            path_steps=args.path_steps)

    if args.frames:
        idxs = [ctx.resolve_frame(f) for f in args.frames]
    else:
        rng = np.random.default_rng(args.seed)
        idxs = [int(x) for x in rng.integers(0, len(ctx.valid), size=args.n_frames)]

    ocean = ctx.ocean_np
    steps = args.inference_steps if args.inference_steps is not None else (
        6 if args.sampler == "dpmpp" else 100)
    rows = []
    print(f"device={device}  sampler={args.sampler}  n_model={args.n_model}  steps={steps}\n")
    hdr = f"{'frame':>7} {'cov%':>6} {'r_angle':>9} {'r_magn':>8} {'r_overall':>10} {'RMSE':>8} {'angle':>7}"
    print(hdr)
    for si in idxs:
        cond, target, path_mask, cov, src_f = ctx.frame_conditioning(si)
        members = draw_members(ctx, cond, si, args.sampler, steps,
                               args.n_model, device)
        fused = fuse_coupled(members, cond, ctx.land_np, ctx.het_net, ctx.hsm,
                             ctx.hss, ctx.het_clip, ctx.data_std, device)
        empirical = ctx.empirical_neighbours(si, path_mask, n_emp=args.n_emp)

        def corr(emp_s, mod_s):
            v = ocean & np.isfinite(emp_s) & np.isfinite(mod_s)
            return pcorr(emp_s[v], mod_s[v])

        r_ang = corr(directional_spread(empirical, ocean),
                     directional_spread(members, ocean))
        r_mag = corr(magnitude_spread(empirical, ocean),
                     magnitude_spread(fused, ocean))
        r_all = corr(vector_spread(empirical, ocean),
                     vector_spread(fused, ocean))
        fmean = np.mean(fused, axis=0).astype(np.float32)
        rmse, ang = accuracy(fmean, target, ocean, ctx.data_std)
        rows.append((r_ang, r_mag, r_all, rmse, ang))
        print(f"{src_f:>7} {cov:>6.1f} {r_ang:>+9.3f} {r_mag:>+8.3f} "
              f"{r_all:>+10.3f} {rmse:>8.4f} {ang:>7.1f}")

        maps = {
            "angle": (directional_spread(empirical, ocean),
                      directional_spread(members, ocean), r_ang),
            "magnitude": (magnitude_spread(empirical, ocean),
                          magnitude_spread(fused, ocean), r_mag),
            "overall": (vector_spread(empirical, ocean),
                        vector_spread(fused, ocean), r_all),
        }
        out = os.path.join(args.out_dir, f"uncertainty_frame{src_f}.png")
        render(maps, ctx.land_np, src_f, cov, out)

    arr = np.array(rows)
    m, s = arr.mean(0), arr.std(0)
    print(f"\nMEAN±STD over {len(rows)} frames:")
    print(f"  r_angle     = {m[0]:+.3f} ± {s[0]:.3f}")
    print(f"  r_magnitude = {m[1]:+.3f} ± {s[1]:.3f}")
    print(f"  r_overall   = {m[2]:+.3f} ± {s[2]:.3f}")
    print(f"  RMSE (m/s)  = {m[3]:.4f} ± {s[3]:.4f}")
    print(f"  angle (deg) = {m[4]:.1f} ± {s[4]:.1f}")
    print(f"\nmaps saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
