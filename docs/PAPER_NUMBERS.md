# The paper's three models: final numbers and methods facts

One page to write the paper from. Every number is from the frozen benchmark
(`benchmark/ocean_bench_v1.npz`, 40 cases, 2 h Dubins-style track, time-varying
observations, 3749 scored cells, seed 20260829) unless marked otherwise, and every
table traces to a `.pt` in `benchmark/` stamped with the benchmark's MD5.

RMSE convention throughout: **vector magnitude**, `sqrt(mean(du² + dv²))`. The
per-component convention is smaller by exactly `sqrt(2)`; state whichever is used,
because confusing the two cost this project a week.

---

## The one-line story

Three ways of putting physics into a reconstruction model, tested on the same task:

| model | philosophy | outcome |
|---|---|---|
| **CorrDiff** | no physics — learn everything, lean on temporal priors | **wins** |
| **DistAttn** | soft physics — a curl/div penalty (λ=0.002) + observation attention | middle |
| **Stream** | hard physics — exactly divergence-free by construction | **provably capped** |

Hard physics caps performance because the constraint is wrong: surface currents
are not divergence-free (geostrophic flow is; ageostrophic flow is not; the
divergent part carries 14% of this field's energy, and Fablet et al. 2024
independently report ~47% ageostrophic content). Soft physics does nothing
measurable (controlled experiment, §vorticity). No-physics + temporal priors wins.

## Headline table (realistic task, each model at its best configuration)

| model | RMSE ↓ | angle RMS (rad) ↓ |
|---|---|---|
| **CorrDiff** (priors, 1 h cutoff) | **0.0618** | **0.6842** |
| DistAttn (full track) | 0.0738 | 0.7875 |
| Stream (full field, full track) | 0.0908 | 0.9068 |

- CorrDiff vs DistAttn: −0.0121, CI [−0.0196, −0.0049] — **significant**
- CorrDiff vs Stream: −0.0291, CI [−0.0360, −0.0223] — **significant**
- DistAttn vs Stream: −0.0170, CI [−0.0255, −0.0083] — **significant**

## Uncertainty (the paper's core claim)

Same benchmark. Each model then gets its **own** split-conformal factor (fitted on
20 cases, coverage verified on the held-out 20) — "calibrated" alone is cheap, so
the fair comparison is **sharpness at matched calibration**: who needs the
narrowest intervals to be honest.

| model | CRPS ↓ | coverage, raw (→0.90) | own factor | coverage, calibrated | interval width ↓ |
|---|---|---|---|---|---|
| **CorrDiff** (timed factor) | **0.0242** | **0.908** | **1.009** | 0.920 | **0.174** |
| DistAttn | 0.0296 | 0.737 | 1.621 | 0.898 | 0.178 |
| Stream | 0.0400 | 0.502 | 3.141 | 0.903 | 0.255 |

All three pairwise CRPS gaps significant (corrdiff−distattn −0.0054
CI [−0.0095, −0.0020]; corrdiff−stream −0.0158; distattn−stream −0.0103).

The row to build the paragraph on: **CorrDiff arrives calibrated** — raw coverage
0.908 against the 0.90 target, and its residual conformal factor is 1.009, i.e.
the shipped timed factor is already right to within 1%. DistAttn's intervals need
inflating 1.6× and Stream's 3.1× before they are honest, and even then Stream's
honest intervals are 46% wider than CorrDiff's. DistAttn calibrates to a
respectable width (0.178) — its problem is accuracy, not spread shape.

Fitted factors are shipped as `DISTATTN_SIGMA_SCALE_TIMED = 1.621` and
`STREAM_SIGMA_SCALE_TIMED = 3.141` in `config.py`.

Stated choice for the methods section: CRPS is computed from (mean, σ) under a
Gaussian assumption, identically for every model, rather than from raw ensembles
(whose member counts differ, 10–20). The v1 table computed CorrDiff's CRPS on its
calibrated σ and the others on raw σ; the replication set recomputes all models on
calibrated σ, which is the apples-to-apples version.
→ `results_uncertainty_final.pt` (all four arms re-verified locally from the
saved arrays)

"Best configuration" is itself a measured protocol, not a favour to CorrDiff: the
1 h discard significantly *hurts* the prior-less models (DistAttn +15.5%, GP
+10.9%), so each model runs at its own optimum.

## Replication (independent set — run this section at a reviewer)

Everything above replicates on `ocean_bench_v1b`: 40 fresh cases, **zero frame
overlap** with v1 (verified), fresh generation seed, fresh diffusion seeds, and —
critically — the 1 h cutoff and all calibration factors **fixed in advance**, so
these numbers are clean of any selection-on-test:

| model | RMSE (v1 → v1b) | CRPS | coverage@90, shipped factor applied blind |
|---|---|---|---|
| **CorrDiff** (1 h) | 0.0618 → **0.0566** | **0.0228** | **0.9171** |
| DistAttn | 0.0738 → 0.0728 | 0.0286 | 0.8943 |
| Stream (full field) | 0.0908 → 0.0872 | 0.0359 | 0.9168 |

Identical ranking, all pairwise gaps significant again (corrdiff−distattn
−0.0162 CI [−0.0232, −0.0095]; corrdiff−stream −0.0306), and the cutoff-selected
configuration got *better* on unseen cases — the opposite of selection inflation.

**The shipped conformal factors hold blind**: 2.1801 / 1.621 / 3.141 fitted on
v1, applied untouched to v1b, give coverage 0.917 / 0.894 / 0.917 against the
0.90 target. That is the calibration claim, validated out-of-sample.

→ `results_replication.pt`, benchmark `ocean_bench_v1b.npz`

## RePaint: the honest footnote (measured, not assumed)

RePaint ties CorrDiff on accuracy for the **third time on a third independent
set** (v1b: −0.0021, CI [−0.0079, +0.0028]; CRPS also tied) — the tie is beyond
doubt. Measured cost (Titan Xp, paper configs): CorrDiff 1.2 s/field, RePaint
49.7 — a **41×** gap. (Both prior folklore numbers were wrong: CorrDiff's ~15 s
by 12×, RePaint's ~4 min was its stride-1 worst case.)
What separates them is **calibration**: even granted its own conformal fit within
v1b, RePaint covers 0.863 — the worst of the four — while CorrDiff covers 0.917
using a factor fitted on different data. RePaint's spread has the wrong shape to
calibrate well; CorrDiff's does not. For a paper whose deliverable is honest
uncertainty for a planner, that is the principled reason CorrDiff is the headline
model and RePaint is related work, and it is measured rather than asserted.

## The supporting results, in the order a reader will ask

**1. Discard observations older than 1 h (models with temporal priors only).**
−8.2% RMSE, CI [−0.0102, −0.0010], and 34% narrower intervals at matched 90%
coverage. A genuine interior optimum — 0.75 h and 1.25 h are both worse. CorrDiff
gains −8% and RePaint −14%; Stream, though it carries priors, is **tied** (+1.0%,
ns) — the discard is safe for it but not beneficial, so Stream runs the full
track. Prior-less models are actively harmed (DistAttn +15.5%, GP +10.9%).
Controlled for spatial extent with matched-count controls.
→ `results_age_calibration.pt`

**2. Calibration depends on the observation process.** The conformal factor fitted
for simultaneous observations (1.6787) leaves intervals **2.3× too narrow** on 2 h
collection (needs 3.9142). At the 1 h cutoff the required factor is 2.2006 —
essentially the shipped `CORRDIFF_SIGMA_SCALE_TIMED = 2.1801`.
→ `results_age_calibration.pt`

**3. A vorticity loss term does nothing (the soft-physics test).** Fine-tuned
CorrDiff ±λ·vorticity on byte-identical batches (determinism verified
bit-for-bit). Tied on everything including vorticity RMSE itself; training loss
improved 9% and did not transfer. Without the λ=0 control this would have been
misreported as a significant eddy-recall win.
→ `results_vorticity.pt`

**4. The divergence-free constraint has a provable floor (the hard-physics test).**
Helmholtz projection is orthogonal, so no divergence-free field is closer to the
truth than `divfree(truth)`: **0.0613 RMSE** on held-out frames. A *perfect* Stream
would beat CorrDiff's 0.0665 by only 8%. Caveat kept honest: the floor is 35% of
Stream's squared error, so it is the ceiling, not the whole explanation of the gap.
(The projection quality behind this is verified: the training pickle removes 99% of
divergence and moves vorticity 1%.)

**5. Stream was being run wrong, by everyone.** Two shipped defaults degraded it:
`full_field=False` (output exactly divergence-free, +12.5% RMSE) and `n_draws=1`
(a single noisy draw returned as the mean, +3.8%). Together **+18.8%**. Both fixed
2026-08-31; any Stream number from before then understates it. Stream's floor and
defaults are separate facts: even with both defaults fixed (0.0908), it trails.
→ `results_stream_fullfield.pt`, `results_stream_defaults.pt`

**6. The Okubo–Weiss eddy metric is biased by divergence — and the obvious fix is
boundary-condition dependent.** Adding a curl-free field cannot change vorticity
but drives the raw metric from 0.458 to 0 (pinned by a test). The projection-based
correction gives opposite model rankings under periodic vs Neumann boundary
conditions, so no eddy ranking is reported. Report **vorticity RMSE** instead — it
needs no decomposition. This applies equally to the collaborator's eddy-IoU.
→ `docs/EDDY_METRIC_BIAS.md`

## Methods facts (for the methods section)

| | CorrDiff | DistAttn | Stream |
|---|---|---|---|
| parameters | 16.90 M (14.97 diffusion + 1.93 mean CNN) | 17.78 M | 28.89 M (14.96 direction + 13.93 magnitude) |
| parameterization | v-prediction | ε-prediction | x₀-prediction |
| loss | 1 term (Min-SNR-γ MSE) | 3 terms | 4 + 2 terms (two networks) |
| training data | `data_raw_chrono` | `data_interp` (PCHIP, sub-hourly) | `data_divfree_chrono` (projected) |
| epochs | 200 | 150 (warm-started from Sam's base) | 80 + 25 |
| temporal priors (13/25 h) | yes | no | yes |
| observation timestamps used | no | yes (age tokens; age-weighted loss never trained) | no |
| inference | 20 draws, 50 DDIM steps | 10 draws, stride 10 | 20 draws, dpmpp, + magnitude net + Helmholtz recombination |
| measured s/field (Titan Xp) | **1.2** | 16.1 | 0.3 |
| calibration | split conformal, 2.1801 (timed) | 1.621 (fitted here) | 3.141 (fitted here) |

Loss equations, weights, and references: `docs/loss_doc/loss_functions.docx`.

## Claims to avoid

- **"CorrDiff is the most accurate model we built."** RePaint (excluded from the
  paper) ties it: 0.0595 vs 0.0618, CI [−0.0080, +0.0028]. Say "the most accurate
  of the three presented," or lean on calibration + speed (measured: 1.2 vs
  49.7 s/field), where
  CorrDiff's advantage over RePaint is real.
- **Any eddy-recall or eddy-IoU ranking**, ours or the collaborator's, until the
  metric's divergence bias is handled (§6).
- **"The eddy metric correction fixes it."** Withdrawn — boundary-condition
  dependent.
- **"Stream's constraint explains its whole gap."** It explains the ceiling
  (0.0613 of 0.1039); the rest is the model.
- **Model blending.** Tested and closed: at each model's best configuration,
  corrdiff+distattn gains nothing (+0.0002, 68% of held-out splits — noise) and
  corrdiff+repaint is borderline (+0.0031, 94% of splits, interval crosses zero).
  Not a result.

## Simple framing for Stream (recommendation)

Present Stream as *the hard-constraint arm of the physics question*, not as a
pipeline to be defended. One sentence for the machinery: "a second network
restores the speed information the constraint removes, and the output is
reprojected." Its complexity then reads as evidence about the constraint — even
with two networks patching around it, the constraint still costs 12.5–19% — rather
than as engineering to apologise for. Details to an appendix.
