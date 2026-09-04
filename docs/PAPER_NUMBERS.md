# The paper's three models: final numbers and methods facts

One page to write the paper from. Every number is from the frozen benchmark
(`benchmark/ocean_bench_v1.npz`, 40 cases, 2 h Dubins-style track, time-varying
observations, 3749 scored cells; benchmark generation seed 20260829, evaluation seed 20260830) unless marked otherwise, and every
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
| Stream (full field, full track) | 0.0865 | 0.8663 |

- CorrDiff vs DistAttn: −0.0121, CI [−0.0196, −0.0049] — **significant**
- CorrDiff vs Stream: −0.0247, CI [−0.0310, −0.0188] — **significant**
- DistAttn vs Stream: −0.0127, CI [−0.0210, −0.0043] — **significant**

Stream's row reflects two 2026-09-02 changes: the spread term dropped (§6) and the
sampler default re-swept from 6 steps to 2 (§7c). Against the previously shipped
configuration it was 0.0908 / 0.9068. Its pairwise gaps versus CorrDiff and DistAttn are unchanged in
sign and significance.

## Uncertainty (the paper's core claim)

Same benchmark. Each model then gets its **own** split-conformal factor (fitted on
20 cases, coverage verified on the held-out 20) — "calibrated" alone is cheap, so
the fair comparison is **sharpness at matched calibration**: who needs the
narrowest intervals to be honest.

| model | CRPS ↓ | coverage, raw (→0.90) | own factor | coverage, calibrated | interval width ↓ |
|---|---|---|---|---|---|
| **CorrDiff** (timed factor) | **0.0242** | **0.908** | **1.009** | 0.920 | **0.174** |
| DistAttn | 0.0296 | 0.737 | 1.621 | 0.898 | 0.178 |
| Stream | 0.0389 | 0.435 | 3.162 | 0.894 | **0.207** |

All three pairwise CRPS gaps significant (corrdiff−distattn −0.0054
CI [−0.0095, −0.0020]; corrdiff−stream −0.0160 CI [−0.0194, −0.0126];
distattn−stream −0.0106 CI [−0.0144, −0.0068] — the last two recomputed against
the 2026-09-02 Stream weights).

The row to build the paragraph on: **CorrDiff arrives calibrated** — raw coverage
0.908 against the 0.90 target, and its residual conformal factor is 1.009, i.e.
the shipped timed factor is already right to within 1%. DistAttn's intervals need
inflating 1.6× and Stream's 3.2× before they are honest, and even then Stream's
honest intervals are 19% wider than CorrDiff's. DistAttn calibrates to a
respectable width (0.178) — its problem is accuracy, not spread shape.

Fitted factors are shipped as `DISTATTN_SIGMA_SCALE_TIMED = 1.621` and
`STREAM_SIGMA_SCALE_TIMED = 3.165` in `config.py`.

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
| Stream (full field) | 0.0865 → 0.0834 | 0.0329 | 0.9101 |

Identical ranking, all pairwise gaps significant again (corrdiff−distattn
−0.0162 CI [−0.0232, −0.0095]; corrdiff−stream −0.0268 CI [−0.0335, −0.0207]), and the cutoff-selected
configuration got *better* on unseen cases — the opposite of selection inflation.

**The shipped conformal factors hold blind**: 2.1801 / 1.621 / 3.165 fitted on
v1, applied untouched to v1b, give coverage 0.917 / 0.894 / 0.910 against the
0.90 target. That is the calibration claim, validated out-of-sample.

→ `results_replication.pt`, benchmark `ocean_bench_v1b.npz`

## RePaint: the honest footnote (measured, not assumed)

RePaint ties CorrDiff on accuracy for the **third time on a third independent
set** (v1b: −0.0021, CI [−0.0079, +0.0028]; CRPS also tied) — the tie is beyond
doubt. Measured cost (Titan Xp, paper configs): CorrDiff 1.2 s/field, RePaint
49.7 — a **41×** gap. (Both prior folklore numbers were wrong: CorrDiff's ~15 s
by 12×, RePaint's ~4 min was its stride-1 worst case.)
**Correction, 2026-09-03: the calibration argument against RePaint does not hold.**
The earlier claim — that RePaint covers only 0.863 and its spread has "the wrong
shape to calibrate" — came from a factor fitted *within* v1b, a single split. Under
the same protocol every other model gets (fit on the first half of v1, apply
untouched to all of v1b), RePaint reaches **0.9109** with factor 2.4513, against
CorrDiff 0.917, Stream 0.910 and DistAttn 0.894. It calibrates as well as the rest.

So what separates them is **cost, not calibration**: 49.7 s/field against CorrDiff's
1.2, a **41×** gap, for accuracy that ties. That is a weaker reason to prefer
CorrDiff than the one previously written here, and the paper should say the honest
version — RePaint is excluded on compute, and because two of the three presented
models already come from collaborators, not because its uncertainty is worse.

→ `results_repaint_calibration.pt`

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

**3. A vorticity loss term does nothing on CorrDiff (the soft-physics test).** Scoped to CorrDiff — on Stream the same term does transfer, see §7. Fine-tuned
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
defaults are separate facts: even with every default fixed (0.0865), it trails.
→ `results_stream_fullfield.pt`, `results_stream_defaults.pt`

**6. Stream is the worst of the three at vorticity, and one of its own loss
terms is why.** Vorticity RMSE — the metric §6 below says to report instead of any
eddy statistic — was never measured for Stream. It is 0.01822 against CorrDiff's
0.01219 and DistAttn's 0.01385, i.e. only **3.5% better than predicting zero
vorticity everywhere** (0.01889); vorticity correlation 0.518 vs 0.725. Not
ensemble smearing: a single draw carries 1.89× the true vorticity amplitude, and
averaging 20 brings it to 1.06, so the amplitude is right and the pattern is wrong.

A controlled ablation attributes much of this to the **spread term** (the 4th
direction-loss term). Matched at 1000 fine-tune steps, turning it off improves
vorticity RMSE and correlation (v1 −0.00101/+0.0436; v1b −0.00111/+0.0492, both
significant), improves **calibrated** CRPS, narrows 90% intervals 8% (0.2351 vs
0.2563), and — the point — raises r(σ, actual error) from 0.225 to 0.289, which is
the quantity the term exists to raise. Cost: RMSE tied on v1, +0.7% on v1b.
A weight sweep rules out mis-weighting: over lambda_spr in {0, 0.1, 0.3, 1.0} every
metric it should help degrades monotonically as the weight rises, including its own
target (r(σ,error) 0.289 → 0.225). Strongest evidence needs no retrain: the shipped checkpoint is bit-identical to
`StreamFn_Cond_x0_mag_spread.pt` (ep 48), a spread-term fine-tune of
`StreamFn_Cond_x0_mag.pt` (ep 78, no spread). That **superseded predecessor beats the
shipped model** on vorticity (corr 0.562 vs 0.518), r(σ,error) (0.296 vs 0.223) and width (0.2297 vs
0.2550), RMSE tied. Vorticity and r(σ,error) replicate significantly on v1b; the
calibrated-CRPS gap does NOT replicate (same direction, interval crosses zero) — and it lands where
the matched fine-tune did, from a model 0.26 weight-distance away. Effect persists unchanged at 20,000 steps on
CUDA (vort corr +0.0479, r(σ,err) +0.0681, both significant), so it holds across a short
fine-tune, a 20× longer one on different hardware, and two separately-trained historical
models. Recommendation is to retrain without the term and re-fit the conformal factor.
Beware: on *raw* σ the comparison favours keeping the term, an artefact of both
arms being under-dispersed (~0.50 raw coverage) — report CRPS on calibrated σ.
→ `docs/STREAM_LOSS_ABLATIONS.md`

**7. A vorticity loss term on Stream works and is far too weak to matter.**
Unlike on CorrDiff (§3, null on everything), it transfers: significant and
monotone in λ against a matched λ=0 control (λ=1 +0.0048 correlation, λ=5 +0.0101),
tied on RMSE and angle. But correlation moves 0.560 → 0.570 across a 5× change in
λ against a 0.165 gap to CorrDiff, while the training term itself falls 8–18×.
→ `docs/STREAM_LOSS_ABLATIONS.md`

**7b. Stream's per-cell uncertainty map does not beat a single scalar.** Replacing
sigma with its own per-case ocean mean — zero spatial information — gives the shipped
Stream model **narrower** intervals (0.2131 vs 0.2551) and better calibrated CRPS
(0.0364 vs 0.0376) at equivalent coverage (0.9021 vs 0.9032). Separately,
`STREAM_UNC_SMOOTH_SIGMA` was **0.8** and far from optimal: width and CRPS improve
monotonically out to sigma 12.8, which was costing the shipped model ~16% width. It is
now **3.2** — chosen by a criterion fixed in advance (max r(sigma, error) subject to
width no worse than 0.8's, at matched coverage), and r peaks at 3.2 rather than
improving monotonically, so 3.2 is an interior optimum on that criterion even though
width alone would keep going. The predecessor does retain
spatial content (blur beats constant; r(σ,error) peaks 0.304 vs 0.231), so this is a
third way the spread term hurts. **Now tested on all three: the failure is Stream-specific.** CorrDiff's per-cell map
beats a scalar on width AND coverage simultaneously (0.1446 at 0.8959 vs 0.1465 at
0.8843) with r(σ,error) 0.447; DistAttn wins at matched coverage (0.1837 at 0.9176 vs
0.1954 at 0.9202), r 0.414. Both roughly double Stream's 0.231. So per-cell calibrated
uncertainty IS claimable for the headline model, and Stream's failure is a fourth
strike against it. → `docs/STREAM_LOSS_ABLATIONS.md` §2b,
`results_uncertainty_spatial_value.pt`

**7c. Stream's sampler default was tuned against the old weights and cost 6%.**
`STREAM_DPMPP_STEPS = 6` was documented as a validated "sweet spot"; that validation
predates the 2026-09-02 weights. Re-swept over {1,2,3,4,5,6,10,16}, **2 is the optimum**
and beats 6 on RMSE (0.0865 vs 0.0919), vorticity RMSE (0.01654 vs 0.01743), vorticity
correlation (0.571 vs 0.562), calibrated CRPS (0.0347 vs 0.0368) and interval width
(0.216 vs 0.230) — replicated on v1b — at a third of the compute. 1 step lowers RMSE and
CRPS *further* but degrades vorticity (corr 0.528): that is where the metric is bought by
blurring, and it is the cleanest illustration in the project of why RMSE alone cannot
select a configuration. The conformal factor is **coupled to both the step count and the
smoothing sigma** (2.909 at 6 steps/sigma 0.8, 3.304 at 2/0.8, 3.165 at 2/3.2 as shipped)
— re-fit whenever either changes. → `docs/STREAM_LOSS_ABLATIONS.md`

**7d. No auxiliary loss term in any of the three models earns its place.** All four
ablated against matched controls: Stream's spread term is **harmful** (removed),
Stream's magnitude TV term is **inert** (adequately powered — the trainable head moved
4.5% — and null on everything including the σ roughness it exists to suppress),
CorrDiff's vorticity term is **null**, and DistAttn's curl/divergence term is **inert at
its shipped 0.002 and significantly harmful at 1000×** — at λ=2.0, which moves the
weights 144× further, divergence RMSE (its own target) worsens +0.000035
CI [+0.000002, +0.000066]. Three of the four had never been tested.

DistAttn's is now settled at full strength rather than by fine-tune: two arms warm-started
from the pre-timecond base (epoch 136), whose trainer had **no such term**, trained 40
epochs each on one GPU at one seed, differing only in λ. They ended **15% apart in
relative weight distance** — 13,700× the original 8-epoch pair — and are **tied on all six
metrics**, with the control reaching the better validation loss and divergence RMSE again
nominally worse *with* the term. So the term does nothing at any weight tested across
three orders of magnitude, and this null has power behind it.
**The physics ladder's soft rung is now directly evidenced**, not just inferred from
CorrDiff. → `docs/LOSS_TERM_ABLATIONS.md`

**8. The Okubo–Weiss eddy metric is biased by divergence — and the obvious fix is
boundary-condition dependent.** Adding a curl-free field cannot change vorticity
but drives the raw metric from 0.458 to 0 (pinned by a test). The projection-based
correction gives opposite model rankings under periodic vs Neumann boundary
conditions, so no eddy ranking is reported. Report **vorticity RMSE** instead — it
needs no decomposition. This applies equally to the collaborator's eddy-IoU.
→ `docs/EDDY_METRIC_BIAS.md`

**9. Stream's magnitude network is the one component that pays — and it is the power
check on every null above.** Ablating only the fusion step (`fuse_coupled`
monkeypatched; conditioning, sampler, seeds and masking byte-identical) costs
**+16% RMSE** (0.08651 → 0.10352), vorticity correlation 0.571 → 0.503 and calibrated
CRPS 0.0344 → 0.0435, all three significant. The sharpest number is the conformal
factor: without the magnitude net the raw spread needs **11.5×** inflation to reach
90% coverage against 3.16× with it, and the calibrated intervals are 47% wider at
equal coverage. That is the magnitude-collapse premise the two-network design was
built on, confirmed directly for the first time. It also answers the obvious
objection to §7: the same 40-case paired bootstrap that returns TIED for four loss
terms returns three significant effects here, so those nulls are properties of the
terms, not of an underpowered test.
→ `docs/LOSS_TERM_ABLATIONS.md` §5

**10. CorrDiff's two runtime dials, measured for the first time.** `n_draws = 20` is
confirmed rather than inherited: halving to 10 is significantly worse on **all five**
metrics, doubling to 40 buys nothing but a marginal vorticity RMSE gain at twice the
cost. **The conformal factor is not portable across ensemble sizes** — it runs 5.053 /
4.116 / 3.805 / 3.688 at n = 5 / 10 / 20 / 40, so a factor fitted at 20 and applied at
5 under-covers by a third; all four predictors now warn on the mismatch. The
`sensor_noise` channel does work as designed, with the cleanest dose-response in the
project (every significant effect roughly doubles from 5% to 10% observation noise),
but the effect is small — 2.6% better calibrated CRPS at the largest level the model
supports. CorrDiff is robust to sensor noise at these levels because 10% of the field
std is only 16% of its own RMSE.
→ `docs/DEFAULTS_AND_DIALS.md`

**11. Inference cost, measured for all six models on one device.** Stream 0.2 s/field,
CorrDiff 2.6, DistAttn 38.1, RePaint 133.6 (MPS; VCNN and GP are both under 0.25).
**Stream is the cheapest of the three paper models by 13x** — the hard-physics model is
also the fastest, two DPM-Solver++ steps against CorrDiff's 50. So the physics ladder
has a cost axis as well as an accuracy axis: Stream is capped on accuracy *and* 13x
cheaper, CorrDiff buys the accuracy at 13x the cost. DistAttn is the expensive one at
15x CorrDiff and also the least accurate of the three, so it is dominated on both axes.
The RePaint ratio holds across devices — 51x on MPS against 41x on the Titan Xp — so the
cost argument for excluding it does not depend on the machine.
→ `docs/DEFAULTS_AND_DIALS.md` §4

**12. The spread term's removal no longer rests on an early checkpoint.** The shipped
no-spread Stream net is epoch 78 of the lineage that predates the term, which left "no
spread term" confounded with "less training", and every fine-tune ablation started from
a model that had already absorbed it. A direction net trained from **random init** for
120,000 steps that never saw the term is **tied on all six metrics** (RMSE 0.08714 vs
0.08651, CI [−0.000328, +0.001735]; divergence RMSE identical at 0.00946). Not shipped —
it is nominally worse on five of six, so there is no gain to justify refitting the factor
and recomputing every Stream number. → `docs/STREAM_LOSS_ABLATIONS.md`

**13. DistAttn's ensemble size is the one shipped default that is wrong.** CorrDiff's 20
and Stream's 20 were both confirmed (below costs real accuracy, above buys nothing), but
`DISTATTN_DEFAULT_N_DRAWS = 10` is inherited from the collaborator's evaluation, and 20
significantly improves **four of six** metrics — calibrated CRPS −0.00093, vorticity
correlation +0.0135, vorticity RMSE −0.00020, divergence RMSE −0.00012 — with the
calibrated interval 14% narrower. 40 is not better than 20. Recorded rather than changed:
it doubles DistAttn's cost (to ~29x CorrDiff), `DISTATTN_SIGMA_SCALE_TIMED` was fitted at
n=10 and would need refitting, and Sam's published numbers use 10.
→ `docs/DEFAULTS_AND_DIALS.md` §3b

**14. Both retrains came back tied, which is the result.** The two open training-run
questions are closed and neither changes a shipped model. DistAttn's curl/divergence term:
tied on all six from a term-naive base with the arms 15% apart in weight space (§7d).
Stream's spread term: a direction net trained from random init for 120,000 steps that never
saw the term ties the shipped epoch-78 checkpoint on all six (§12). Read together, the two
strongest objections to the loss-term story — "you only fine-tuned from a model that already
had the term" and "your no-spread checkpoint just stopped early" — are both answered
empirically, at full training length, with measured power.
→ `docs/LOSS_TERM_ABLATIONS.md`, `docs/STREAM_LOSS_ABLATIONS.md`

## Methods facts (for the methods section)

| | CorrDiff | DistAttn | Stream |
|---|---|---|---|
| parameters | 16.90 M (14.97 diffusion + 1.93 mean CNN) | 17.78 M | 28.89 M (14.96 direction + 13.93 magnitude) |
| parameterization | v-prediction | ε-prediction | x₀-prediction |
| loss | 1 term (Min-SNR-γ MSE) | 3 terms | 3 + 2 terms (two networks; the 4th, spread, was dropped 2026-09-02) |
| training data | `data_raw_chrono` | `data_interp` (PCHIP, sub-hourly) | `data_divfree_chrono` (projected) |
| epochs | 200 | 150 (warm-started from Sam's base) | 78 + 25 |
| temporal priors (13/25 h) | yes | no | yes |
| observation timestamps used | no | yes (age tokens; age-weighted loss never trained) | no |
| inference | 20 draws, 50 DDIM steps | 10 draws, stride 10 | 20 draws, dpmpp 2 steps, + magnitude net + Helmholtz recombination |
| measured s/field (Titan Xp) | **1.2** | 16.1 | 0.3 |
| calibration | split conformal, 2.1801 (timed) | 1.621 (fitted here) | 3.165 (fitted here) |

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
- **CRPS on raw sigma, when comparing models or arms with different spread
  widths.** Every model here is under-dispersed raw, so a wider raw spread wins
  for the wrong reason — it flipped the sign of the Stream spread-term result.
  Compute CRPS on each arm's conformally calibrated sigma.
- **Any claim that Stream represents rotational structure well.** It is the worst
  of the three on vorticity RMSE, 3.5% better than predicting zero vorticity (§6).
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
