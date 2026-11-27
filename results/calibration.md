# Step 2.3 — Uncertainty calibration

**Date:** 2026-11-27 · `benchmark` instance, H1's split (seed 0) · CPU only, $0 ·
`scripts/calibration.py` · `data/processed/calibration.json`

The plan's reason for this step: *"The policy consumes uncertainty directly. An
uncalibrated epistemic estimate is worse than none."* That is the right framing and it
makes the result below a Phase 6 constraint rather than an appendix.

**Headline: none of the three uncertainty estimates ranks error, and selecting the
confident predictions makes accuracy worse.**

| source | ρ(predicted sd, realised error) | risk–coverage gain at 25% |
|---|---:|---:|
| GP predictive variance (closed form) | **+0.015** | **−0.019** (positive on 3/11) |
| independent-trunk ensemble, 5 members | **−0.048** | **−0.128** (positive on 2/11) |
| shared-trunk ensemble, 5 heads | **+0.019** | **−0.051** (positive on 2/11) |

All three are zero to within their spread. The GP's per-assay range is −0.074 … +0.186,
so no assay shows a usable signal either.

## The metric had to be fixed before any of this could be read

The first pass measured error as the absolute residual against a z-scored truth, and
reported the independent ensemble's uncertainty as **anti**-correlated with error at
−0.204 — an apparently dramatic finding.

It was an artifact of the loss. These heads are trained with a **pairwise ranking loss**,
which fixes the *order* of the outputs and leaves their scale and offset entirely free. The
independent ensemble's members came out with prediction spreads around 2.5 while the
shared-trunk one sat near 0.06, so an absolute residual measured arbitrary output scale far
more than it measured error. Replacing it with rank error — `|rank(pred) − rank(truth)|/n`,
which is scale-free and is the error notion a ranking model is actually optimising — moves
the number from −0.204 to −0.048, i.e. from "anti-calibrated" to "no signal".

Sixth instance in this project of an instrument artifact read as a finding, and the first
one caught *before* it reached a write-up rather than after.

## Risk–coverage: declining to act does not help

A planner may decline to act on its least confident candidates, so the operational question
is whether accuracy rises as coverage falls. It does not:

| coverage | GP mean ρ |
|---|---:|
| most confident 5% | +0.395 |
| most confident 25% | +0.408 |
| all variants | +0.427 |

Accuracy is *flat to falling* as the model's own confidence rises. Two assays illustrate
the spread rather than contradicting it: `F7YBW8 (Aakre)` does behave as intended
(+0.355 → +0.662 at 25% → +0.818 at 5%), while `CAPSD` and `SPG1` run the other way. On 8
of 11 assays, keeping only the confident quarter is worse than keeping everything.

## Shared versus independent trunk: the plan's specific question, answered

The plan asks for this because the shared-trunk ensemble is the variant one is tempted to
ship — it costs one training run instead of five.

| | mean ρ | mean predicted sd |
|---|---:|---:|
| independent trunk, 5 members | **+0.3015** | **2.65** |
| shared trunk, 5 heads | **+0.2986** | **0.065** |
| ratio | — | **0.031** |

**The cheap ensemble is exactly as accurate and reports 3.1% of the dispersion** — it
understates epistemic uncertainty by about **32×**, consistently across all 11 assays
(per-assay ratio 0.017 … 0.098). This is the trap the plan anticipated, and the mechanism
is plain: members sharing a trunk cannot disagree about anything the trunk has already
decided, so what is left is five linear read-outs of one representation.

It is also, on this evidence, **a moot point**: an estimate 32× too small and one of the
right magnitude are equally uninformative about which predictions are wrong, so the saving
is real and buys nothing either way.

## Interval coverage, for the one model with a meaningful scale

The GP is fit with a Gaussian likelihood on standardised y, so unlike the ranking-trained
heads its predictive scale means something. Its nominal 90% interval covers **82.6%** on
average — mildly over-confident — but the per-assay range is **0.549 … 0.993**, so it is
over-confident on `GFP` and `HIS7` and badly under-confident on `PHOT` and `D7PM05`. There
is no single correction that would fix both ends.

## What this constrains

- **H3 and the exploration policy.** The plan's uncertainty-guided exploration assumes an
  epistemic estimate that identifies where the model is ignorant. On this reward model that
  estimate does not exist yet, so a Phase 6 policy must either not depend on it or the
  dependence must be reported as resting on an uncalibrated signal.
- **Do not spend five training runs on a deep ensemble.** It buys dispersion of the right
  magnitude and no more information than the one-run variant.
- **The GP's variance is free**, arriving from the same Cholesky factor as its mean, so it
  costs nothing to keep reporting even while it is uninformative — and it is the only one
  of the three whose scale can be checked at all.

**Not done:** calibration of the *depth-aware* head's uncertainty specifically, and
calibration under the matched-literature protocol. Both reported as not done rather than
estimated.
