# Step 2.1 — Reward head: censoring decision and first training

**Run:** 2026-11-27. Module `biodreamer/reward/`, 266k trainable parameters.
**No GPU needed** — Steps 1.3 and 1.4 had already cached every latent this used.

---

## The censoring decision, measured rather than assumed

`RESEARCH_PLAN.md` required this written down before training, because "silently
regressing on floored values will corrupt every epistasis estimate downstream." It
offered two options — censored regression or principled exclusion. The measurement says
the two targets need different answers, and that neither option fits the fitness one.

### Fitness: rank, and drop pairs where **both** members are censored

| assay | n | at floor | modal share | detection limit? |
|---|---:|---:|---:|---|
| **D7PM05_CLYGR** | 24,515 | 40.9% | **31.9%** | yes, at 590 |
| **GFP_AEQVI** | 51,714 | 21.4% | 8.1% | yes, at 1.301 |
| **Q8WTC7_9CNID** | 33,510 | 13.5% | 2.7% | yes, at 449 |
| Q6WV12_9MAXI | 31,401 | 15.4% | 0.9% | no — soft tail |
| CAPSD_AAV2S | 42,328 | 0.0% | 0.1% | no |
| PHOT_CHLRE | 167,529 | 0.0% | 0.0% | no |
| GCN4_YEAST | 2,638 | 0.6% | 0.6% | no |

**The modal share is the diagnostic, not the floor fraction.** Q6WV12 has 15.4% of
variants near its minimum but only 0.9% at any single value — a soft tail from genuinely
weak variants. D7PM05 has **31.9% at one exact number**, which is an instrument reading
its limit, not a measurement.

Excluding those variants would discard real information: *this variant is dead* is worth
learning. Regressing on them asserts a precise value the assay never produced. Ranking
resolves it — a pair of two floored variants carries no ordering information and is
dropped; a floored-versus-live pair carries the real signal and is kept. **Censored
points are used for exactly the comparisons they can support.**

Ranking is also what H1 is scored on (per-assay Spearman), and it is scale-free, which
is what makes training across assays with incomparable units legitimate at all.

### Stability: weight by precision, do not censor

`dG_ML` is rail-clipped to [−1, +5], but **0.01% of values reach a rail** — there is
nothing meaningful to censor. What varies is precision: the median 95% CI is 0.132
kcal/mol and **9.7% exceed 1.0**. So stability uses inverse-variance weighted SmoothL1,
which uses every point in proportion to what it is worth. SmoothL1 rather than MSE
because the ddG distribution has a long destabilising tail that squared error would let
dominate the gradient.

---

## First training run

Cross-validated over **held-out proteins**, five folds, 1,000 Tsuboyama mutants across
50 domains. Splitting by mutant would be meaningless here: twenty mutants of one domain
share 95% of their residues, so a random split puts near-duplicates on both sides and
measures memorised protein identity.

| | pooled ρ | per-protein median | mean | p10 | ρ > 0 | ρ > 0.3 |
|---|---:|---:|---:|---:|---:|---:|
| **SaProt** (no 3Di) | **+0.337** | **+0.753** | +0.674 | +0.311 | **100%** | **90%** |
| ESM-2 650M | +0.209 | +0.598 | +0.525 | +0.172 | 92% | 80% |

The pooled figure is much lower than the per-protein median because it mixes domains
with different ddG ranges — which is exactly why the plan asks for the distribution and
not the mean. On a held-out domain the head typically reaches ρ ≈ 0.75.

**This run overturned the Phase 1 gate's backbone conclusion.** That report compared
ESM-2 (+0.445) against SaProt *with* 3Di (+0.304) and called ESM-2 the winner, when
SaProt *without* 3Di was already ahead at +0.459. Trained, SaProt leads on every
statistic. See the correction appended to `phase1_gate.md`. **SaProt stays the default,
run with its structure channel off.**

---

## What this establishes for later phases

**This is the frozen-encoder regression baseline.** H1 registers
`spearman_mean_vs_frozen_regression > 0`, and no world model exists yet, so this run
*is* that comparator. Phase 3's world-model training has to beat **+0.337 pooled /
+0.753 per-protein median** on stability to have added anything.

It is also a floor on what the latent carries: Step 1.3 measured the *unsupervised*
displacement norm correlating with |ddG| at +0.459. A trained head reaching +0.337
pooled on held-out proteins is doing something a distance metric cannot, but the gap is
not large, and Phase 2 should not claim more than it shows.

## What is not done

> **Resolved, 2026-11-27.** All three items below are done — see
> `diagnostics/fitness_head.md`. The corpus is encoded (946,365 pooled vectors) and both
> instances are trained. Headline: the fitness target reaches mean ρ **+0.260** on
> `planning` and **+0.290** on `benchmark`, clear of a 0.10 chance band but **tied with a
> substitution-count predictor at 0.31**, against H1's registered 0.55. The stability
> baseline below is unaffected.

- ~~**The fitness target is untrained.**~~ Trained; the encoding cost ~$3 of A40.
- ~~**Neither the `benchmark` nor the `planning` instance exists yet.**~~ Both exist, with
  11 and 7 assays.
- ~~Those are the first items of the next rental~~ — done, H1 (Step 2.2) remains.

---

## Reproduction

```
python -I scripts/censoring_audit.py     # the decision, ~3 min CPU
python -I scripts/train_reward.py --backbone saprot
python -I scripts/train_reward.py --backbone esm2
```

Writes `data/processed/censoring_audit.json` and `reward_training.json`; tracker runs
under `2.1 / censoring_decision` and `2.1 / reward_training`. 19 tests in
`tests/unit/test_reward.py`, most of them on the censoring behaviour rather than the
network — it is the part most likely to be quietly undone.
