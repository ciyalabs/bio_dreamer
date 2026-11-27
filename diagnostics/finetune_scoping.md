# Re-scoping Phase 2: what a fine-tune could actually buy

**Date:** 2026-11-27 · **Status:** scoping, no new training run · **Cost:** $0

H1 failed at **+0.409** against a registered **0.55**. `results/reward_benchmark.md` recorded
the diagnosis as *the frozen encoder's representation is the ceiling*, and named the next
lever as fine-tuning SaProt or changing backbone. This document checks that conclusion
against the literature before any GPU is rented, because the rental is the expensive part
and the conclusion is the thing that justifies it.

**The conclusion does not survive the check. The recorded diagnosis was wrong, and a
fine-tune is close to the worst-value lever available.**

---

## 1. The recorded diagnosis is contradicted by the baseline we never ran

`Kermut` scores **0.62** on this project's exact protocol — fewer than 300 single and
double mutants for training, unseen order ≥ 3 for test. Its backbone is **frozen** and its
sequence kernel reads **mean-pooled** protein-language-model embeddings: the same input,
at the same pooling, from the same class of encoder this project feeds its head.
`ProteinNPT` reaches 0.56 on frozen embeddings as well.

The match to this project's own features is exact, not analogous. Kermut's sequence kernel
is a squared-exponential kernel on **ESM-2 650M embeddings, mean-pooled across the length
dimension, in R^1280** — the same backbone size, the same pooling, the same dimension as the
vectors in `data/processed/pooled_cache`.

And Kermut assigns that kernel the *opposite* role to the one this project concluded it
could not play. In Kermut's own decomposition the structure kernel is the linear part and
**the mean-pooled sequence kernel is the term that carries epistasis**. This project
concluded from `positional_readout` that a pooled vector was the obstacle to modelling
epistasis. The method that beats us by 0.21 on this protocol uses a pooled vector *as* its
epistasis term.

To be precise about what that does and does not prove: Kermut's 0.62 is **not** from pooled
embeddings alone. It is pooled embeddings **plus a structure kernel plus a zero-shot prior
mean**, and ProteinNPT likewise adds zero-shot features. So the claim is not "pooled
embeddings are worth 0.62". The claim is narrower and still fatal to the recorded
diagnosis: **a frozen, mean-pooled representation is demonstrably not what caps this task
at 0.409**, because the same representation reaches 0.62 once other terms are added to it.

`diagnostics/positional_readout.md` was right that the *readout* does not matter and that
`pool_first` is innocent. The error was the inference drawn from it — that with capacity,
loss, normalisation and readout all acquitted, nothing remained but the encoder. That
enumeration was of the components this model **has**. It could not detect a component the
model **lacks entirely**, and that is where the gap turns out to be.

## 2. The missing component is a zero-shot likelihood

Every method above 0.55 on this protocol carries a pretrained likelihood as a distinct
term. Kermut uses a zero-shot score as the **GP prior mean**; ProteinNPT takes zero-shot
predictions as input features; PoET *is* a likelihood. This project has **never computed
one**, and `vs_zeroshot` has stood unchecked in Step 2.2 since the step was written.

Its measured value, from the one paper that isolates it on multi-mutant ProteinGym data
(Table 5, `bioRxiv 2024.05.28.596156v4`):

| features | n = 24 | n = 48 | n = 128 | n = 512 |
|---|---:|---:|---:|---:|
| one-hot encoding alone | 0.255 | 0.314 | 0.473 | 0.664 |
| one-hot **+ zero-shot score** | **0.607** | **0.622** | **0.684** | **0.763** |
| gain from the zero-shot term | **+0.352** | **+0.308** | **+0.211** | **+0.099** |

The zero-shot term is worth more than every architectural choice this project has made put
together, and it is worth most where labels are scarcest — which is the registered regime.

## 3. What a fine-tune buys, at our data scale, is approximately nothing

The same paper isolates fine-tuning against a frozen head **on the same backbone and the
same ranking loss this project uses** (Table 4, multi-mutant datasets, ESM-2 650M):

| | n = 24 | n = 32 | n = 48 | n = 128 | n = 512 |
|---|---:|---:|---:|---:|---:|
| frozen pooled embedding + linear head, ranking loss ← **our architecture** | 0.429 | 0.447 | 0.503 | 0.648 | 0.773 |
| **all parameters fine-tuned**, wt-marginals + ranking loss | 0.577 | 0.593 | 0.603 | 0.651 | 0.758 |
| gain from fine-tuning | +0.148 | +0.146 | +0.100 | **+0.003** | **−0.015** |

**The fine-tune's advantage closes by n ≈ 128 and inverts by n = 512.** The registered H1
protocol caps training at **300** per assay — inside the crossover. The honest
pre-registered expectation for a fine-tune at our n is therefore **+0.00 to +0.05, and the
sign is not guaranteed**.

Two things follow that are easy to get backwards:

- **The cost was never the obstacle.** Fine-tuning ESM-2 with this recipe takes **4
  minutes** per dataset on an H100 (Table 8) — cheaper than the $0.70 positional-readout
  rental. A fine-tune was rejected here on *expected payoff*, not on price. The earlier
  note that this lever is "materially larger than a $0.70 rental" was wrong about the cost
  as well as about the value.
- **The fine-tune's target is itself additive.** The scoring rule being fine-tuned towards
  (wt-marginals) sums independent per-position log-odds, so it cannot represent epistasis.
  The paper's own authors attribute their one multi-mutant regression against ranking to
  exactly this. Our test set is order ≥ 3 — the pure-epistasis regime — so the mechanism
  is poorly matched to the target even where it does help.

## 4. Structure, and why the 3Di result does not settle it

Kermut's **largest** ablation is its structure kernel: −0.065 mean ΔSpearman when removed,
bigger than the sequence kernel's −0.045 and the zero-shot mean function's −0.026.

`PREREGISTRATION.md` already records that SaProt's 3Di channel **hurt** here: |ddG|
correlation +0.459 → +0.304, because ESMFold reports a confident fold even for variants at
ddG ≤ −2, so the tokens assert "still folded" for destabilised proteins. That finding stands
and is not reopened.

But it does not transfer. Kermut's structure kernel is **ProteinMPNN site-conditioned
amino-acid distributions** — inverse-folding log-probabilities used as a feature — not a
structure token fed into a sequence model. The failure mode measured here (tokens assert a
fold that is not there) is a property of the token channel, not of structural information.
The two routes are different enough that the first does not rule out the second.

## 5. The levers, in measured value per unit cost

This was the order going in, with the gains predicted from the literature:

| # | lever | predicted gain | cost | outcome |
|---|---|---|---|---|
| 1 | **zero-shot score as a ranker** | — (diagnostic) | $0, 1,201 forward passes | **done** — +0.13 of *skill*, §6.1 |
| 2 | **zero-shot as prior mean / feature** | +0.10 … +0.21 | $0, reuses the pooled cache | **done** — **+0.055**, and 0.00 on 9 of 11 assays, §6.2 |
| 3 | MSA-based zero-shot instead of single-sequence | +0.04 on top of 2 | hmmer/mmseqs2 already installed | not started |
| 4 | ProteinMPNN site distributions as a feature | up to +0.065 | 11 wild types, already folded | not started |
| 5 | **fine-tune the encoder** | **+0.00 … +0.05, sign not guaranteed** | 4 min/assay on an H100 | **deprioritised** |

Levers 1 and 2 were run in the same session as this document, for nothing, and the
prediction for lever 2 was **too optimistic by a factor of four** on the mean and by
infinity on the median. The published +0.211 is from random splits with multi-mutants in
training; under strict order-extrapolation the same idea is worth +0.055, concentrated
entirely in two assays.

**That result demotes levers 3 and 4 as well.** Both are refinements of the same move —
a better pretrained prior, bolted onto the same ridge. Lever 2 has now shown that this
model absorbs about +0.05 from a prior and then saturates, with six of eleven assays not
moving in the third decimal. A better prior plausibly widens the rescue (lever 3 would
likely help GCN4 further) without touching the nine assays that did not respond.

**So the lever that matters is the one that was never on the list: the predictor class.**

MSA zero-shot above single-sequence zero-shot is the paper's Table 9, multi-mutant column:
MSA Transformer **0.534** against ESM-2 wt-marginals **0.493**. Note what that row also
says — **ESM-2's zero-shot score alone, with no training at all, scores 0.493 on their
multi-mutant sets, above this project's entire trained head at +0.409.** The assay sets
differ, so the comparison is indicative and not a result; §6 is the controlled version.

## 6. Measured on our own 11 assays

`scripts/zeroshot_baseline.py`, `benchmark` instance, H1's exact split (seed 0, locked),
all **943,211** unseen order ≥ 3 variants. SaProt 650M, the project's own default backbone,
no new training. 1,201 forward passes, $0, local GPU.

### 6.1 Zero-shot alone is not better than what we have, and most of it is depth

| rule | total ρ | median | **within-order ρ** | depth-attributable |
|---|---:|---:|---:|---:|
| `wt-marginals` | **+0.321** | +0.279 | **+0.131** | +0.190 |
| `masked-marginals` | +0.204 | +0.118 | **+0.137** | +0.067 |

The two rules differ by 0.117 in total ρ and by **0.006 in within-order ρ**. They have the
same skill; `wt-marginals` only looks better because more of its score is a mutation
counter. This is the fifth time in this project a number has had to be decomposed before it
could be read, and the standing rule caught it: the scoring rule sums one mostly-negative
log-odds term per substitution, so deeper variants score lower *by construction*. Step 2.1
measured that counting mutations alone reaches **0.31** on this test set — so a zero-shot
total of +0.321 is, on its own, consistent with no sequence-specific skill whatsoever.

**So the published framing does not transfer.** Zero-shot alone is worth +0.13 of skill
here, not the 0.49 the multi-mutant literature reports for ESM-2 — because that literature
scores random splits with multi-mutants in training, while this is strict order-extrapolation.

### 6.2 As a feature it clears its threshold — and the mean is not the honest summary

| features | mean ρ | median |
|---|---:|---:|
| frozen ridge + mutation count | +0.390 | +0.425 |
| **+ zero-shot column** | **+0.444** | — |
| `zeroshot_feature_margin` | **+0.055** | **+0.002** |

**Threshold: `> 0.05`. Measured: `+0.055`. PASS — by 0.005, and the median delta is
+0.002.** The per-assay table is the result; the mean is almost an artifact of two rows:

| assay | ridge + count | + zero-shot | Δ |
|---|---:|---:|---:|
| **GCN4_YEAST** | **−0.180** | **+0.163** | **+0.342** |
| **PHOT_CHLRE** | +0.484 | **+0.748** | **+0.263** |
| F7YBW8 (Ding) | +0.652 | +0.682 | +0.029 |
| F7YBW8 (Aakre) | +0.365 | +0.387 | +0.021 |
| Q8WTC7 | +0.236 | +0.240 | +0.004 |
| SPG1 | +0.365 | +0.367 | +0.002 |
| Q6WV12 | +0.400 | +0.402 | +0.002 |
| D7PM05 | +0.425 | +0.426 | +0.000 |
| HIS7 | +0.484 | +0.485 | +0.000 |
| CAPSD | +0.572 | +0.565 | −0.007 |
| **GFP_AEQVI** | +0.482 | **+0.425** | **−0.057** |

**Excluding GCN4 and PHOT, the mean delta over the other nine assays is −0.0006.** Not
small — *zero*. Four assays gain more than 0.02, one loses more than 0.02, and six are
unchanged to three decimals.

So the zero-shot prior is **not a general lift. It is a floor.** It rescues the two assays
where the supervised pipeline had already failed, and does nothing where the pipeline
works:

- `GCN4_YEAST` is the assay Step 2.1 called **unlearnable** — a tuned RidgeCV found no
  in-sample signal in its 88 training variants, and it ranked backwards on every run. The
  zero-shot prior fixes its sign. That is exactly the behaviour Kermut's prior mean is for:
  when the supervised signal is absent, fall back on the pretrained likelihood.
- `PHOT_CHLRE` had no within-depth signal at all under the supervised head and sat inside
  its chance band. Its zero-shot ρ is **+0.812** with a within-order ρ of **+0.795** — the
  one assay where the pretrained likelihood is simply right, and not via depth.
- `GFP_AEQVI` is the cost: 37,853 variants, strong supervised signal, and the prior
  actively hurts it.

**Sensitivity, pre-specified as such:** `masked-marginals` is ProteinGym's standard for
ESM-family models and is the registered rule. Because the two rules have equal skill and
differ only in depth content, and the ridge already carries an `order` column, the
prediction is that `wt-marginals` adds *less* as a feature despite scoring higher alone.
Reported rather than selected on.

### 6.3 What this does and does not buy

+0.444 against a threshold of **0.55** and a published band of **0.56–0.65**. The missing
component was real, it was worth finding, and it is **not enough** — it closes about a third
of the distance to the threshold on the mean, and none of it on nine of eleven assays.

Which changes the conclusion a second time, and this is the part to carry forward:

> **0.55 is not reachable by adding terms to this model.** Kermut is not "our ridge plus a
> zero-shot column". It is a Gaussian process whose kernel composes a mean-pooled sequence
> similarity with a site-level structural one, with the zero-shot score as its prior
> *mean function* — a different estimator, not our estimator with another feature. The
> remaining distance is in the **predictor class**, which is the one thing Step 2.2's
> diagnosis never examined, because a ridge and a 266k MLP agreeing told us the features
> were saturated **for that class of predictor** and nothing more.

## 7. The comparability problem, which is now blocking

Every published number above is a mean over **its own assay list**. This project's +0.409 is
a mean over **our** 11. Those are not the same quantity, and the registered 0.55 threshold
was set from published means — so part of the 0.141 shortfall may be assay difficulty
rather than method quality, and nobody can currently say how much.

`vs_literature` was recorded as "not done rather than estimated", which was the right call.
It is now a **prerequisite** rather than an optional extra: without it, neither a pass nor a
fail of H1 can be attributed. Two ways to close it, in preference order:

1. Run this project's head on the assay list a published method reports, and compare
   per-assay rather than mean-to-mean.
2. Run a published method's released code on our 11 assays. Kermut is the practical choice:
   it is ~10 minutes for three CV schemes on four datasets against ProteinNPT's 400 hours,
   and its code is public.

## 8. What this changes in the plan

- **Struck:** "the encoder's representation is the ceiling". A frozen mean-pooled 1280-d
  ESM-2 vector supports 0.62 on this protocol in Kermut's hands; ours yields 0.409. The gap
  is not in what the encoder discards.
- **Struck:** fine-tuning as the next lever, and the "materially larger than $0.70" costing
  that accompanied it. It is 4 min/assay on an H100 and worth +0.003 at our n.
- **Done, and it passed its threshold:** a zero-shot prior, Step 2.4, +0.055 against > 0.05.
- **Recorded against us:** that pass is carried by 2 of 11 assays and is −0.0006 on the
  other nine. A +0.055 that clears a 0.05 bar by 0.005 on a median of +0.002 is a pass on
  the letter of the pre-registration, and the per-assay table is the result. Reported this
  way rather than as "H1's margin condition met".
- **The new binding constraint, and the one the original diagnosis structurally could not
  see:** the **predictor class**. Step 2.2 found a ridge and a 266k MLP scoring the same
  and concluded the *features* were saturated. What that actually shows is that the
  features are saturated **for ridge-like predictors**. A ridge and a small MLP on the same
  pooled vector are near-equivalent function classes, so their agreement is close to
  tautological — it was never evidence about the features.
- **Promoted:** `vs_literature` from an unchecked box to a precondition. +0.444 against 0.62
  is a gap of 0.18 between *different methods on different assay lists*, and nobody can
  currently apportion it.

### The next experiment, and why it is the right one

Reimplement the estimator rather than the features: a **Gaussian process with a composite
kernel** on the vectors already in `data/processed/pooled_cache`, with the zero-shot score
as its prior mean function. Every input it needs now exists locally. It is the minimal
change that tests the predictor class instead of the feature set, it is the published
method that scores 0.62 on this protocol, and it costs no rental — Kermut runs three CV
schemes on four datasets in about ten minutes, against ProteinNPT's 400 hours.

Run it against our own 11 assays **and** against the assay list it reports, so the
comparison and the comparability prerequisite close in the same experiment.

### Sources

- Cerebra Low-N 1M/2M→3M+ benchmark, and the Kermut / MULTI-evolve / ProteinNPT figures on
  that protocol: <https://www.biorxiv.org/content/10.64898/2026.09.24.753701v1.full>
- Fine-tuning against frozen heads, the zero-shot feature ablation, run times and base-model
  zero-shot levels: <https://www.biorxiv.org/content/10.1101/2024.05.28.596156v4.full>
- Kermut's composite kernel, its zero-shot prior mean and its component ablations:
  <https://proceedings.neurips.cc/paper_files/paper/2024/file/34547650b2ca69d91f3b3c3ae8b21962-Paper-Conference.pdf>
