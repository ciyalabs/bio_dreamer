# Phase 2 — H1: **PASS** at +0.574 (2026-11-28). Phase 3 is unblocked.

> **This file opens with the Step 2.2 FAIL, which is superseded but kept.** The verdict
> moved twice and the intermediate readings are part of the record, not noise to tidy
> away. Nothing below this banner was edited to match the outcome.

**Final gate**, on `gp_structure_kernel` — a GP with Kermut's composite kernel (eq. 4) and
an alignment-based zero-shot prior mean — over the same 11 `benchmark` assays and split
seed 0:

| condition | threshold | 2.2 head | 2.4 GP | **2.5 completed** | |
|---|---|---:|---:|---:|---|
| `spearman_mean` | ≥ 0.55 | +0.409 | +0.479 | **+0.574** | **PASS** |
| `margin_over_additive_null` | > 0.05 | +0.085 | +0.155 | **+0.250** | PASS |
| `frac_assays_beating_null` | ≥ 0.70 | 0.636 | 0.818 | **0.909** | PASS |
| `spearman_mean_vs_frozen_regression` | > 0 | +0.108 | +0.178 | **+0.273** | PASS |

Within-order ρ — the depth-free diagnostic — went **+0.161 → +0.417 → +0.500**.

**No threshold was edited at any point in Phase 2.** The verdict changed because two
experiments registered under Step 2.4 were run for the first time: `msa_zeroshot` (never
run) and `mpnn_site_features` as a *kernel* rather than the scalar mean term that was
tested and measured −0.002.

**What did it:** the zero-shot prior. Every `f0` this project used read one sequence in
isolation; a published alignment-based prior is worth **+0.096**, against **+0.049** for
the structure kernel.

**Against our own interest, in full:** `structure_kernel_margin` **FAILS** at +0.0494
against its registered > 0.05 (passes on within-order at +0.0658; the registered metric is
the total). Step 2.4's `spearman_mean_gp`, defined on Kermut-*sequence*, **remains a FAIL**
at +0.525. The assay-list difficulty offset — six published methods score 0.12 … 0.27
lower on these eleven assays than on the other 206 — contributed **nothing**, because H1
clears 0.55 unadjusted. `f0` is now read from ProteinGym's published archive rather than
computed here, which is what Kermut does but is a dependency this project now owns. The
structures are ESMFold, not the AlphaFold2 ones Kermut used.

Full account: `diagnostics/finetune_scoping.md` §9, `PREREGISTRATION.md` Phase 2 gate.
Reproduce with `python -I scripts/gp_reward.py --instance benchmark --zs-backbone
msa-transformer --structure-kernel` then `python -I scripts/h1_verdict_gp.py --source
data/processed/gp_reward_zs-msa-transformer_structkern.json`.

---

# Step 2.2 — H1: **FAIL.** Do not proceed to Phase 3. *(superseded 2026-11-28)*

**Run:** 2026-11-27, CPU only, `benchmark` instance, 11 assays, 943,411 unseen order ≥ 3
variants. Reproduce with `python -I scripts/h1_benchmark.py`.

No threshold was edited. The verdict was written by the script whichever way it fell.

---

## The gate

| condition | threshold | measured | |
|---|---|---:|---|
| `spearman_mean` | ≥ 0.55 | **+0.409** | **FAIL** |
| `margin_over_additive_null` | > 0.05 | +0.085 | PASS |
| `frac_assays_beating_null` | ≥ 0.70 | **0.636** (7/11) | **FAIL** |
| `spearman_mean_vs_frozen_regression` | > 0 | +0.108 | PASS |

`PREREGISTRATION.md` is explicit about the consequence: *"do not proceed to Phase 3. The
reward model is the foundation; a planner on a weak oracle generates confident nonsense."*

---

## The finding that matters more than the verdict

| method | mean ρ | median |
|---|---:|---:|
| **reward head** (266k params, depth-aware, ranking loss) | **+0.409** | +0.404 |
| **frozen ridge + mutation count** | **+0.390** | +0.425 |
| additive null (one coefficient per substitution) | +0.324 | +0.347 |
| frozen ridge, no mutation count | +0.301 | +0.355 |

**The whole reward head buys +0.019 over `RidgeCV(pooled_embedding ++ mutation_count)`** —
and loses to it on median. Adapter, shared trunk, RankNet loss, censored-pair masking,
learned depth penalty: +0.019, and it wins on only 7 of 11 assays.

`spearman_mean_vs_frozen_regression` passes at +0.108 **only because the registered
baseline omits mutation count.** Give the ridge that one extra column and the margin
collapses by a factor of five. The condition was written to isolate what world-model
training adds; at Step 2.2 there is no world model, so what it actually measured is that
the reward head is barely distinguishable from a linear model on the same features.

This reframes the diagnosis. Step 2.1 concluded the head needed depth information. It
did — but so does a ridge, and once both have it they perform the same. The binding
constraint is not the head's architecture, its loss, or its capacity. **It is the frozen
pooled representation.**

---

## Per assay

| assay | head | additive null | ridge | ridge+depth | best | |
|---|---:|---:|---:|---:|---|---|
| CAPSD_AAV2S_Sinai_2021 | **+0.659** | +0.196 | +0.178 | +0.572 | head | |
| GFP_AEQVI_Sarkisyan_2016 | **+0.535** | +0.276 | +0.338 | +0.482 | head | |
| SPG1_STRSG_Wu_2016 | **+0.499** | +0.362 | +0.366 | +0.365 | head | BANNED |
| F7YBW8_MESOW_Ding_2023 | +0.490 | +0.334 | +0.519 | **+0.652** | ridge+d | BANNED |
| D7PM05_CLYGR_Somermeyer_2022 | **+0.448** | +0.373 | +0.308 | +0.425 | head | |
| HIS7_YEAST_Pokusaeva_2019 | +0.404 | +0.394 | +0.428 | **+0.484** | ridge+d | BANNED |
| F7YBW8_MESOW_Aakre_2015 | +0.390 | **+0.420** | +0.355 | +0.365 | null | BANNED |
| Q6WV12_9MAXI_Somermeyer_2022 | +0.388 | **+0.518** | +0.355 | +0.400 | null | |
| Q8WTC7_9CNID_Somermeyer_2022 | +0.335 | **+0.347** | +0.178 | +0.236 | null | |
| PHOT_CHLRE_Chen_2023 | +0.226 | +0.313 | +0.414 | **+0.484** | ridge+d | |
| GCN4_YEAST_Staller_2018 | **+0.120** | +0.030 | −0.129 | −0.180 | head | |

**The additive null wins outright on four assays**, `Q6WV12` by a wide margin (+0.518
against +0.388). Those assays are substantially additive, and a model built to capture
epistasis is losing to one that assumes none. That is the single most uncomfortable row
in this table for the project's premise, because ProteinDreamer's whole case rests on
structured epistasis being there and being exploitable.

The head's wins are real where they happen — `CAPSD` +0.659 against a null of +0.196 is a
3.4× margin on 30,963 unseen variants — but they are concentrated, not general.

---

## What is not done, and is not estimated

- **ProSST / VenusREM zero-shot** (`vs_zeroshot`): needs those checkpoints downloaded and
  run. Not attempted, not guessed at.
- **Literature baselines** (`vs_literature`): ProteinNPT 0.56, MULTI-evolve 0.61, Kermut
  0.62, Cerebra 0.65 are **cited from `PREREGISTRATION.md`, measured on their own assay
  sets**, not recomputed on these 11. Our +0.409 is not strictly comparable to them: the
  assay set differs, and 4 of our 11 are banned-list assays chosen for Phase 6 relevance
  rather than for literature overlap. The honest statement is that we are below the
  bottom of that band on our own set, and a strict comparison needs the same assays.

Both gaps weaken the `vs_literature` and `vs_zeroshot` boxes but not the verdict: the two
failing conditions are measured on our own data against our own baselines.

---

## Diagnosis, in the order the pre-registration prescribes

It says: *encoder (D2 result), adapter capacity, loss (ranking vs regression), per-assay
normalisation.* The evidence now points hard at the first and acquits the rest.

1. ~~**Encoder — the binding constraint.** Every configuration tried reads the same 1280-d
   mean-pooled SaProt vector. A linear model on it matches a trained 266k head (+0.019).
   Within-order ρ ceilings at ~+0.21 across all configurations. Mean-pooling 735 residues
   into one vector is the prime suspect: it discards which position changed, and Step 1.4
   already measured that per-residue structure is load-bearing.~~
   **WITHDRAWN** — see "What the diagnosis missed" below. The pooling suspicion was tested
   and cleared (`diagnostics/positional_readout.md`), and the fallback to "then it is the
   encoder" does not survive either.
2. **Adapter capacity — acquitted.** Step 1.3 swept d = 64/128/256/512 and found 128
   matches the unprojected 1280. More capacity on a saturated representation is not the
   gap.
3. **Loss — acquitted.** Ranking versus regression was tested; the ridge *is* the
   regression comparator and it performs the same.
4. **Per-assay normalisation — acquitted, and provably so.** Per-assay Spearman is
   invariant under increasing per-assay rescaling, so this cannot be the cause.

~~**Recommended next step: abandon the pooled cache for the fitness head and re-encode
per-residue**, or fine-tune the encoder.~~ Both halves of this recommendation were acted on
and both were wrong. The per-residue re-encode was run for $0.70 and changed nothing
(`diagnostics/positional_readout.md`); the fine-tune was costed and is near-worthless at
this data scale (`diagnostics/finetune_scoping.md`).

---

## What the diagnosis missed

**The four-item list above enumerates components this model *has*.** Capacity, loss,
normalisation and readout were each tested and acquitted, and the conclusion treated the
list as exhaustive: if everything present is innocent, the encoder must be guilty. That
inference is invalid, because the list cannot detect **a component the model lacks
entirely.**

It lacks a **zero-shot likelihood**. Every method above 0.55 on this protocol carries one as
a separate term — Kermut as its Gaussian-process prior mean, ProteinNPT as input features,
PoET as the score itself. `vs_zeroshot` stood unchecked in Step 2.2 from the day the step was
written, and was reported in the row above as "not done rather than estimated". It was the
one box whose contents would have falsified the diagnosis.

**The direct refutation.** Kermut scores **0.62** on this exact protocol, and its sequence
kernel is a squared-exponential kernel on **ESM-2 650M embeddings, mean-pooled across the
length dimension, in R^1280** — the same backbone size, pooling and dimension as
`data/processed/pooled_cache`. A frozen mean-pooled 1280-d vector therefore supports 0.62
here, not 0.409. It is also, in Kermut's own decomposition, the term that carries
**epistasis** — the exact capability this report concluded a pooled vector could not provide.

**And the fine-tune was mis-costed in both directions.** It is not "materially larger than a
$0.70 rental": the relevant recipe is 4 minutes per assay on an H100. What it is, is nearly
worthless at our n — **+0.003** over a frozen pooled head with the same ranking loss at
n = 128 multi-mutants, and **−0.015** at n = 512, against **+0.211** for simply adding a
zero-shot score to the features. The registered cap of 300 sits inside that crossover.

Step 2.4 replaces the recommendation with the lever order that follows from those numbers.
A fine-tune is last on it, on payoff rather than price.

---

## Honest accounting of this run

- `benchmark` instance, as the pre-registration requires. 4 of 11 assays are on the banned
  list, so **nothing here may support a Phase 6 claim**.
- The tracker flagged the run as non-reproducible because the working tree was dirty.
- Head init varies the mean by about ±0.012, which does not change any verdict: +0.409
  against 0.55 is not an unlucky draw, and 7/11 against 8/11 (needed for 0.70) is one
  assay away, which init noise does not bridge for the four assays the null wins by
  0.01–0.13.

---

## Phase 2 outcome (2026-11-27) — this report's verdict superseded, not reversed

H1 still **FAILS** and Phase 3 stays blocked. What changed is the reward model and the
quality of the explanation. `RESEARCH_PLAN.md`'s **Phase 2 gate** is the authoritative
statement; this section records only what it means for the numbers above.

**The 266k head is no longer the reward model.** It was replaced by a Gaussian process:
RBF on the same mean-pooled 1280-d vectors, composed with a substitution kernel, with a
zero-shot likelihood as the prior mean. Re-reading H1's four conditions against it
(`scripts/h1_verdict_gp.py`, thresholds read from `plan.yaml`, baselines reused from this
report rather than recomputed):

| condition | head, above | GP, composite |
|---|---:|---:|
| `spearman_mean` ≥ 0.55 | +0.409 ❌ | **+0.479 ❌** |
| `margin_over_additive_null` > 0.05 | +0.085 ✅ | +0.155 ✅ |
| `frac_assays_beating_null` ≥ 0.70 | 0.636 ❌ | **0.818 ✅** |
| `spearman_mean_vs_frozen_regression` > 0 | +0.108 ✅ | +0.178 ✅ |

**The "finding that matters more than the verdict" above has been superseded.** This report
recorded that the whole reward head buys +0.019 over `RidgeCV(pooled ++ mutation_count)`
and loses on median — true, and the right thing to have noticed. But the comparison was
made on *total* Spearman, and total Spearman on this protocol is inflated by mutation
count. On **within-order ρ**, where depth carries no information:

| | total | within-order | of which depth |
|---|---:|---:|---:|
| reward head | +0.409 | **+0.161** | **61%** |
| frozen ridge + count | +0.390 | +0.300 | 23% |
| GP, composite | +0.479 | **+0.417** | 13% |

The head did not merely tie the ridge; it was **worse than the ridge at reading the
sequence** and made the difference up with a mutation counter. That is a sharper and less
flattering statement than the one this report made, and it only became visible once the
ridge was measured on the depth-free diagnostic — which this report did not do.

**The four substantially additive assays are no longer uncomfortable.** This report flagged
that the additive null wins outright on four assays, `Q6WV12` at +0.518 against +0.388, and
read it as uncomfortable for the project's premise. A composite kernel that can weight
substitution identity against sequence similarity, choosing per assay from the training
likelihood alone, scores **+0.580** there and beats the null on **9 of 11** assays. The
discomfort was about the model, not the data.

**And the diagnosis section above is wrong in its conclusion** — see "What the diagnosis
missed". The encoder was acquitted by `diagnostics/finetune_scoping.md`, and a matched-
protocol run reproduces published Kermut-sequence (+0.631 at n = 128 against +0.614), so
the implementation is sound and the shortfall is neither the encoder nor a bug.
