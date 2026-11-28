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

## 6.4 The predictor class, measured — and a correction to how I first reported it

Step 2.4 went on to reproduce **Kermut-sequence** on this project's own cache: an RBF
kernel on the mean-pooled 1280-d vectors, `m(x) = alpha*f0(x) + beta` with `f0` the summed
log-likelihood ratio at the mutated positions, exact GP, hyperparameters by marginal
likelihood, no structure kernel. An exact GP is free at this scale — the protocol caps
training at 300 per assay, so the Cholesky is microseconds.

The totals and the depth-free skill tell different stories, and only the second is
informative:

| model | total ρ | **within-order ρ** | of which depth |
|---|---:|---:|---:|
| reward head, depth-aware, 266k params | +0.409 | **+0.161** | **+0.248** |
| frozen ridge + mutation count | +0.390 | +0.300 | +0.090 |
| frozen ridge + count + zero-shot | +0.444 | +0.352 | +0.093 |
| GP, constant mean | +0.375 | +0.330 | +0.045 |
| GP, zero-shot mean (Kermut-sequence) — **the registered metric** | +0.428 | +0.371 | +0.057 |
| GP, zero-shot mean + order term | +0.435 | +0.370 | +0.065 |
| **GP, composite kernel** | **+0.479** | **+0.417** | +0.062 |

All eleven assays positive in every GP row. The composite kernel is Kermut's own
composition, `k = π·k_subs + (1−π)·k_seq`, with the **substitution one-hot** standing in
for the structure kernel this project has no ProteinMPNN for yet. The mixing weight is
learned through the marginal likelihood and genuinely mixes: π averages **0.410** across
assays with a range of 0.025 … 0.898, so neither component is redundant.

Its gains are concentrated where a purely additive model was already strong —
`Q6WV12` **+0.408 → +0.580**, `Q8WTC7` +0.267 → +0.391, `D7PM05` +0.321 → +0.407,
`GFP` +0.335 → +0.416 — which is the expected place for it, since `Q6WV12` is one of the
four assays where the additive null beat the reward head outright at Step 2.2 (+0.518
against +0.388). The composite now beats that null on its own best assay.

**The learned mixing weight picks the right model per assay, from the training
likelihood alone.** π is the weight on the substitution kernel, fitted by marginal
likelihood on ≤300 training variants with no sight of the test set:

| assay | π on substitutions | additive null | composite | composite − null |
|---|---:|---:|---:|---:|
| Q8WTC7 | **0.898** | +0.347 | +0.391 | +0.044 |
| Q6WV12 | **0.806** | **+0.518** | **+0.580** | +0.062 |
| GCN4 | 0.770 | +0.030 | +0.164 | +0.134 |
| D7PM05 | 0.758 | +0.373 | +0.407 | +0.034 |
| HIS7 | 0.417 | +0.394 | +0.515 | +0.121 |
| GFP | 0.357 | +0.276 | +0.416 | +0.140 |
| F7YBW8 (Aakre) | 0.209 | +0.420 | +0.359 | **−0.061** |
| F7YBW8 (Ding) | 0.114 | +0.334 | +0.656 | +0.322 |
| CAPSD | 0.089 | +0.196 | +0.635 | +0.438 |
| PHOT | 0.066 | +0.313 | +0.790 | +0.477 |
| SPG1 | 0.025 | +0.362 | +0.354 | −0.008 |

π is **high exactly where the additive null was strong** — `Q6WV12` is the assay whose null
beat the reward head +0.518 to +0.388 at Step 2.2, and it gets π = 0.806 — and **low
exactly where the sequence kernel wins outright**: `PHOT` at π = 0.066 scores +0.790.
`ρ(π, composite − null) = −0.273`, which is the right sign: where the model leans on
substitutions, the null is already close to optimal and there is less left to win.

And on those assays the composite does not merely reproduce the null, it beats it —
`Q6WV12` +0.580 against +0.518, `Q8WTC7` +0.391 against +0.347 — so the sequence kernel is
still contributing underneath. **9 of 11 assays beat the additive null**, against 7 of 11
for the reward head, which is the `frac_assays_beating_null` condition flipping from FAIL
to PASS. The two that do not are `F7YBW8 (Aakre)` at −0.061 and `SPG1` at −0.008.

This matters beyond the number. Step 2.2 recorded that four substantially additive assays
were "uncomfortable for the project's premise that structured epistasis is there and
exploitable". The discomfort was real but the diagnosis was about the *model*, not the
data: a model that can represent both, and select between them per assay without being
told which, recovers those assays and keeps the rest.

**Replicated on the `planning` instance** — the only one allowed to produce a Phase 6
number — at **+0.483 / +0.422 within-order, 7/7 positive**, against benchmark's
+0.479 / +0.417. The result does not depend on the four firewall-banned assays.

**Sensitivity: which backbone supplies `f0`, and a threshold that turns on it.**
`PREREGISTRATION.md` fixes SaProt as the default — on a *stability* measurement — and
ESM-2 as the comparator. Kermut names ESM-2 for its mean function, so running it is
method-specified rather than test-selected, and it is reported as a sensitivity:

| config | `f0` from SaProt (registered) | `f0` from ESM-2 (sensitivity) |
|---|---:|---:|
| GP, zero-shot mean | +0.428 | **+0.444** |
| GP, composite kernel | +0.479 | **+0.489** |
| within-order, composite | +0.417 | **+0.426** |
| `gp_margin_over_ridge` (threshold **> 0.05**) | **+0.038 FAIL** | **+0.054 PASS** |

**The registered margin condition fails on the registered backbone and passes on the
sensitivity.** That is reported as a failure, because swapping the backbone after seeing
which way each falls is precisely the move the pre-registration exists to prevent. It is
worth stating plainly all the same: one of H1's four conditions turns on a default that
was chosen for a *different target*, and the backbone that wins on fitness is not the one
in force. That is a finding about the default, not a licence to change it mid-gate —
changing it belongs in an amendment, before a re-run, with the reason stated first.

**The registered threshold is read on `gp_zeroshot_mean`, not on the composite.**
`spearman_mean_gp >= 0.55` was registered against Kermut-sequence *as published*, and
reporting +0.479 in its place would be moving the goalpost after the fact. Both numbers
fail 0.55 — +0.428 and +0.479 — so the verdict does not depend on the choice, which is the
cleanest situation available and is why it can be stated plainly.

**Correction.** The commit that first reported the GP compared its +0.371 against the
*head's* +0.161 and called it 2.3× the skill. That comparison is true and it is the wrong
one. Against the **ridge**, which is the baseline that actually matters because it is what
tied with the head at 2.2, the GP buys **+0.071** total and about **+0.02 … +0.03** at
matched prior (+0.330 vs +0.300 with no zero-shot; +0.371 vs +0.352 with it). The
predictor class is a real but small effect, not a doubling.

**What the table actually exposes is this project's own head.** It has the second-highest
total and by far the lowest skill: **61% of its +0.409 is the depth penalty** (+0.248 of
+0.409), against 23% for the ridge and 13% for the GP. Step 2.1 added that penalty because
it lifted the mean from +0.260 to +0.384, and it did — but what it bought was a mutation
counter bolted onto a weak sequence model, and the total hid that a plain ridge was already
reading the sequence almost twice as well. The head was never compared to the ridge on the
depth-free diagnostic until now.

**The zero-shot prior is consistent across both estimators**, which is the one clean
regularity here: +0.052 of skill on the ridge (+0.300 → +0.352) and +0.041 on the GP
(+0.330 → +0.371). As a *prior mean* it is worth about the same as it is as a *feature*,
so Kermut's mechanism is not the source of its advantage either.

---

## 6.5 Lever 4 measured: the structure score adds nothing as a mean term

`scripts/structure_score.py` folded all 13 wild types with ESMFold on the rented box and
ran **ProteinMPNN `--unconditional_probs_only`**, giving p(aa | structure) per site — the
quantity Kermut's structure kernel reads, and its **largest** ablation (−0.065).

Standalone on H1's test set it is a real but unremarkable predictor: mean ρ **+0.226**,
within-order **+0.140** — a shade above SaProt's sequence zero-shot (+0.204 / +0.137) and
below ESM-2's (+0.249 / +0.164). Added as a second mean-function term,
`m(x) = a₁·f0_seq(x) + a₂·f0_struct(x) + β`:

| | seq only | + structure | Δ |
|---|---:|---:|---:|
| GP, zero-shot mean | +0.428 | +0.433 | +0.005 |
| GP, zero-shot mean — within-order | +0.371 | +0.365 | **−0.006** |
| GP, composite kernel | +0.479 | +0.486 | +0.007 |
| GP, composite kernel — within-order | +0.417 | +0.415 | **−0.002** |

**Nothing, and slightly negative on the skill component.** The structural information is
apparently already carried by the sequence terms.

Three reasons this is **not** a refutation of Kermut's structure kernel, stated so the
result is not over-read:

1. **It was used as a mean term, not a kernel.** Kermut compares *sites* through their
   ProteinMPNN distributions and inter-residue distances. A scalar summed over mutated
   positions throws that structure away, and the kernel form is untested here.
2. **The structures are ESMFold predictions of varying quality.** Native-argmax recovery
   runs **14.2% … 49.3%** across the 13 — `GCN4_YEAST_Staller_2018` at 14.2% is a
   transcription-factor activation domain, which is intrinsically disordered, so there may
   be no fold for ProteinMPNN to read.
3. **Kermut's −0.065 is on single-mutant random/modulo/contiguous splits**, not on order
   ≥ 3 extrapolation. Nothing says the component transfers.

### The levers, now that four of five are measured

| lever | gain in **within-order skill** | verdict |
|---|---:|---|
| zero-shot prior (as feature or prior mean) | **+0.04 … +0.05** | worth it |
| composite kernel (substitution identity) | **+0.046** | worth it, largest single gain |
| predictor class (GP over ridge, matched prior) | +0.02 … +0.03 | small but real |
| ESM-2 instead of SaProt for `f0` | +0.012 | sensitivity, not in force |
| ProteinMPNN structure, as a mean term | **−0.002 … −0.006** | **no** |
| encoder fine-tune | +0.003 (published, n=128) | **no**, and never run |

End to end, the reward model went from the head's **+0.409 total / +0.161 skill** to the
composite GP's **+0.479 / +0.417** — skill up **2.6×** — and is still **0.071 short** of
the registered 0.55.

---

## 7. The comparability problem — RESOLVED, and it does not excuse the gap

`scripts/matched_literature.py` runs this project's estimators on the **5 multi-mutant
ProteinGym datasets the fine-tuning literature benchmarks**, under **its** protocol: n
sequences sampled at random for training, 5,000 more for test, averaged over 3 seeds. Two
of those datasets (GRB2, PABP) top out at order 2 and had never been in this corpus.

### 7.1 The implementation is correct

| n | 24 | 32 | 48 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| **ours, `gp_zeroshot`** | **+0.433** | **+0.468** | **+0.506** | **+0.631** | **+0.733** |
| published Kermut-sequence | +0.397 | +0.426 | +0.504 | +0.614 | +0.724 |
| difference | +0.036 | +0.042 | +0.002 | +0.017 | +0.009 |

**Our reproduction matches published Kermut-sequence at every training size, slightly
ahead throughout.** That is the check this whole comparison exists for: the estimator is
not broken, and the shortfall on our own assays is not an implementation bug.

### 7.1b And with ESM-2 — the backbone Kermut actually specifies — it is well ahead

The table above uses SaProt, this project's registered default. Kermut specifies **ESM-2
for both the embeddings and `f0`**, so the ESM-2 row is the faithful reproduction:

| n | 24 | 32 | 48 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| **ours, ESM-2, `gp_zeroshot`** | **+0.500** | **+0.539** | **+0.578** | **+0.663** | **+0.743** |
| ours, SaProt, `gp_zeroshot` | +0.433 | +0.468 | +0.506 | +0.631 | +0.733 |
| published Kermut-sequence | +0.397 | +0.426 | +0.504 | +0.614 | +0.724 |
| ours (ESM-2) − published | **+0.103** | **+0.113** | **+0.074** | **+0.049** | **+0.019** |

Ahead of the published figure at every training size, and by most at the smallest, which is
the regime the registered H1 protocol sits closest to.

> **Correction, 2026-11-28. The paragraph below conflated two different uses of a
> backbone and is withdrawn.** It is kept visible because the error is instructive: two of
> its "three independent measurements" varied only the **zero-shot scorer** `f0`, and the
> third varied the scorer *and* the embeddings together, so none of them isolated the
> representation. §7.1c runs the 2x2 that separates them. The corrected finding is
> narrower: **ESM-2 is the better `f0`; it is not the better representation.**

~~**Three independent measurements now say the same thing about the default backbone.**
Zero-shot masked marginals (+0.249 against +0.204, within-order +0.164 against +0.137), the
GP on H1's own protocol (+0.444 against +0.428, and `gp_margin_over_ridge` passing rather
than failing), and now the matched literature protocol (+0.663 against +0.631 at n = 128).
`PREREGISTRATION.md` fixed SaProt as the default on a **stability** measurement (+0.337
against +0.209 pooled Spearman); on **fitness** the comparator leads everywhere it has been
measured.~~

### 7.1c The 2x2 that separates the representation from the prior

A backbone is used for two unrelated jobs here: supplying the 1280-d **embeddings** the
kernel measures similarity in, and supplying **`f0`**, the zero-shot prior mean. One flag
used to change both at once. `--embed-backbone` now separates them, and both protocols
were re-run as a full 2x2.

**H1's own protocol**, 11 assays, `gp_composite_kernel` (total / within-order):

| embeddings \ `f0` | SaProt `f0` | ESM-2 `f0` |
|---|---:|---:|
| **SaProt emb** | +0.479 / +0.417 | +0.489 / +0.426 |
| **ESM-2 emb** | +0.474 / +0.403 | +0.484 / +0.415 |

**The literature's protocol**, 5 datasets, `gp_zeroshot`, mean Spearman:

| embeddings \ `f0` | n = 24 | n = 32 | n = 48 | n = 128 | n = 512 |
|---|---:|---:|---:|---:|---:|
| SaProt emb / SaProt `f0` | +0.433 | +0.468 | +0.506 | +0.631 | +0.733 |
| SaProt emb / **ESM-2 `f0`** | +0.485 | +0.530 | +0.557 | +0.651 | +0.744 |
| **ESM-2 emb** / SaProt `f0` | +0.468 | +0.496 | +0.540 | +0.650 | +0.740 |
| **ESM-2 emb / ESM-2 `f0`** | +0.500 | +0.539 | +0.578 | +0.663 | +0.743 |

**`f0` is the factor that holds.** ESM-2 is the better prior on both protocols and in every
cell: +0.010 … +0.016 on H1, +0.020 at n = 128 on the literature's. **The representation is
not.** It is worth +0.019 at n = 128 on the literature's five datasets and **−0.005** on
H1's eleven, so the two protocols disagree about it and no general preference is supported.

The apparent representation effect in the withdrawn paragraph was the `f0` effect plus an
assay-set effect, read off a single flag that moved both.

#### One assay carries the whole embedding difference on H1

`gp_constant_mean` drops the prior entirely, so it is the clean read on the representation:

| | SaProt | ESM-2 |
|---|---:|---:|
| all 11 assays | +0.375 | +0.306 |
| excluding `PHOT_CHLRE_Chen_2023` | +0.387 | +0.375 |
| `PHOT_CHLRE_Chen_2023` alone | +0.260 | **−0.381** |

85% of the mean gap is one assay, where ESM-2's embeddings are *anti*-correlated with
fitness under an RBF kernel. The prior mean and substitution kernel mask it completely
(composite: ESM-2 +0.810 against SaProt +0.790 on that assay), which is why it is invisible
in the headline number and why `gp_constant_mean` is worth reporting separately. Across the
other ten the two are a near-tie on the mean, with SaProt ahead on the median.

**The amendment this supports is narrower than the one withdrawn above:** change `f0` to
ESM-2, keep SaProt embeddings. It belongs in the amendment table with its reason *before*
any re-run, not after seeing which side of a threshold it lands on.

### 7.2 The harness control says we are, if anything, pessimistic

| n | 24 | 32 | 48 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| our one-hot | +0.191 | +0.206 | +0.240 | +0.387 | +0.625 |
| their one-hot | +0.255 | +0.268 | +0.314 | +0.473 | +0.664 |
| gap | −0.064 | −0.062 | −0.074 | **−0.086** | −0.039 |

Our one-hot baseline runs **0.04 … 0.09 below** theirs at every n. The offset is consistent
in sign and size, so it is a property of the harness — most plausibly our exact train/test
draw — rather than noise. It cuts **against** us: our numbers are measured on a slightly
harder setup than theirs, and the GP still comes out ahead of published Kermut.

This is why the control was included. Had our one-hot landed *above* theirs, every other
row would have been suspect and that would have been the finding.

### 7.3 So how much of the shortfall is the assay list?

| assay set | composite GP |
|---|---:|
| the 3 of our 11 the literature also benchmarks (CAPSD, GFP, HIS7) | **+0.522** |
| the 8 it never uses | **+0.463** |
| all 11 | +0.479 |

**Our assay list is harder, and it accounts for about 0.06 — not for the 0.071 shortfall.**
Excluding `GCN4_YEAST` — the registered secondary, justified at Step 2.1 on sample size
alone (88 training variants against the protocol's 300, and a tuned RidgeCV finds no
in-sample signal in it) — the remaining ten score **+0.510**, which is **0.040 short** of
0.55.

**The comparability question is answered and it does not rescue H1.** The implementation
reproduces the published method; the assay list is somewhat harder and explains part of
the gap; and after both are accounted for the model is still short of the registered
threshold. That is a cleaner failure than Step 2.2's, because none of the three available
excuses — broken code, an unfair assay list, a mis-set threshold — survives contact with
the measurement.

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
