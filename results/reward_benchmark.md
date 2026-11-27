# Step 2.2 — H1: **FAIL.** Do not proceed to Phase 3.

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

1. **Encoder — the binding constraint.** Every configuration tried reads the same 1280-d
   mean-pooled SaProt vector. A linear model on it matches a trained 266k head (+0.019).
   Within-order ρ ceilings at ~+0.21 across all configurations. Mean-pooling 735 residues
   into one vector is the prime suspect: it discards which position changed, and Step 1.4
   already measured that per-residue structure is load-bearing.
2. **Adapter capacity — acquitted.** Step 1.3 swept d = 64/128/256/512 and found 128
   matches the unprojected 1280. More capacity on a saturated representation is not the
   gap.
3. **Loss — acquitted.** Ranking versus regression was tested; the ridge *is* the
   regression comparator and it performs the same.
4. **Per-assay normalisation — acquitted, and provably so.** Per-assay Spearman is
   invariant under increasing per-assay rescaling, so this cannot be the cause.

**Recommended next step: abandon the pooled cache for the fitness head and re-encode
per-residue**, or fine-tune the encoder. The pooled cache was adopted on a measurement
(`pool_first` cost nothing on *stability*, +0.337 vs +0.329) that was taken on 1,000
Tsuboyama single mutants of 36–72 residue domains and generalised to multi-mutant fitness
on proteins up to 735 residues. **That generalisation is now the leading suspect, and it
was never tested on fitness.** Testing it costs a GPU rental and 170 GiB rather than
2.25 GB, which is why the cheap path was taken first — the cheap path is now exhausted.

---

## Honest accounting of this run

- `benchmark` instance, as the pre-registration requires. 4 of 11 assays are on the banned
  list, so **nothing here may support a Phase 6 claim**.
- The tracker flagged the run as non-reproducible because the working tree was dirty.
- Head init varies the mean by about ±0.012, which does not change any verdict: +0.409
  against 0.55 is not an unlucky draw, and 7/11 against 8/11 (needed for 0.70) is one
  assay away, which init noise does not bridge for the four assays the null wins by
  0.01–0.13.
