# Step 2.2 follow-up — mean-pooling is not H1's constraint. The encoder is.

**Run:** 2026-11-27. A40 rental, 43 min of GPU, **$0.70 all in**. 11 assays, 66.3M
residues, test variants capped at 30,000 per assay. Reproduce with
`bash scripts/window_b.sh`, then `python -I scripts/positional_readout.py`.

---

## Verdict: the hypothesis is refuted, and the sign is backwards

H1 failed, and the diagnosis in `results/reward_benchmark.md` blamed the representation:
every configuration reads the same 1280-d **mean-pooled** SaProt vector, which cannot
express *which* position changed. This rental tested that directly — four readouts of the
**same frozen latents**, under the same head, recipe, seed and test subsample, with only
the input vector differing.

| readout | mean ρ |
|---|---:|
| `pooled` (the baseline) | +0.410 |
| `mut_mean` — latents at the variant's own mutated positions | +0.413 |
| `displacement` — (variant − wild type) at those positions | +0.408 |
| `pooled+displacement` | +0.413 |

**Spread across all four: 0.005.** The init-to-init spread of a single configuration is
±0.012, so these are one number. Reading the mutated positions, which is exactly what
mean-pooling cannot do, changes nothing.

### The length test, which the hypothesis should have passed

If mean-pooling were the constraint, its damage would grow with sequence length — one
vector summarising 735 residues loses more than one summarising 93. So `displacement`
should gain most on the longest proteins.

| assay | length | pooled | displacement | gain |
|---|---:|---:|---:|---:|
| CAPSD_AAV2S_Sinai_2021 | **735** | **+0.658** | +0.562 | **−0.096** |
| SPG1_STRSG_Wu_2016 | 448 | +0.503 | +0.490 | −0.013 |
| GCN4_YEAST_Staller_2018 | 281 | +0.120 | +0.116 | −0.004 |
| Q8WTC7_9CNID_Somermeyer_2022 | 238 | +0.335 | +0.359 | +0.024 |
| GFP_AEQVI_Sarkisyan_2016 | 238 | +0.534 | +0.480 | −0.054 |
| D7PM05_CLYGR_Somermeyer_2022 | 235 | +0.448 | +0.371 | −0.077 |
| Q6WV12_9MAXI_Somermeyer_2022 | 222 | +0.388 | +0.381 | −0.007 |
| HIS7_YEAST_Pokusaeva_2019 | 220 | +0.408 | +0.377 | −0.031 |
| PHOT_CHLRE_Chen_2023 | 118 | +0.234 | **+0.408** | **+0.174** |
| F7YBW8_MESOW_Ding_2023 | 93 | +0.490 | +0.568 | +0.078 |
| F7YBW8_MESOW_Aakre_2015 | 93 | +0.390 | +0.378 | −0.012 |

**Spearman(length, gain) = −0.484; Pearson = −0.575.** The predicted correlation was
*positive*. Pooling helps **more** on long proteins, and the single worst case for
`displacement` is the longest protein in the corpus.

This is not a null result that leaves the question open. It is a result pointing the
opposite way from the hypothesis, which is much stronger evidence against it.

### Why pooling probably wins on long proteins

A mean over 735 residues is a low-variance estimate of a global property; the latents at
5 mutated positions are a high-variance estimate of a local one. On a long protein the
averaging is doing useful noise reduction, and the local signal is a smaller share of
what determines fitness. The representation was never throwing away the information — the
information was not sufficient to begin with.

### The one real exception

`PHOT_CHLRE` gains **+0.174** from displacement, far outside noise, and it is the assay
that had *no* within-depth signal under pooling (within-order ρ −0.033). So reading the
mutated positions does recover something there. It is one assay of eleven, it is 118
residues rather than a long one, and it does not move the mean. Worth remembering if PHOT
matters later; not worth a conclusion.

---

## What this settles

**The cap did not distort anything.** `pooled` scores **+0.410** here against H1's
**+0.409** on the full test set — the 30,000-variant cap reproduces the uncapped number to
0.001. That validates the cost-saving choice after the fact.

**`pool_first` is exonerated.** It was adopted on a stability measurement (+0.337 vs
+0.329) taken on 1,000 single mutants of 36–72 residue domains and generalised, untested,
to multi-mutant fitness on proteins up to 735 residues. That generalisation was flagged as
the leading suspect in `results/reward_benchmark.md`. **It was a correct generalisation.**
The 2.25 GB pooled cache stays; the 512× storage saving costs nothing.

**The frozen encoder is the ceiling, and H1's failure is structural.** The pre-registered
diagnosis order — encoder, adapter capacity, loss, per-assay normalisation — now has every
item resolved. Capacity was swept at Step 1.3 (d=128 matches 1280). The loss was tested
against a ridge, which ties. Per-assay normalisation is provably irrelevant. And the
encoder's *readout* is now shown not to matter either, which leaves the encoder's
**representation** itself.

**So the next step is not cheaper than this one.** Fine-tuning SaProt on the fitness task,
or moving to a backbone trained for variant effects, is the remaining lever. That is a
materially larger undertaking than a $0.70 rental, and it should not start before Phase 2
is re-scoped against what a fine-tune could plausibly buy.

---

## Honest accounting

- **Cost:** $0.70, 43 min GPU. The experiment was sized down from an initial "170 GiB and
  2.5 h" estimate by two observations: pooling only ever saved storage, never compute, so
  per-residue costs the same; and the hypothesis concerns mutated positions, so caching
  k × 1280 per variant plus one wild-type latent per assay suffices (~3 GB).
- **The smoke test paid for itself.** `python -I` isolates `sys.path`, and `biodreamer`
  was not installed on the pod — caught in 3 minutes rather than after the model download.
- **The cap-aware resume fired correctly**, re-encoding the two assays cached at cap 200
  rather than silently mixing caps across the corpus.
- 4 of 11 assays are banned-list, so nothing here may support a Phase 6 claim.
- The tracker flagged the run non-reproducible: dirty working tree.
