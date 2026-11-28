# Phase 3, Steps 3.1–3.2 — World Model B and the collapse gate

**Run:** 2026-11-28, local GTX 1650, `planning` instance, 20 epochs.
Reproduce with `python -I scripts/train_wm_b.py --epochs 20`.

---

## The gate, as registered: FAIL

`latent_effective_rank_ratio` **+0.022** against the registered **≥ 0.50**.

**No threshold has been edited and the remedy ladder has not been applied.** The reason
for both is below, and it is not "the number is close".

### Every loss-based signal said the opposite

On `PHOT_CHLRE_Chen_2023`, over the same run that produced the 0.022:

| signal | value | reads as |
|---|---:|---|
| inverse-dynamics position top-1 | **1.000** | perfect |
| dynamics loss, epoch 0 → 19 | 0.99 → 0.073 | learning |
| one-step error ÷ identity baseline | **0.022** | 45× better than "nothing changed" |
| latent effective rank | **7.3 of 256** | collapsed |

Had Phase 3 been gated on training curves it would have passed without a second look.
This is exactly the case Step 3.2 was written for, and the reason the gate is on the
representation rather than the objective: **a collapsed model has a low prediction loss**,
because predicting a nearly-constant target is easy.

---

## Why the remedy ladder is the wrong response

The registered remedies — a masked-token loss, an EMA target, AC-MTM — all treat
*training* pathologies. Before reaching for them, the input was measured:

| | effective rank | of |
|---|---:|---:|
| frozen encoder states, `PHOT_CHLRE` | **6.6** | 1280 |
| frozen encoder states, `CAPSD_AAV2S` | 14.8 | 1280 |
| frozen encoder states, `GFP_AEQVI` | 21.8 | 1280 |
| world-model latent, `PHOT_CHLRE` | 7.3 | 256 |

**The input is already collapsed, and the adapter is not what collapsed it.** Every
variant of one protein is a near-identical sequence, so the frozen encoder places the
whole assay in a thin shell around the wild type — on PHOT the mean deviation is 13% of
the wild-type vector's own length, on CAPSD 4%. A latent cannot span more directions than
its input contains, so `≥ 0.5 × d_model` is unreachable on single-protein variant data by
**any** model, for reasons that have nothing to do with training. None of the three
remedies adds information that is not in the input.

### The diagnostic that does discriminate

Rank *relative to the input* separates a faithful adapter from a collapsing one, and on
these three assays it separates them cleanly:

| assay | transitions | latent rank | input rank | **preserved** | inverse top-1 | one-step ÷ identity |
|---|---:|---:|---:|---:|---:|---:|
| `PHOT_CHLRE_Chen_2023` | 60,000 | 7.3 | 6.6 | **1.10** | 1.000 | 0.022 |
| `CAPSD_AAV2S_Sinai_2021` | 20,142 | 5.0 | 14.8 | 0.34 | 0.726 | 0.111 |
| `GFP_AEQVI_Sarkisyan_2016` | 3,769 | 4.8 | 21.8 | 0.22 | 0.271 | 0.257 |

Rank preservation, inverse-dynamics accuracy and one-step skill all move together, and all
three track the number of transitions. PHOT, with 60,000, preserves its input rank
completely. GFP, with 3,769, keeps a fifth of it. That is the signature of data
starvation, not of an anti-collapse mechanism failing — and the starved assays are exactly
the ones the cache-coverage gap below created.

**Amendment DECIDED 2026-11-28, and made harder than the number in hand.**
`latent_effective_rank_ratio >= 0.5` (against `d_model`) is recorded as a FAIL at +0.022
and kept. The gate becomes `rank_preserved_vs_input` with **mean ≥ 0.7 and min ≥ 0.5**.

The originally proposed `>= 0.5` would have passed at the measured 0.55 — by 0.05, on a
mean that `PHOT` alone carries, with `GFP` at 0.22. A gate cleared that way is not a gate.
The bar is instead set from what a working adapter was *observed* to do: `PHOT_CHLRE`, the
one assay with ample transitions, preserves **1.10**. Full preservation is achievable here,
so 0.7 / 0.5 is the weakest bar consistent with that. **Both conditions FAIL on the numbers
in this document** (mean 0.55, min 0.22), and the gate is registered in that state.

Two conditions on how it is read, registered in `PREREGISTRATION.md` before the run: on the
**per-residue** state Step 3.1 registers, and **after** the coverage gap below is closed.
See `results/state_representation.md` — going per-residue raises the encoder's rank ceiling
only ~3× (to 26–70 of 1280, against the 640 the original floor needs), which is what rules
out rescuing the original reading rather than replacing it.

---

## Two data problems that shape what Phase 3 can be

### 1. The per-residue state does not exist

Step 3.1 asks for `Z ∈ R^(L×d)`. `data/processed/positional_cache/` stores each variant's
latents only at *its own* mutated positions, plus one full wild type. Reconstructing a
variant as "wild-type rows with the mutated rows substituted in" measures, on CAPSD:

| | |
|---|---:|
| cosine(reconstructed pooled, true pooled) | 0.9991 |
| ‖true − wt‖ | 0.141 |
| ‖reconstructed − wt‖ | 0.023 |
| **fraction of the displacement captured** | **0.052** |
| direction agreement of that displacement | 0.30 |

The cosine is 0.9991 because both vectors are dominated by the wild type. Against the
displacement that actually distinguishes variants, the reconstruction keeps 5% and points
elsewhere. This is Step 1.4's finding — 10.6% of the encoder's movement lands on the
substituted residue — appearing at the pooled level. The option is refused in code rather
than offered behind a flag, because every cosine-based diagnostic would have called it fine.

**So the run above uses pooled states**, which `scripts/encode_corpus.py` explicitly
reserves for the reward head: *"the world model needs the per-residue tensor and must not
read these files."* It is a pipeline validation and a collapse measurement. **It is not
Step 3.1 proper**, and nothing here should be read as one-step accuracy for Step 3.3.

### 2. Transitions need both endpoints, and the cache has only one

The pooled cache was built for the H1 and literature splits, so it holds ≤ 300 order ≤ 2
variants plus all order ≥ 3 — and the excluded low-order variants are precisely the
**parents** of transitions. Of 1,395,965 transitions available in the `planning` instance,
84,000 are usable, and four of seven assays fall below 1,000:

| assay | coverage | usable transitions |
|---|---:|---:|
| `PHOT_CHLRE_Chen_2023` | 99% | 1,231,742 (capped to 60,000) |
| `CAPSD_AAV2S_Sinai_2021` | 83% | 20,142 |
| `GFP_AEQVI_Sarkisyan_2016` | 81% | 3,769 |
| `Q8WTC7_9CNID_Somermeyer_2022` | 64% | 741 — skipped |
| `GCN4_YEAST_Staller_2018` | 100% | 144 — skipped |
| `Q6WV12_9MAXI_Somermeyer_2022` | 46% | 92 — skipped |
| `D7PM05_CLYGR_Somermeyer_2022` | 55% | 55 — skipped |

A `transitions` workload in `scripts/encode_corpus.py` closes this; encoding resumes per
sequence, so it pays only for the ~59k that are missing.

---

## Cost of the per-residue option

GPU time is not the constraint; **storage and transfer are**.

| assay | L | transitions | per-residue cache | GPU | rental |
|---|---:|---:|---:|---:|---:|
| `PHOT_CHLRE_Chen_2023` | 118 | 1,231,742 | 50.6 GB | 13 min | $0.12 |
| `CAPSD_AAV2S_Sinai_2021` | 735 | 45,848 | 79.6 GB | 20 min | $0.20 |
| all seven | — | 1,395,965 | 216.7 GB | 54 min | $0.53 |

Downloading PHOT's 50.6 GB at the measured ~10 MB/s takes ~85 minutes, against 13 minutes
to produce it. **The sensible shape is to train on the rented box where the cache lives
and bring home metrics and a checkpoint**, not the cache — roughly $0.35 for encode plus
training, against an hour and a half of transfer for the same result.

**Two follow-ups measured since, both in `results/state_representation.md`:**

- A **windowed** cache — wild type plus only the rows near each mutation — would have cut
  the seven-assay total from 216.7 GB to 81.9 GB. It is **refuted**: on `CAPSD` a radius-16
  window recovers 37.6% of the pooled displacement (cosine 0.697), because a substitution's
  effect on the pooled vector is a diffuse chain-wide shift rather than a local one. It
  works on short `PHOT` (93.6% at radius 16) and fails exactly where the saving would be.
  Store the full tensor.
- The local card does **270 res/s** per-residue against the rented A40's ~26,000: 45 hours
  against 28 minutes for the 43.6M residues this needs. Storage is not the constraint —
  `/home` has 464 GB free — throughput is, by 96×.

---

## Two bugs, both caught by tests rather than by a run looking wrong

**SIGReg was blind to the failure it exists to detect.** It standardised each random
projection to unit variance before the normality test. A latent with all its variance in
one direction projects to a *scaled Gaussian* on every direction, and once each projection
is rescaled, it is indistinguishable from isotropic — the statistic read ≈ 0 for total
collapse. Since SIGReg is the only anti-collapse term in a model with no EMA target, this
would have invalidated the entire Phase 3 gate while reporting healthy numbers. Now
centred globally against one shared scale: 0.0006 isotropic → 0.405 collapsed to a single
direction, monotone in between.

**SIGReg built its projection matrix on CPU while the latents were on GPU.** Drawn on CPU
from the seeded generator and moved, so the directions do not depend on the device and a
CPU run and a GPU run report the same statistic for the same latent.
