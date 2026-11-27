# Step 2.1 — Fitness target trained, and it ties with counting mutations

**Status: Step 2.1 COMPLETE.** **Run:** 2026-11-27, CPU only, on the pooled cache
encoded the same day. Training takes 12–21 s; the cache cost ~$3 of rented A40.
Both the `benchmark` and `planning` instances now exist.

---

## Verdict: the head learns real signal, and it is not yet good enough

Two numbers, and they point in opposite directions. Both are reported because reporting
only the first would be the mistake this project keeps making.

**It is well clear of chance.** 6 of 7 `planning` assays beat a random-head band by 3–11×.
**It does not beat substitution count.** A predictor that ignores the protein entirely and
reads only *how many mutations a variant carries* scores mean |ρ| **0.312**; the trained
head scores **+0.260**.

| instance | assays | mean ρ | median | min | chance band | order-only | H1 needs |
|---|---:|---:|---:|---:|---:|---:|---:|
| **planning** | 7 | **+0.260 ± 0.012** | +0.370 | −0.175 | 0.098 | **0.312** | 0.55 |
| **benchmark** | 11 | **+0.290** | +0.347 | −0.313 | 0.113 | **0.303** | 0.55 |

The mean is quoted over three head initialisations (+0.250 / +0.277 / +0.253). The spread
is ±0.012, so the gap to 0.55 is not an unlucky draw.

### Per assay, `planning`

| assay | ρ | chance band | order-only | verdict |
|---|---:|---:|---:|---|
| GFP_AEQVI_Sarkisyan_2016 | **+0.413** | 0.072 ± 0.081 | −0.464 | above chance, below order |
| D7PM05_CLYGR_Somermeyer_2022 | **+0.397** | 0.036 ± 0.045 | −0.351 | above both |
| CAPSD_AAV2S_Sinai_2021 | **+0.396** | 0.117 ± 0.149 | −0.451 | above chance, below order |
| Q6WV12_9MAXI_Somermeyer_2022 | **+0.370** | 0.078 ± 0.090 | −0.207 | above both |
| Q8WTC7_9CNID_Somermeyer_2022 | +0.269 | 0.032 ± 0.033 | −0.213 | above both |
| PHOT_CHLRE_Chen_2023 | +0.082 | 0.209 ± 0.226 | −0.328 | **inside chance** |
| GCN4_YEAST_Staller_2018 | −0.175 | 0.140 ± 0.167 | −0.170 | **ranks backwards** |

### Per assay, `benchmark`

| assay | ρ | chance band | order-only | |
|---|---:|---:|---:|---|
| SPG1_STRSG_Wu_2016 | **+0.495** | 0.182 ± 0.204 | −0.125 | BANNED |
| GFP_AEQVI_Sarkisyan_2016 | +0.414 | 0.072 ± 0.081 | −0.464 | |
| F7YBW8_MESOW_Ding_2023 | +0.405 | 0.139 ± 0.165 | −0.505 | BANNED |
| F7YBW8_MESOW_Aakre_2015 | +0.379 | 0.117 ± 0.119 | −0.322 | BANNED |
| D7PM05_CLYGR_Somermeyer_2022 | +0.351 | 0.036 ± 0.045 | −0.351 | |
| CAPSD_AAV2S_Sinai_2021 | +0.347 | 0.117 ± 0.149 | −0.451 | |
| Q6WV12_9MAXI_Somermeyer_2022 | +0.341 | 0.078 ± 0.090 | −0.207 | |
| HIS7_YEAST_Pokusaeva_2019 | +0.319 | 0.124 ± 0.131 | −0.202 | BANNED |
| Q8WTC7_9CNID_Somermeyer_2022 | +0.266 | 0.032 ± 0.033 | −0.213 | |
| PHOT_CHLRE_Chen_2023 | +0.181 | 0.209 ± 0.226 | −0.328 | inside chance |
| GCN4_YEAST_Staller_2018 | **−0.313** | 0.140 ± 0.167 | −0.170 | backwards |

Above chance on **10 of 11**; beats order-only on **6 of 11**.

Three of the four banned assays rank in the top four — SPG1_STRSG **+0.495**,
F7YBW8_Ding **+0.405**, F7YBW8_Aakre **+0.379** — and they are the deepest libraries in
the corpus. Worth stating plainly: those are among the best numbers in the table and
**none of them may ever support a Phase 6 claim**, because only `planning` is allowed to.
That is the two-instance rule doing visible work rather than sitting in a registry.

Adding the four banned assays also made GCN4 *worse*, −0.175 to **−0.313**. With 88
training rows out of 2,954 it has almost no weight in the shared gradient, so it is the
assay the trunk gives up first when asked to fit more of the corpus.

---

## The control had to be fixed before any of this could be read

The first version of this run used **one** untrained head as its control. On
PHOT_CHLRE that head scored **+0.517** — above the trained head's +0.082 — and the
obvious reading was that training had destroyed PHOT.

That reading was wrong. Twenty random initialisations on PHOT give:

| | mean | sd | min | max |
|---|---:|---:|---:|---:|
| PHOT_CHLRE, 20 random heads | −0.031 | **0.248** | −0.438 | +0.472 |
| HIS7_YEAST, 20 random heads | −0.019 | 0.135 | −0.204 | +0.207 |
| CAPSD_AAV2S, 20 random heads | −0.023 | 0.184 | −0.301 | +0.361 |

**+0.517 was a draw from a distribution with sd 0.25.** A single random linear readout of
a 1280-d embedding is not a control; it is one sample of a wide null. The control is now
mean |ρ| over 10 seeds with its spread, and PHOT's honest status is *inside the chance
band*, not *worse than random*.

This is the fifth time in this project that an instrument artifact read as a finding —
after pooling attenuation, structure invariance, greedy's success rate and the backbone
comparison. The pattern is stable enough to plan around: **any control that is a single
draw needs its variance measured before its value is used.**

---

## What is actually wrong

**The order-only tie is real but it is not the diagnosis.** Fitness falls with mutation
count in every one of these assays (order-only ρ is negative in all 11, −0.125 to
−0.505), so a predictor reading depth alone scores ~0.31. The head scores 0.26 *without*
using depth: see the within-order measurement below, which shows +0.18 to +0.36 at fixed
depth on five of seven assays. The head and the trivial baseline are therefore
complementary rather than competing, and the aggregate tie understates what the head
learned while correctly reporting that it is not yet an H1-competitive predictor.

**Two assays fail outright.**

- **GCN4_YEAST** ranks backwards on every run (−0.088 to −0.313). It has **88**
  training variants, the fewest, and 44-order variants in its test set. Its order-only
  score is −0.170, so depth barely helps either — this assay may simply be unlearnable
  from 88 low-order examples.
- **PHOT_CHLRE** sits in its chance band on both instances (+0.082, +0.181) and is **58%
  of the planning test set**, so it drags the mean hardest of any single assay.

**What this does not establish.** This is not H1. H1 (Step 2.2) requires the additive
null, Kermut / ProteinNPT / MULTI-evolve / Cerebra, and the zero-shot baselines; none are
computed here. The order-only control is deliberately *weaker* than the additive null —
it fits nothing at all — which makes the tie worse news, not better: the additive null
fits per-site effects and should score higher than 0.31.

---

## Depth-stratified pair sampling: implemented, measured, and it does not work

This was the leading candidate fix, on the reasoning that uniform pair sampling fills
the gradient with order-1-versus-order-2 comparisons whose right answer is "the
shallower one". It is implemented (`equal_order_frac` on `ranking_pairs` and
`pairwise_ranking_loss`, 7 tests) and it fails.

| `equal_order_frac` | 0.0 | 0.5 | 1.0 |
|---|---:|---:|---:|
| planning mean ρ | **+0.260 ± 0.012** | +0.257 | **+0.228** |

**The reasoning was wrong, and a diagnostic printed before the first run said so.** The
training set holds 441 singles and 1,447 doubles, so doubles-versus-doubles dominates
combinatorially: **84.6% of comparable pairs are already equal-order.** Stratifying can
only touch the remaining 15%, and forcing it to 100% makes things slightly worse
(−0.03, about 2.5× the init spread).

`PHOT_CHLRE` degrades monotonically as depth pairs are removed — +0.082 → +0.020 →
−0.069 — which is the clue that led to the next measurement.

**The default stays `equal_order_frac = 0.0`.** The option is kept because the
measurement is worth preserving and the diagnostic it added (pair composition, printed
every run) is worth keeping.

---

## What is actually happening: the head and depth are complementary

Measured **within a single mutation depth**, where depth carries no information by
construction:

| assay | ρ(pred, fitness) | ρ(pred, order) | **within-order ρ** |
|---|---:|---:|---:|
| CAPSD_AAV2S_Sinai_2021 | +0.400 | +0.099 | **+0.364** (15 depths) |
| Q6WV12_9MAXI_Somermeyer_2022 | +0.371 | −0.146 | **+0.322** (4) |
| Q8WTC7_9CNID_Somermeyer_2022 | +0.283 | −0.099 | **+0.228** (5) |
| GFP_AEQVI_Sarkisyan_2016 | +0.417 | −0.239 | **+0.195** (8) |
| D7PM05_CLYGR_Somermeyer_2022 | +0.407 | −0.321 | **+0.181** (6) |
| PHOT_CHLRE_Chen_2023 | +0.094 | **−0.472** | −0.028 (11) |
| GCN4_YEAST_Staller_2018 | −0.206 | −0.132 | −0.250 (3) |

**The head is not a depth predictor.** On five of seven assays it ranks variants at the
same depth at +0.18 to +0.36 — skill the order-only baseline has exactly none of, since
within a depth it is constant. So the earlier framing, *"ties with counting mutations"*,
was numerically true and causally misleading. The two predictors are complementary: the
head reads the substitution and barely uses depth; the baseline uses depth and cannot
read the substitution.

`PHOT_CHLRE` is the exception that proves it — ρ(pred, order) = **−0.472** and
within-order ρ = −0.028. Its entire apparent +0.094 came from accidentally tracking
depth. It is the one assay where the head really did learn only depth, and that is why
stratifying hurt it most.

**Combining them, with a fixed 50/50 rank average:**

| | head | order-only | **50/50 blend** |
|---|---:|---:|---:|
| mean ρ over 7 assays | +0.252 | +0.312 | **+0.375** |
| CAPSD_AAV2S | +0.400 | +0.451 | **+0.637** |
| GFP_AEQVI | +0.417 | +0.464 | **+0.567** |

+0.12 over the head and +0.06 over the baseline, with two assays at or above H1's 0.55
threshold. A per-assay weight fitted on the test set reaches +0.417; that is an **upper
bound, not a result**, and is recorded here only to bound the headroom.

---

## The depth-aware head: +0.260 → +0.384, and `beta` is not fitted

Making the combination part of the model rather than a post-hoc blend. The score becomes
`fitness(h) − softplus(beta) · f(order)`, with `softplus` so the penalty is structurally
non-negative: deeper is never scored better. That monotonicity is a prior, justified by
order-only ρ being negative on all 11 assays, not a thing to be learned from scratch.
`fitness_sequence_only` is returned alongside, so the two components stay separable.

| configuration | mean ρ | median | min | all positive |
|---|---:|---:|---:|---|
| head alone, `planning` | +0.260 ± 0.012 | +0.370 | −0.175 | no |
| order-only baseline | 0.312 | — | — | — |
| unfitted 50/50 rank blend | +0.375 | — | — | — |
| **depth-aware `linear`, `planning`** | **+0.384** | +0.413 | **+0.099** | **yes** |
| depth-aware `log`, `planning` | +0.295 | +0.389 | −0.010 | no |
| **depth-aware `linear`, `benchmark`** | **+0.393** | +0.393 | +0.003 | **yes** |

`linear` was the pre-specified prior — fitness is roughly multiplicative in independent
deleterious mutations, so a log-scale score is roughly linear in their count — and it is
the one that works, so no selection on test numbers was needed. `log` is the sensitivity
check and is reported rather than chosen between.

**Every assay is positive for the first time**, GCN4 included. The sequence component is
undamaged: seq-only mean +0.247 against +0.252 standalone, so the depth term added
information instead of crowding anything out. Within-order ρ rose from +0.145 to +0.161.

### `beta` is a hyperparameter the training data cannot identify

`beta` moved from 0.6926 to 0.6271 over 300 epochs — 9%. Sweeping its initial value:

| init | learned `beta` | mean ρ | seq-only ρ |
|---:|---:|---:|---:|
| 0.05 | 0.0655 | +0.331 | +0.289 |
| 0.2 | 0.2535 | +0.345 | +0.259 |
| **0.693** (default) | 0.6270 | **+0.384** | +0.247 |
| 2.0 | 1.7991 | +0.407 | +0.245 |
| 6.0 | 5.7202 | +0.388 | +0.197 |

**Every init stays within 10% of where it started, across a 120× range.** The gradient
reaching `beta` comes only from order-1-versus-order-2 pairs, and that contrast cannot
identify it. So calling this "fitted on the training split" would be wrong: `beta` is a
hyperparameter, and the honest statement is that the result is **+0.33 to +0.41 across
two orders of magnitude of it**, with every assay positive throughout.

That robustness is what makes the finding usable; the precise +0.384 is not a fitted
optimum. The best init (2.0, +0.407) is **not** quoted as the result — selecting it would
spend the pre-registration on H1's own test set.

Note the trade visible in the last column: a larger depth penalty lets the sequence head
do less work (seq-only +0.289 → +0.197). The total is flat because the two components
substitute for each other, which is further evidence they are complementary.

### Still short of H1, and two assays are still carried by the prior

Mean +0.384 against a registered 0.55. The chance band also rose to ~0.30, because the
control heads carry the depth penalty too — the correct bar for a depth-aware model. The
margin for the learned component is **+0.384 against 0.301**, about +0.08.

`PHOT_CHLRE` (+0.181 vs chance 0.344) and `GCN4_YEAST` (+0.099 vs 0.170) sit below their
own chance bands on `planning`: both are now carried by the depth prior rather than by
anything learned. On `benchmark`, GCN4's within-order ρ is **−0.383**, its worst figure
anywhere.

Best within-depth signal in the corpus is `SPG1_STRSG` at **+0.575** (benchmark only),
with `F7YBW8_Aakre` +0.396 and `CAPSD` +0.346 — three of which are banned assays,
unusable for a Phase 6 claim.

---

## What to try next, in the order worth trying

1. ~~**Make the reward depth-aware inside the model.**~~ **Done** — +0.384 / +0.393, see
   above. The weight turned out not to be identifiable from orders 1–2, so it is a
   hyperparameter rather than a fitted blend; the result is robust across 120× of it.
2. **Fix or drop `GCN4_YEAST`.** Negative on every configuration, within-order ρ −0.250,
   88 training variants. Either it is unlearnable from 88 low-order examples — a finding
   worth stating — or it needs separate treatment.
3. **Per-assay affine output.** Still untried, still cheap.
4. **Joint training with the stability target.** The trunk is shared and stability
   reached +0.337 / +0.753; the two caches are in different formats, which is the only
   obstacle.

**Depth-invariant input is the wrong idea** and is recorded here so it is not retried:
the measurement says depth is information the head is *failing to use*, not a confound
to divide out.

**Do not tune on the order ≥3 set.** It is H1's test set and the 0.55 threshold is
pre-registered. The recipe here (300 epochs, lr 1e-3, the stability run's values) was
fixed in advance and not adjusted after seeing any test number.

---

## Reproduction

```
python -I scripts/train_fitness.py --instance planning
python -I scripts/train_fitness.py --instance benchmark
python -I scripts/train_fitness.py --instance planning --init-seed 1 --no-track
python -I scripts/train_fitness.py --instance planning --dry-run    # 3 epochs, *.dryrun.json
```

Writes `data/processed/fitness_training.json`; tracker runs under
`2.1 / fitness_training`. `--seed` is the **split** seed and is locked to 0: the pooled
cache holds exactly the seed-0 H1 split, so another value produces cache misses and
re-encoding costs a GPU rental. `--init-seed` varies the head only.
