# Step 1.3 — D2: Latent capacity and dimension

**Status: COMPLETE.** The ddG and dimension work ran on the laptop; the backbone
comparison needed Step 1.2's folds and was settled on 2026-11-26 — see the correction
immediately below, which reverses the backbone conclusion.
**Run:** 2026-11-26 · `scripts/d2_latent_capacity.py` · 1,000 Tsuboyama mutants with
measured ddG, 50 domains, lengths 36–72.

---

## Verdict: go — but the stated reason for the architecture does not survive

The go/no-go was: *"Per-residue should track |ddG| where pooled does not — that is the
revision's central claim about the latent. If both fail, stop and reconsider the
encoder before Phase 2."*

Both track it. Neither fails. **And they track it about equally well**, so the clause
that was supposed to discriminate between them does not.

| Spearman against \|ddG\| | SaProt 650M | ESM-2 650M |
|---|---:|---:|
| displacement at the substituted residue | +0.131 | +0.265 |
| **whole per-residue matrix** (Frobenius) | **+0.459** | **+0.445** |
| **mean-pooled** | **+0.410** | **+0.361** |
| per-residue advantage | **+0.049** | **+0.084** |

A 0.05–0.08 advantage is real but it is not the categorical difference the proposal
describes. Per-residue latents remain the right choice — Step 1.4 shows the
inverse-dynamics anchor **cannot** work without them, because a pooled latent has no
position to predict — but that argument, not this one, is what justifies them.

**The attenuation mechanism is confirmed, as a different claim.** Pooled displacement
shrinks with protein length (ρ = **−0.174** SaProt, **−0.307** ESM-2 against L) while
per-residue displacement at the substituted site does not (−0.075, −0.005). So
mean-pooling does dilute a point mutation as O(1/L) predicts. What the measurement
rejects is the inference from there to "pooled latents cannot carry the effect": over
36–72 residues the dilution still leaves ρ = 0.36–0.41. The proposal should claim the
mechanism and not the consequence.

---

## Dimension: 128

The adapter maps 1280 → d, and Phase 2 onward sees only the projection. The sweep
projects the whole per-residue difference matrix onto the top-d directions of the
training folds' residue differences and takes its Frobenius norm, scored out of fold.

| d | SaProt | ESM-2 |
|---|---:|---:|
| 64 | +0.449 | **+0.503** |
| **128** | **+0.473** | +0.502 |
| 256 | +0.493 | +0.506 |
| 512 | +0.511 | +0.511 |
| *unprojected (1280)* | *+0.459* | *+0.445* |

**d = 128.** It already matches or exceeds the unprojected signal on both backbones,
and a 4× larger latent buys at most +0.04. Note the projection is *better* than the
raw 1280-dimensional norm at every d for ESM-2 and from d=128 for SaProt — PCA is
discarding directions that carry no stability signal, so the adapter is denoising, not
merely compressing.

This matters beyond accuracy. `DATA_REGISTER.md` measured the full ProteinGym cache at
**1.7 TiB** at d=1280 and **~170 GiB** at d=128, and the landscape cache at 256 GiB
versus 26 GiB. The rental plan assumed d=128 would be adequate; it now is, by
measurement.

**One caveat on the sweep.** The basis is fitted on 400 mutants' residue differences
rather than all 1,000, to bound memory. The ordering across d is stable and the folds
agree, but a basis from more data would likely shift the absolute values slightly.

---

## ⚠ Updated 2026-11-26 after the folds existed: SaProt loses

The section below was written when SaProt had **no** structure tokens, because the
Tsuboyama domains had no structures. Step 1.2's folds removed that excuse, and the
re-run reverses the backbone conclusion.

| Spearman against \|ddG\| | SaProt, no 3Di | **SaProt + real 3Di** | ESM-2 *(ignores 3Di)* |
|---|---:|---:|---:|
| whole per-residue matrix | +0.459 | **+0.304** | **+0.445** |
| mean-pooled | +0.410 | +0.311 | +0.361 |
| at the substituted residue | +0.131 | +0.197 | +0.265 |
| projected to d = 128 | +0.473 | **+0.332** | **+0.502** |

**Feeding SaProt its structure channel makes the stability signal worse**, by 0.155
on the per-residue matrix. ESM-2's numbers are byte-identical to the earlier run,
which is the control: a sequence-only model must ignore the structures, and it does.

Why: Step 1.2 measured TM-score **0.984** between wild type and mutant even at
ddG ≤ −2. ESMFold returns a confident, well-folded structure for a severely
destabilising mutation, so the 3Di tokens assert *"still folded"* for proteins that
are measurably destabilised — a misleading input, not merely a weak one.

**ESM-2 650M is therefore the leading backbone** (+0.445 raw, +0.502 at d = 128).
Both are MIT-licensed, so nothing but the measurement is at stake. Confirm on H1 at
Step 2.2 before making it permanent — this is one task on 50 small domains with one
structure predictor.

**The d = 128 choice is unaffected** and holds on both backbones.

---

## Backbone: the original reasoning, before the folds existed

ESM-2 is marginally ahead in the projected space (+0.503 against +0.449 at d=64) and
marginally behind unprojected (+0.445 against +0.459). On this metric the two are a
wash.

**But SaProt ran without its structure channel.** Its input is interleaved
(residue, 3Di) pairs, and the Tsuboyama domains used here have no structures, so every
position got the unknown-structure token. SaProt-without-3Di is close to a plain
sequence model, so this compares ESM-2 to a deliberately hobbled SaProt rather than to
SaProt. The comparison cannot decide the backbone and should not be reported as
though it did.

Resolving it properly needs predicted structures for these domains — which is Step
1.2's folding job, on the rented box. **Re-run this sweep with real 3Di after the
folds exist**; `scripts/d1_fold.py` writes exactly the PDBs it needs.

ProSST is absent deliberately: it publishes no licence (`DATA_REGISTER.md`), which is
why SaProt is the default. ESM-C is not run — it needs a download, and the question it
would settle is the one the missing 3Di already blocks.

---

## What this changes

- **Fix d = 128** in the adapter. Measured, not assumed, and it unlocks the 10×
  storage reduction the rental plan depends on.
- **Keep per-residue latents**, on Step 1.4's position argument. Drop the claim that
  pooled latents cannot track stability — they can, at ρ ≈ 0.4.
- **Rewrite the proposal's pooling paragraph.** It currently rests on an O(1/L)
  attenuation argument whose conclusion is measurably too strong. The attenuation is
  real (ρ = −0.17 to −0.31 against length); the inference is not.
- **Do not report a backbone winner yet.** Re-run after Step 1.2's folds supply 3Di.
- **A number to keep for Phase 2.** ρ ≈ 0.46 between latent displacement and |ddG| is
  the frozen encoder's unsupervised ceiling on a stability signal. The reward head
  should beat it comfortably; if it does not, the adapter is not learning anything the
  norm of the difference already gave away.

---

## Reproduction

```
python -I scripts/d2_latent_capacity.py                      # both backbones, ~10 min warm
python -I scripts/d2_latent_capacity.py --backbones saprot --limit 10
```

Reads `data/processed/d1_fold_manifest.json`, writes `data/processed/d2_latent.json`,
and records runs under `1.3`. Reuses the embedding cache, so after Step 1.4 the SaProt
pass costs almost nothing. Mutants are stratified by measured ddG (5 severe / 5
moderate / 5 mild / 5 stabilising per domain), so the correlation is not carried by a
narrow band of effect sizes.
