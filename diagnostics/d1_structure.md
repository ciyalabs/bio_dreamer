# Step 1.2 — D1: Structure invariance under point mutation

**Status: COMPLETE.** **Run:** 2026-11-26 · ESMFold on a rented A40 · 1,188 folds over
54 proteins (36–390 residues), 1,080 mutants stratified by measured ddG.
**Cost: about $1.05.**

---

## Verdict: drop the per-step structure stream — but not for the stated reason

The proposal removes the per-step structure stream on the grounds that **a point
mutation barely moves the predicted structure**. Measured, that premise is wrong in
one direction and irrelevant in another.

**The structure does move, and it moves in proportion to how destabilising the
mutation is:**

| ddG band | n | Cα RMSD (Å) | 3Di Hamming |
|---|---:|---:|---:|
| **severe** (≤ −2 kcal/mol) | 243 | **0.541** | **0.0896** |
| moderate (−2 to −1) | 257 | 0.322 | 0.0667 |
| mild (−1 to 0) | 250 | 0.214 | 0.0348 |
| stabilising (> 0) | 250 | 0.210 | 0.0295 |
| *length probe (no ddG)* | 80 | 0.567 | 0.0433 |

Monotone across all four measured bands, on both metrics. That is a dose–response,
not noise.

**The floor is zero, so every one of those numbers is signal.** Each wild type was
folded **twice**, and across all 54 proteins the two passes gave **bitwise identical
3Di strings** (floor = 0.0 exactly) and a Cα RMSD of 3.4 × 10⁻¹⁵ Å — float64 rounding
in the superposition, not structure. ESMFold is deterministic. Without that second
fold the comparison would have had no scale and "the structure does not move" could
not have been falsified.

So **SaProt's structure channel is not invariant to point mutation**: 5.2% of 3Di
tokens change at the median, 9.0% for severe destabilisation.

**But the global fold is untouched.** TM-score is **0.9838** and median RMSD **0.32 Å**
— within crystallographic noise. And a fold costs **30 seconds** on an A40, which is
prohibitive inside a planning loop that takes hundreds of steps.

**The decision survives; the argument for it does not.** Per-step structure is dropped
because it is expensive and the global fold is static, **not** because the structure
channel carries nothing. It carries a graded, measurable signal.

---

## And the structure channel actively hurts

> **Correction, 2026-11-27.** The backbone conclusion below compared ESM-2 against
> SaProt **with** 3Di only. Like for like, SaProt **without** 3Di was already ahead
> (+0.459 vs +0.445), and Step 2.1's trained stability head confirms it: pooled
> Spearman **+0.337 for SaProt against +0.209 for ESM-2**, per-protein median +0.753
> against +0.598. **SaProt stays the default, run without its structure channel.** The
> finding that survives is about the channel, not the model: do not feed a
> structure-aware encoder predicted 3Di for stability work.


Step 1.3 was re-run with these predicted structures, which is the first time SaProt
has been measured with its 3Di channel fed rather than blanked:

| Spearman against \|ddG\| | SaProt, no 3Di | **SaProt + real 3Di** | ESM-2 *(ignores 3Di)* |
|---|---:|---:|---:|
| whole per-residue matrix | +0.459 | **+0.304** | **+0.445** |
| mean-pooled | +0.410 | +0.311 | +0.361 |
| at the substituted residue | +0.131 | +0.197 | +0.265 |
| projected to d = 128 | +0.473 | **+0.332** | **+0.502** |

**Giving SaProt the predicted structure makes its stability signal worse** — 0.459 →
0.304 on the per-residue matrix, 0.473 → 0.332 at d = 128.

ESM-2's figures are byte-identical to the pre-fold run, which is the control: it is
sequence-only, so it must ignore the structures, and it does.

The explanation is in the table above. ESMFold returns a confident, well-folded
structure for a severely destabilising mutant — TM-score 0.984 even at ddG ≤ −2. The
3Di tokens therefore assert *"still folded"* for proteins that are measurably
destabilised. That is not a weak signal, it is a **misleading** one, and it dilutes
the sequence signal SaProt would otherwise have.

**Consequence: switch the channel off, not the model.** SaProt without 3Di scores
+0.459 against ESM-2's +0.445, and its trained reward head reaches +0.337 against
ESM-2's +0.209. Structure-*awareness* is not the liability; being handed a
stability-blind predicted structure is.

**Caveat before acting on it.** This is one task — correlation of latent displacement
with \|ddG\| — on 50 small domains of 36–72 residues, with structures from one
predictor. The reward head consumes the latents differently, and Phase 2's H1
benchmark is the test that matters. Treat this as a strong prior for ESM-2, to be
confirmed at Step 2.2, not as a settled encoder change.

---

## What this changes

- **Keep the architecture decision**, restate the reason: cost and a static global
  fold, not an unmoving structure channel.
- **Rewrite the proposal's §3 structure paragraph.** It claims recent evidence shows
  structure is invariant under point mutation. The measurement here shows a graded
  response against a zero floor; what is invariant is the *fold*, not the tokens.
- **Carry SaProt into Phase 2 with its structure channel off**, ESM-2 as the
  comparator, and settle it on H1 rather than here.
- **Do not feed predicted 3Di to a structure-aware encoder for stability work** until
  a predictor that responds to destabilisation is available. This is the sharpest
  practical finding of Phase 1.

---

## Reproduction

```
python -I scripts/d1_build_manifest.py        # CPU, picks the proteins and mutants
python -I scripts/d1_fold.py                  # needs ~16 GB VRAM; 36 min on an A40
python -I scripts/d1_fold.py --log-only       # re-record tracker runs, no folding
python -I scripts/d2_latent_capacity.py       # uses the folds; --no-3di for the control
```

Writes `data/processed/d1_structure.json` and the predicted structures under
`data/processed/d1_folds/` (65 MB, gitignored; the manifest that generates them is
committed). Tracker runs under `1.2 / struct_displacement` and
`1.2 / substitution_stratified`.

**Mutants are stratified by measured ddG** — 5 severe / 5 moderate / 5 mild / 5
stabilising per protein — which is what makes the dose–response visible. An
unstratified sample would be dominated by mild substitutions and would have shown a
flat median, i.e. the invariance the proposal assumed.
