# Step 1.4 — D4: Inverse-dynamics feasibility probe

**Status: COMPLETE.** No GPU rental was needed; this ran on the laptop GTX 1650.
**Run:** 2026-11-26 · `scripts/d4_inverse_dynamics.py` · SaProt 650M, frozen ·
1,080 wild-type/mutant pairs over 54 proteins, lengths 36–390.

---

## Verdict: the precondition holds, decisively

The inverse-dynamics anchor is the most load-bearing new component in the revised
architecture, and it only works if the pair `(Z_wt, Z_mut)` identifies which
substitution was applied. It does.

| | result | chance | |
|---|---:|---:|---|
| **Position, top-1** | **0.999** | 0.019 | **53×** |
| Position, top-5 | 1.000 | 0.094 | |
| **New residue, top-1** | **0.999** | 0.050 | 20× |
| New residue, top-5 | 1.000 | | |

The go/no-go asked for position recovery "well above chance (1/L)". It is 53× chance,
at 0.999 on 1,080 pairs — one miss.

**Position recovery is parameter-free.** Residues are ranked by how far their
embedding moved and the substituted one is simply read off the top; nothing is fitted,
so the number cannot be an artifact of a model with enough capacity to memorise. The
median rank of the true position is **0**.

Residue identity does need a classifier, so it is a multinomial logistic regression
scored **out of fold**. Fitting and scoring on the same 1,280-dimensional feature
would have reported its memorisation capacity instead of a property of the embedding.

---

## The pooling question, answered more narrowly than the proposal assumes

Section 3 argues that mean-pooling attenuates a single-residue change by O(1/L), and
that argument is part of why the architecture uses per-residue latents. The probe
separates two claims that the proposal runs together.

**For position, pooling is not merely worse — it is impossible.** A pooled latent is a
single `d`-vector with no per-position structure, so there is nothing to rank. This is
the architecturally decisive half, because the anchor must predict `(i, AA_new)` and
`i` exists only per residue.

**For residue identity, pooling costs almost nothing.** Same mutants, same classifier,
only the feature changed:

| feature | top-1 | top-5 |
|---|---:|---:|
| per-residue difference at the substituted position | **0.999** | 1.000 |
| mean-pooled difference | **0.942** | 0.983 |

A 0.057 gap, not a collapse. On a 4-protein pilot the gap was 0.525, and it closed as
the training set grew — which is the explanation: pooling shrinks the *magnitude* of
the difference vector but largely preserves its *direction*, and a linear classifier
with enough examples reads the direction. **Attenuation is not the same as
destruction**, and the proposal should not claim the second from the first.

The attenuation itself is real and measurable, as the concentration of movement at the
substituted residue:

| length | n | position top-1 | share of total movement at the substituted residue |
|---|---:|---:|---:|
| 36–50 | 480 | 0.998 | **11.8%** |
| 50–60 | 260 | 1.000 | 10.7% |
| 60–70 | 220 | 1.000 | 9.5% |
| 70–100 | 80 | 1.000 | 8.1% |
| 100–390 | 40 | 1.000 | **6.3%** |

The share falls with length as expected. **Recoverability does not** — position top-1
stays at 1.000 out to 390 residues. So within this range the signal thins without
becoming undecodable, and the O(1/L) argument bounds the signal's *share*, not the
anchor's feasibility.

Two caveats on the length row: it stops at 390 because the measured proteins stop
there, and the 100–390 band is 40 pairs from four proteins, so it is thin. A real
decay could appear at 500+ residues; nothing here rules that out.

---

## What this changes

- **Build the anchor.** The precondition is satisfied with a wide margin, and the
  factorised cross-entropy over L×20 that the proposal specifies is decoding a signal
  that is nearly perfectly present.
- **Keep per-residue latents** — but on the position argument, which is absolute,
  rather than on O(1/L) attenuation of residue identity, which is a 0.057 effect here.
- **Soften the proposal's pooling claim.** It currently implies pooled latents cannot
  carry a point mutation's effect. For *which residue was introduced*, they mostly can.
  The defensible version is that pooling cannot represent *where*.
- **A ceiling to remember in Phase 3.** The anchor's task is close to free at this
  difficulty. An inverse-dynamics loss that is already near-perfect contributes little
  gradient, so if it is meant to shape the latent it may need a harder variant —
  multi-step actions, or prediction from a pooled or noised latent — rather than the
  single-substitution version measured here. This is a design note, not a finding.

---

## Reproduction

```
python -I scripts/d4_inverse_dynamics.py            # full manifest, ~7 min on a GTX 1650
python -I scripts/d4_inverse_dynamics.py --limit 10
```

Reads `data/processed/d1_fold_manifest.json`, writes `data/processed/d4_inverse.json`,
and records three tracker runs under `1.4`. Mutants come from Tsuboyama 2023 with
measured ddG, stratified 5 severe / 5 moderate / 5 mild / 5 stabilising per protein,
so the pairs are not all easy substitutions.
