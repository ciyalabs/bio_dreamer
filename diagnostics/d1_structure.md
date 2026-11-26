# Step 1.2 — D1: Structure invariance under point mutation

**Status: PARTIAL.** The local half is done; the folding half is blocked on hardware.
**Run:** 2026-11-26 · `scripts/d1_noise_floor.py` · SaProt 650M, real 3Di from shipped
wild-type structures.

---

## What this step decides

Whether to keep a **per-step structure stream** in the world model — the largest
architectural change in the proposal revision was removing it. The argument for removal is
that a point mutation barely moves the predicted structure, so re-folding at every planning
step buys nothing. That argument needs measuring, not asserting.

It splits cleanly into two halves, and only one runs on this machine.

| | Measures | Needs | Status |
|---|---|---|---|
| **Noise floor** | how much the encoder moves when nothing changes | SaProt, 1.3 GiB | **done** |
| **Sequence-channel signal** | displacement under one substitution, structure held fixed | SaProt | **done** |
| **Structure displacement** | TM-score, Cα RMSD, 3Di token change, WT vs mutant | ESMFold | **blocked** |
| **Folding noise floor** | fold the same sequence twice — the floor for the row above | ESMFold | **blocked** |

---

## The noise floor is fp16 rounding

The comparison the step rests on is meaningless without knowing how much the encoder's output
moves when the input does not. Three perturbations, none biological, measured as relative
Frobenius displacement `‖A−B‖/‖A‖`:

| protein | L | identical call | padded in batch | batch size 1 vs 8 | **floor** | mutation | SNR | change within ±5 residues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GB1 | 56 | 0 | 1.4e-3 | 1.4e-3 | **1.45e-3** | 0.1318 | 91× | 39.8% |
| ParD3 | 85 | 0 | 1.4e-3 | 1.4e-3 | **1.40e-3** | 0.1201 | 86× | 44.3% |
| ParD2 | 87 | 0 | 1.4e-3 | 1.4e-3 | **1.38e-3** | 0.1196 | 87× | 44.2% |
| TEV | 229 | 0 | **0** | **0** | **0** | 0.0736 | — | 32.8% |
| TrpB | 389 | 0 | **0** | **0** | **0** | 0.0504 | — | 25.9% |

**The two zeros are not better stability.** The backbone runs in **float16**, whose epsilon is
9.77e-4, and the measured floor is 1.38–1.45e-3 — **1.4 to 1.5 × fp16 epsilon**. The floor is
float16 rounding under a changed reduction order: cuBLAS picks kernels by tensor shape, and on
the two long proteins it happened to pick the same kernel on both sides of the comparison, so
the difference came out bitwise zero. Rerun with different padding and it will not.

So the figure to use downstream is the **pooled floor, 1.45e-3**, and against it the weakest
single-substitution signal (TrpB, 0.0504) is still **35×**. Recorded as a standing number:
any structural displacement below ~1.5e-3 relative is indistinguishable from arithmetic.

Two further things worth having on the record:

**A single substitution is not a local perturbation of the embedding.** Only 26–44% of the
total per-residue change falls within ±5 residues of the substituted site; the majority is
spread across the rest of the chain. This is the sequence channel alone — the structure
channel was held fixed — so it is a property of the encoder's attention, not of protein
physics. It matters because it sets a ceiling on how much a per-step structure update could
add: the embedding already moves globally without one.

**Displacement falls with length** (0.132 at L=56 to 0.050 at L=389), roughly as one would
expect from one residue in L changing under a relative-norm metric. Comparisons across
proteins of different length must therefore be length-normalised or stratified, not pooled.

---

## What is blocked, and why

The remaining experiments need a structure predictor for every mutant: ~50 proteins × ~20
single mutants ≈ 1,000–2,000 folds.

**ESMFold does not fit this machine.** The GPU is a laptop **GTX 1650 with 4,096 MiB total
and ~2,700 MiB free**. `facebook/esmfold_v1` carries an ESM-2 3B trunk plus a folding head;
its weights alone exceed the card in half precision, before any activation memory, which for
the folding trunk scales with L². No chunking setting reduces weight memory.

**Do not rent for this step alone.** `scripts/cost_model.py` puts the whole of Phase 1 at
~6.5 A100-hours of pure compute, of which D1's folds are 1.7:

| Phase 1 item | EFLOP | A100-hours |
|---|---:|---:|
| Encode ProteinGym (210 trainable assays) | 0.90 | 2.5 |
| D1 ESMFold folds (~2,000 mutants) | 0.60 | 1.7 |
| D2 backbone sweep (Step 1.3) | 0.50 | 1.4 |
| Encode Tsuboyama + held-out landscapes | 0.31 | 0.9 |
| **Total** | **2.31** | **6.5** |

One rental covering D1's folds, Step 1.3's backbone sweep and the two encoding passes is the
efficient unit. A 24 GB card (L4, A10G, RTX 4090) is sufficient — the 40 GB A100 the cost
model prices is not required, only ≥16 GB for ESMFold.

**One methodological point for when that rental happens.** The floor for the TM-score and RMSD
comparison is *not* the encoder's fp16 floor measured above — it is the **folding model's own
reproducibility**, which is a different and larger number. Fold every wild type twice and
measure TM-score and RMSD between the two predictions before comparing any mutant to anything.
Skipping that would make "the structure does not move" unfalsifiable, which is the same error
Step 1.1 found in the TEV figures: an artifact of the instrument read as a property of biology.

There is also a known confound to state in the write-up rather than discover in review:
**sequence-based structure predictors are documented as insensitive to point mutations.** A
finding that predicted structure does not move is therefore partly a statement about the
predictor. That is still the right measurement for the *architectural* decision — the stream
we would actually build consumes predicted structure, so if the prediction does not move the
stream carries nothing — but it is not evidence that the real structure does not move, and the
two claims must not be conflated. If a biological claim is wanted, the route that needs no GPU
is experimentally determined wild-type/mutant structure pairs from the PDB, compared with the
already-installed Foldseek.

---

## Reproduction

```
python -I scripts/d1_noise_floor.py                      # all five proteins, ~3 min
python -I scripts/d1_noise_floor.py --proteins TEV --n-mutants 10
```

Writes `data/processed/d1_noise_floor.json`; tracker runs under `1.2 / noise_floor`.
Caching is disabled throughout — a cache hit returns the identical array and would report a
floor of zero everywhere.
