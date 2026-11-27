# Phase 1 gate — consolidated diagnostics, read against the pre-registration

**Decided: 2026-11-27.** Steps 1.1–1.4 complete. Total GPU spend: **$1.05**.

---

## Decision: PROCEED to Phase 2

Nothing in Phase 1 undermines the premise. Several things sharpen it, and four
stated justifications turned out to be wrong while the decisions they supported
survived. Four conditions are carried forward; none blocks starting Phase 2.

---

## 1. Is there anything to find? (Step 1.1)

**Yes, on the three landscapes whose wild type could be verified.**

| landscape | structured epistasis | greedy@5 | path accessibility | wild type |
|---|---:|---:|---:|---|
| **GB1** | **+0.421** | 0.7 | 0.208 | verified |
| **TrpB4** | **+0.344** | 0.4 | **0.000** | verified |
| **TEV** *(primary)* | **+0.257** | **0.0** | **0.000** | verified |
| *ParD2* (control) | +0.110 | 1.0 | 1.000 | verified |
| *ParD3* (control) | +0.088 | 1.0 | 1.000 | verified |

The contrast is exactly the right shape: the three H4 landscapes have path
accessibility 0.000–0.208 and greedy failing 30–100% of the time, while the two
controls have **perfect** path accessibility, greedy succeeding always, zero
reciprocal-sign traps, and stack-top-k capturing 100% of the achievable gain.

**The usable set is smaller than the 1.1 write-up implies.** That document lists
TrpB3D/E/G/I as "usable" on their structured-epistasis figures, which are
wild-type-independent and still valid. But their wild types are **unresolved**, so
every wild-type-anchored number for them — greedy, path accessibility, traps,
stack-top-k — is unusable. The tell is `frac_above_wild_type`: 97.3% for TrpB3D and
77.7% for TrpB3A, against 0.1–2.4% on the five verified landscapes. **Three H4
landscapes and two controls is what Phase 1 actually delivers.**

Same caveat retires DHFR's apparent top ranking (+0.452, greedy 0.0): it is
nucleotide-level with an unresolved wild type and 57% of variants above it. At
protein level it is 78% additive and greedy solves it 0.9 — a control, not a
candidate.

**Depth is the strongest result.** On 1,200 HIS3 hypercubes, every measure of
difficulty rises monotonically with dimension:

| dimension | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy hit rate | 0.906 | 0.918 | 0.845 | 0.723 | 0.549 | 0.325 | **0.242** |
| path accessibility | 0.609 | 0.535 | 0.461 | 0.329 | 0.283 | 0.156 | **0.095** |
| has a reciprocal-sign trap | 0.14 | 0.23 | 0.32 | 0.45 | 0.54 | 0.72 | 0.63 |
| order-1 variance share | 0.602 | 0.477 | 0.393 | 0.310 | 0.285 | 0.269 | 0.278 |

The order-1 share plateaus near 0.27 from dimension 7 while greedy keeps failing: what
grows with depth is the **number of ways to be trapped**, not the degree of
non-additivity. Every SSMuLA landscape caps at four sites, so this is the only
evidence the project has above horizon 4 — and it is the evidence that most directly
supports multi-step planning.

## 2. Does the architecture hold up? (Steps 1.2–1.4)

| question | registered expectation | measured | verdict |
|---|---|---|---|
| Inverse-dynamics anchor is feasible | position recovery ≫ 1/L | **0.999 vs chance 0.019** | **confirmed, emphatically** |
| Per-residue beats pooled on stability | pooled should fail | both work (+0.46 / +0.41) | **not confirmed** |
| Per-residue needed for position | — | pooled cannot represent it at all | **confirmed, and decisive** |
| Predicted structure is invariant | displacement ≈ encoder noise | graded 0.210→0.541 Å vs **zero** floor | **refuted** |
| Latent dimension | sweep {64,128,256,512} | **d = 128** matches unprojected | **fixed** |

**Three justifications failed; three decisions stand.**

- *Per-residue latents*: keep them, on Step 1.4's position argument (pooled has no
  position to predict), **not** on O(1/L) attenuation of magnitude — that costs only
  0.057 top-1 on residue identity.
- *No per-step structure*: keep that, on cost (30 s per fold) and a static global fold
  (TM 0.9838), **not** on invariance — the structure moves in proportion to ΔΔG.
- *H4 per-oracle-budget framing*: keep it. The measurement that drove it — "greedy
  with 5 restarts solves 4 of 5 landscapes" — was an artifact of walking from a dead
  variant, and the corrected figures support the framing more strongly, not less.

**The sharpest new finding** is that SaProt's structure channel does not merely fail
to help, it hurts: feeding it real predicted 3Di drops its correlation with |ΔΔG|
from +0.459 to **+0.304**, against ESM-2's +0.445. ESMFold returns a confident folded
structure even at ΔΔG ≤ −2 (TM 0.984), so the tokens assert "still folded" for
destabilised proteins.

---

## Checked against PREREGISTRATION.md

| registered item | status |
|---|---|
| 1.1 `nonadditive_variance_fraction ≥ 0.15` (max) | **PASS** at 0.972 — but the metric is rank-correlated with real epistasis at **ρ = −0.49**, so it passed for the wrong reason. Threshold unchanged; reporting standard replaced |
| 1.1 `greedy_optimum_hit_rate ≤ 0.80` (min) | **PASS** at 0.00 |
| Primary landscape = TEV | **holds**, depth-filtered. Still the only landscape with total greedy failure alongside real structured epistasis |
| Negative controls ParD2, ParD3, DHFR | **hold**, and are now clean: path 1.000, zero traps, stack-top-k 1.00 |
| Greedy = 5 restarts, max 20 steps | **holds**. Supporting figures corrected twice; definition never changed |
| Oracle budget B = 100 | **incompatible** with the registered greedy, which costs order 10³ calls. Blocking on Step 6.3 |
| Rule 2: ≥3 seeds | honoured — 10 seeds throughout Phase 1 |
| Rule 7: no wild-type-anchored figure from an unverified wild type | **added this phase**, after the dead-variant bug |
| Rule 8: no non-additive figure without its reliability ceiling | **added this phase**, after the TEV read-noise bug |

**No threshold was edited in Phase 1.** Nine amendments are logged, all correcting a
measurement or a definition, none following from a model result — no model exists.

**The encoder is not a registered quantity**, so moving the default from SaProt to
ESM-2 needs no amendment. The pre-registration already names "encoder (D2 result)"
as the first thing to diagnose if H1 misses, which is exactly what happened.

---

## Four conditions carried into Phase 2

1. ~~**Default backbone becomes ESM-2 650M**~~ — **WITHDRAWN 2026-11-27, see below.**
   **SaProt stays the default; the finding is "never feed it predicted 3Di".**
2. **Fix the oracle-budget contradiction before Step 6.3.** Already recorded as a
   blocking requirement. `CountingOracle` exists to price every baseline in one
   currency.
3. **Resolve ParD2 and ParD3 variable positions.** Their wild types are settled, but
   without the site list their variants cannot be spliced into the parent, so they
   cannot be encoded as whole proteins for Phase 6. Needs Ding et al. 2023. CPU work.
4. **Do not reopen the TrpB3 landscapes** without resolving their wild types first.
   Their structured-epistasis figures are sound; nothing else about them is.

## One pattern worth naming

Four of this phase's findings have the same shape: **a decision was right, and the
reason recorded for it was wrong.** Pooling, structure invariance, greedy's success
rate, and the gate metric itself. In each case the error was an instrument artifact
read as a property of biology — read noise, a dead reference variant, a too-short
step limit, a predictor insensitive to destabilisation.

That is an argument for the diagnostics-first structure of this plan, and against
trusting any number whose floor has not been measured. Standing rules 7 and 8 were
added for exactly this, and the folding floor in Step 1.2 — folding each wild type
twice — is the pattern to repeat wherever a displacement is claimed.

---

## Reproduction

All four diagnostics and their write-ups:

```
python -I scripts/epistasis_audit.py          # 1.1, CPU, ~6 min
python -I scripts/hypercube_paths.py --segment 6_40
python -I scripts/d1_build_manifest.py        # 1.2 selection, CPU
python -I scripts/d1_fold.py                  # 1.2 folds, needs ~16 GB VRAM
python -I scripts/d2_latent_capacity.py       # 1.3, uses the folds
python -I scripts/d4_inverse_dynamics.py      # 1.4, laptop
```

`diagnostics/epistasis_audit.md`, `d1_structure.md`, `d2_latent.md`, `d4_inverse.md`.


---

## Correction, 2026-11-27: condition 1 was wrong

**The backbone conclusion in this report was drawn from the wrong comparison, and the
first trained reward head overturns it.**

Step 1.3 produced three numbers, not two:

| | Spearman vs \|ddG\| (displacement) | pooled Spearman (trained head) |
|---|---:|---:|
| **SaProt, no 3Di** | **+0.459** | **+0.337** |
| ESM-2 650M | +0.445 | +0.209 |
| SaProt + real 3Di | +0.304 | — |

This report compared ESM-2's +0.445 against SaProt's **+0.304** — ESM-2's only
configuration against SaProt's *worst* one — and concluded ESM-2 was ahead. Compared
like with like, SaProt without its structure channel was already ahead on the
displacement measure, and Step 2.1's trained stability head confirms it on every
statistic: pooled Spearman +0.337 against +0.209, per-protein median +0.753 against
+0.598, 100% of proteins positive against 92%.

**The real finding is narrower and stands undisturbed: do not feed a structure-aware
encoder predicted 3Di for stability work.** Supplying it costs SaProt 0.155. That is
the useful result; "switch to ESM-2" was never supported by it.

**Condition 1 is replaced by:** SaProt 650M remains the default, run **without** the
3Di channel until a structure predictor that responds to destabilisation is available.
ESM-2 is retained as the comparator and the question is still settled on H1.

This is the fifth Phase 1 finding of the shape named below — and the first where the
faulty reasoning was in this gate report rather than in the proposal it was auditing.
The specific failure was comparing a best configuration against a worst one. Worth
adding to the checklist: **when a model has a switch, compare both of its settings
against the alternative, not one.**