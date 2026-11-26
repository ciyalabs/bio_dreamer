# Step 1.1 — D3: epistasis headroom audit

**Run:** 2026-11-26 · `scripts/epistasis_audit.py`, `scripts/hypercube_paths.py`
**Deliverable of Step 1.1.** 16 SSMuLA landscapes, plus 1,818 HIS3 hypercubes at dimensions 3–9 across two segments.

---

## ⚠ Correction, 2026-11-26 (same day)

**`Landscape` was using a dead variant as the wild type on every landscape.** It
inferred the wild type as the modal residue per site, which is close to arbitrary on a
combinatorially complete landscape — every residue appears at every site by design — and
in practice returns something non-functional:

| landscape | true wild type | fitness | inferred (modal) | fitness |
|---|---|---:|---|---:|
| GB1 | `VDGV` *(HD==0 row)* | **1.0000** | `WLVS` | 0.0039 |
| TEV | `TDHS` *(Mutation Distance==0)* | 0.4144 | `FLPP` | −0.0162 |
| TrpB4 | `VFVS` *(Johnston 2024, verified present)* | 0.4081 | `GSHL` | 0.0099 |

**Unaffected:** both gate metrics (`nonadditive_variance_fraction` uses no wild type;
`greedy_optimum_hit_rate`'s min is still TEV at 0.00), the whole interaction-order
decomposition and every `+pair` figure, the landscape-set decisions, and the HIS3 depth
curve — hypercube landscapes construct their wild type explicitly from the corner
geometry, never by the modal rule.

**Corrected**, for the three landscapes whose wild type could be verified:

| metric | GB1 | TEV | TrpB4 |
|---|---:|---:|---:|
| greedy@1 | 0.0 ← 0.0 | 0.0 ← 0.0 | **0.0 ← 1.0** |
| greedy@5 | 0.7 ← 0.7 | 0.0 ← 0.0 | **0.4 ← 1.0** |
| path accessibility | **0.208** ← 0.583 | **0.000** ← 0.042 | **0.000** ← 0.625 |
| reciprocal-sign pairs | **14** ← 87 | **19** ← 372 | **30** ← 67 |
| stack-top-k gain | **0.39** ← 0.07 | **0.12** ← 0.08 | **0.40** ← 0.03 |
| singles→order≥2 ρ | **0.245** ← 0.476 | **0.134** ← −0.005 | **0.196** ← 0.070 |
| fraction above wild type | **2.4%** ← 45.2% | **0.4%** ← 50.5% | **0.7%** ← 56.8% |

That last row was the visible tell, and it was in the Step 0.2 audit the whole time: a
four-site combinatorial landscape in which **half the variants beat the wild type** is
not biologically plausible. From the verified wild type it is 0.4–2.4%.

The corrected numbers make the case for planning **stronger**, not weaker: no monotone
shortest path reaches the optimum on TEV or TrpB4 at all, and 5-restart greedy solves
none of the three reliably. But the Step 0.5 measurement that drove H4's reframing —
"greedy with 5 restarts solves 4 of 5 landscapes" — was an artifact of walking from a
dead variant. H4's per-oracle-budget framing survives and is better supported; its
stated justification does not.

**Still provisional.** ParD2, ParD3, DHFR, T7 and the nine TrpB 3-site landscapes have
no distance column, no registry entry, and alphabetically sorted CSVs, so their wild
type cannot be recovered from shipped data. `Landscape` now warns and sets
`wild_type_source="modal"` for them. Their `+pair` figures are unaffected — so their
roles in the landscape set, including ParD2/ParD3 as near-additive controls, stand —
but every wild-type-anchored column below is suspect for them. Resolving ParD2/ParD3 is
a prerequisite for H4, since a negative control needs a correct starting point.

---

## Verdict

**Gate: GO**, on both pre-registered conditions.

| Pre-registered condition | Aggregate | Threshold | Observed | |
|---|---|---|---|---|
| `nonadditive_variance_fraction` | max | ≥ 0.15 | **0.972** (TrpB3H) | pass |
| `greedy_optimum_hit_rate` | min | ≤ 0.80 | **0.00** (TEV, T7, DHFR) | pass |

The gate passes, and **the metric it passes on turns out to be the wrong metric.**
`1 − additive R²` is rank-correlated with real epistasis at **ρ = −0.49** (p = 0.052,
n = 16) — not merely uninformative but pointing the wrong way. The three landscapes it
ranks highest have almost no exploitable epistasis; the three with the most rank 5th, 6th
and 7th by that measure.

The substantive question — *is there structured epistasis that defeats greedy?* — is
answered **yes**, but by different landscapes than Step 0.5 identified, and most clearly
by data no SSMuLA landscape can provide:

> **Greedy hill-climbing finds the best corner of 91% of 3-site HIS3 hypercubes and 24%
> of 9-site ones. Monotone path accessibility falls from 0.61 to 0.095 over the same
> range.** The premise behind multi-step planning holds, and it strengthens with depth.

---

## What was wrong with the Step 0.2 numbers

Step 0.2 reported `1 − additive R²` from a full-data fit and called it the non-additive
variance fraction. Three different things produce a low additive R², and only one of them
is an opportunity:

| Cause | Looks like epistasis | Is exploitable | Example |
|---|---|---|---|
| Structured epistasis | yes | **yes** | TrpB4, GB1 |
| Measurement noise | yes | no | TEV, T7 |
| Floor censoring | yes | no | GB1's 92% dead |
| Codon redundancy | yes | no | DHFR at nucleotide level |

The test that separates them is whether the non-additive variance is *predictable*:
fit interaction orders 1, 2 and 3 and score them **out of fold**. Real epistasis shows a
large out-of-fold increment from order 1 to order 2. Noise shows none at any order.

Ridge strength is selected inside each outer fold on an inner split of that fold's own
training data, so no test point influences the model that scores it. The saturated top
order is excluded: on a complete landscape a held-out genotype's top-order cell appears in
no training row, so its CV score measures the design rather than the landscape. On TrpB4
the excluded order 4 moves the score by +0.004, confirming it carries nothing.

---

## The 16 landscapes

`cvN` is out-of-fold R² at interaction order N. **`+pair` = cv2 − cv1 is the column that
matters**: predictable variance that site-independence cannot reach.

| landscape | n | sites | dead | ceiling | 1−R² | cv1 | cv2 | cv3 | **+pair** | g@1 | **g@5** | g@20 | g@100 | path | RSE | stk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DHFR *(nt)* | 261,333 | 9 | 0% | — | 0.723 | 0.277 | 0.729 | 0.825 | **+0.452** | 0.0 | 0.0 | 0.2 | 0.7 | 0.035 | 50 | 0.09 |
| GB1 | 149,361 | 4 | 92% | — | 0.806 | 0.193 | 0.610 | 0.885 | **+0.417** | 0.0 | 0.7 | 1.0 | 1.0 | 0.583 | 87 | 0.07 |
| TrpB4 | 159,129 | 4 | 0% | — | 0.743 | 0.256 | 0.600 | 0.759 | **+0.344** | 1.0 | 1.0 | 1.0 | 1.0 | 0.625 | 67 | 0.03 |
| TrpB3I | 7,784 | 3 | 0% | — | 0.465 | 0.527 | 0.782 | 0.783 | +0.256 | 1.0 | 1.0 | 1.0 | 1.0 | 1.000 | 5 | 0.92 |
| TrpB3D | 7,763 | 3 | 0% | — | 0.749 | 0.235 | 0.456 | 0.472 | +0.221 | 0.0 | 0.7 | 1.0 | 1.0 | 1.000 | 0 | 0.22 |
| TrpB3E | 7,964 | 3 | 0% | — | 0.947 | 0.041 | 0.237 | 0.236 | +0.196 | 0.0 | 0.4 | 0.9 | 1.0 | 0.333 | 113 | 0.08 |
| TrpB3G | 7,891 | 3 | 0% | — | 0.855 | 0.130 | 0.247 | 0.241 | +0.117 | 0.0 | 0.8 | 1.0 | 1.0 | 0.667 | 11 | 0.07 |
| *ParD2* | 7,882 | 3 | 0% | — | 0.158 | 0.839 | 0.949 | 0.949 | +0.110 | 1.0 | 1.0 | 1.0 | 1.0 | 1.000 | 0 | 1.00 |
| *ParD3* | 7,882 | 3 | 0% | — | 0.148 | 0.849 | 0.937 | 0.937 | +0.088 | 0.0 | 1.0 | 1.0 | 1.0 | 0.500 | 5 | 0.98 |
| TrpB3H | 7,716 | 3 | 0% | — | **0.972** | 0.016 | 0.059 | 0.062 | +0.043 | 1.0 | 1.0 | 1.0 | 1.0 | 0.167 | 144 | 0.09 |
| TEV | 159,132 | 4 | 0% | **0.67** | **0.952** | 0.047 | 0.077 | 0.077 | +0.030 | 0.0 | **0.0** | 0.0 | 0.6 | 0.042 | 372 | 0.08 |
| TrpB3F | 7,737 | 3 | 0% | — | 0.877 | 0.109 | 0.128 | 0.130 | +0.019 | 1.0 | 1.0 | 1.0 | 1.0 | 1.000 | 62 | 1.00 |
| TrpB3A | 7,971 | 3 | 0% | — | 0.780 | 0.205 | 0.211 | 0.223 | +0.006 | 0.0 | 0.4 | 1.0 | 1.0 | 0.167 | 177 | 0.05 |
| TrpB3B | 7,996 | 3 | 0% | — | 0.900 | 0.086 | 0.092 | 0.091 | +0.005 | 0.0 | 0.4 | 0.6 | 1.0 | 0.000 | 78 | 0.09 |
| T7 | 6,725 | 3 | 17% | **0.73** | **0.959** | 0.023 | 0.027 | 0.027 | +0.003 | 0.0 | 0.0 | 0.3 | 0.7 | 0.000 | 99 | 0.49 |
| TrpB3C | 7,994 | 3 | 0% | — | 0.834 | 0.150 | 0.152 | 0.152 | +0.002 | 0.0 | 0.2 | 0.8 | 1.0 | 0.000 | 23 | 0.10 |

*Italic = pre-registered negative control. `ceiling` = highest R² the published error bars
permit. `g@N` = greedy optimum hit rate at N restarts, 10 seeds, max 20 steps. `path` =
fraction of monotone shortest paths to the optimum. `RSE` = reciprocal-sign-epistasis pairs.
`stk` = fraction of achievable gain captured by stacking the top-k singles.*

Sorted by `+pair`, the table reads top-to-bottom from opportunity to noise. Sorted by
`1−R²` — the Step 0.2 column — TrpB3H, T7 and TEV lead, with +0.043, +0.003 and +0.030 of
actual structure between them.

---

## The three artifacts, separated

### TEV: noise, not epistasis

TEV publishes a per-variant variance, which bounds every R² on it at **0.668**. A third of
its "non-additive variance" is unexplainable by any model. Filtering on read depth, with
nothing else changed:

| reads ≥ | n | noise fraction | cv1 | cv2 | **+pair** |
|---|---:|---:|---:|---:|---:|
| 0 | 159,132 | 0.332 | 0.047 | 0.077 | +0.030 |
| 50 | 123,270 | 0.158 | 0.144 | 0.313 | +0.169 |
| 200 | 81,417 | 0.076 | 0.231 | 0.485 | +0.254 |
| 500 | 42,981 | 0.032 | 0.345 | 0.656 | +0.311 |
| 1,000 | 20,043 | 0.015 | 0.426 | 0.755 | +0.329 |

At above-median depth (≥ 209 reads, n = 79,630) TEV is **+0.257** — comparable to TrpB4 and
ten times its full-depth figure. Its floor-removed `1−R²` is 0.955 against 0.952 raw, so
none of the inflation is censoring: it is **all** noise.

**TEV remains the right primary landscape**, but for a reason Step 0.5 did not have. The
registered justification — "the only landscape where the additive signal is absent (null
Spearman −0.005)" — was measuring read noise. The correct justification is that TEV is the
landscape where greedy failure is **total** — 0.0 at 5 restarts across 10 seeds, 0.6 even at
100 restarts, path accessibility 0.042 — while substantial structured epistasis is present
(+0.257 filtered). It is not the only landscape combining the two: GB1 (+0.417, greedy 0.7)
and TrpB3E (+0.196, greedy 0.4) also do. TEV is the extreme of the set, which is what makes
it the sharpest test.

### GB1: floor, not noise — and the epistasis is real

GB1 is 92% dead, and the floor accounts for a fifth of its raw figure: `1−R²` drops from
0.806 to **0.634** once variants at the detection limit are removed. But its epistasis
survives depth filtering almost untouched (**+0.421** at above-median depth against +0.417
at full depth), so unlike TEV its signal is not noise-limited. GB1 is a genuinely,
strongly epistatic landscape whose weakness is censoring, not precision. Median selected
count is **4** reads per variant, with 19.7% at zero, so per-variant fitness is coarse even
where it is not floored.

### DHFR: a different landscape at protein level than at nucleotide level

DHFR looked like the strongest result in the set — +0.452 structured epistasis, greedy 0.0,
path accessibility 0.035. It is 9 **nucleotide** sites: three codons.

| | nucleotide level | protein level |
|---|---:|---:|
| genotypes | 261,333 | 8,000 (**100% complete**) |
| sites | 9 (nt) | 3 (aa) |
| 1 − additive R² | 0.723 | **0.211** |
| cv1 | 0.277 | **0.784** |
| cv2 | 0.729 | 0.972 |
| **+pair** | **+0.452** | +0.187 |
| greedy @5 restarts | 0.0 | **0.9** |
| path accessibility | 0.035 | 0.500 |
| dead fraction | 0% | 0.1% |

13.5% of variants carry a stop codon and are excluded. Of the remaining nucleotide-level
variance only **6.3%** is synonymous, so the epistasis is not a codon-usage effect — but
averaging the median 16 synonymous codons per protein cuts the measurement noise about
16-fold, and once it is gone the protein landscape is **78% additive and greedy solves it
90% of the time.**

The project's encoder sees proteins. **DHFR is therefore a third near-additive negative
control, not a candidate primary landscape**, and its nucleotide-level numbers must never
be quoted as protein-level ones. It is an unusually clean control: fully complete over
8,000 variants, 0.1% dead, with built-in replicate error bars.

---

## Depth: the result SSMuLA cannot give

Every SSMuLA landscape has at most four mutable sites, so every path-level question the
project could previously ask was capped at horizon 4. The HIS3 extraction supplies complete
2^d hypercubes to dimension 9. For these the variance decomposition is **exact** — each
site has two states and every corner is measured, so the Walsh–Hadamard transform
diagonalises it; no fitting, no regularisation, nothing to tune.

**Segment 6–40**, 200 hypercubes per dimension (100 at 8 and 9), 10 greedy seeds each:

| d | available | order 1 | order 2 | order 3+ | **greedy** | never solves | **path** | no monotone path | has RSE | greedy calls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3,478,667 | 0.602 | 0.304 | 0.094 | **0.906** | 0.00 | **0.609** | 0.26 | 0.14 | 8 |
| 4 | 3,898,512 | 0.477 | 0.310 | 0.214 | 0.918 | 0.00 | 0.535 | 0.26 | 0.23 | 15 |
| 5 | 2,576,913 | 0.393 | 0.267 | 0.340 | 0.845 | 0.00 | 0.461 | 0.32 | 0.32 | 26 |
| 6 | 988,823 | 0.310 | 0.243 | 0.447 | 0.723 | 0.01 | 0.329 | 0.42 | 0.45 | 41 |
| 7 | 209,135 | 0.285 | 0.232 | 0.483 | 0.549 | 0.01 | 0.283 | 0.46 | 0.54 | 60 |
| 8 | 21,741 | 0.269 | 0.229 | 0.502 | 0.325 | 0.07 | 0.156 | 0.64 | 0.72 | 78 |
| 9 | 817 | 0.278 | 0.239 | 0.483 | **0.242** | 0.12 | **0.095** | 0.62 | 0.63 | 97 |

**Cross-check, segment 140–160** (a different mutational context, 26× fewer genotypes):
order-1 fraction 0.418 → 0.164 and greedy 0.991 → 0.678 over dimensions 3–6. The absolute
additivity differs between segments — 0.602 against 0.418 at dimension 3 — but every trend
is the same.

Three things to read out of this table:

**The variance composition stabilises; reachability does not.** Order-1 share plateaus near
0.27 from dimension 7 and order-3+ near 0.49, yet greedy keeps falling (0.55 → 0.33 → 0.24)
and path accessibility keeps collapsing (0.28 → 0.16 → 0.095). The landscapes do not become
proportionally more epistatic with depth — the number of ways to get trapped grows
combinatorially at roughly fixed per-order variance shares. This is the sharpest statement
the project can make about *why* depth needs planning.

**Path accessibility understates greedy's reach.** At dimension 3, 26% of hypercubes have no
monotone shortest path at all, yet greedy solves 99%: it is not restricted to shortest
paths and routes around valleys through other corners. The classical accessibility statistic
and "greedy fails" are different measurements, and conflating them would overstate the case
for planning. Both are reported here for that reason.

**Reciprocal sign epistasis becomes the common case.** The trap greedy provably cannot cross
is present in 14% of 3-site hypercubes and 63–72% of 8- and 9-site ones.

Dimension 9 is sampled at 100 of 817, so its figures carry the widest sampling error; the
dip in RSE prevalence from 0.72 to 0.63 is within that.

---

## The landscape set after the audit

| | landscapes | why |
|---|---|---|
| **Primary** | TEV *(depth-filtered)* | the only one with both structured epistasis and greedy failure |
| **Strong secondary** | TrpB4, GB1 | +0.344 / +0.417 structured; greedy solves them, so not decisive for H4 |
| **Usable** | TrpB3D, TrpB3E, TrpB3G, TrpB3I | +0.12 to +0.26 structured |
| **Negative controls** | ParD2, ParD3, **DHFR (protein)** | near-additive; a planner should gain little here |
| **Dropped** | T7, TrpB3A, TrpB3B, TrpB3C, TrpB3F, TrpB3H | structured epistasis ≤ 0.043 — nothing to learn, so nothing to demonstrate |
| **Depth** | HIS3 hypercubes, d = 3–9 | the only ≥5-step ground truth; 13.5M available |

Dropping six of sixteen landscapes is the audit's main practical consequence. A method
cannot be shown to use epistasis on a landscape whose epistasis is not measurable, and
including them would have diluted every aggregate in the paper with noise.

---

## A protocol inconsistency, found before it mattered

H4 is pre-registered as a margin over the strongest baseline **at a fixed oracle budget of
B = 100 true evaluations**. "Greedy" is pre-registered separately as **5 random restarts of
at most 20 steps**. Those two definitions were never reconciled, and they are not
compatible.

One greedy step on a four-site landscape over 20 residues evaluates **76** neighbours. The
measured cost of the registered greedy, counting distinct sequences only (a real campaign
remembers what it measured):

- 4-site SSMuLA landscapes: order **10³** distinct evaluations at 5 restarts
- HIS3 hypercubes: 8 at dimension 3 rising to 97 at dimension 9, because each site has only
  two states and a corner has at most d neighbours

So on the SSMuLA landscapes the registered greedy spends tens of times the budget the
method under test is held to, and on the hypercubes it spends roughly the budget. Comparing
a method limited to 100 evaluations against a baseline that used thousands is not a
comparison — in either direction.

This is an **unfalsifiability risk in the easy direction**: left unfixed, H4 would be won
trivially by holding the planner to 100 calls and the baseline to none. Reconciling it is
now a blocking requirement on Step 6.3, with `CountingOracle` in place to price every
baseline in the same currency. Note what the budget implies: a single-mutant scan alone
costs 76 of the 100 calls at four sites, so B = 100 leaves 24 for everything else, and the
registered sweep point B = 30 does not pay for the scan at all.

---

## Amendments logged

Five entries added to `PREREGISTRATION.md`. All five correct a measurement error or a
definition, none follows from a model result — **no model exists.** They are logged as
amendments rather than pre-Phase-1 revisions because Step 1.1 is the first step of Phase 1.

---

## Reproduction

```
python -I scripts/epistasis_audit.py                     # 16 landscapes, ~6 min
python -I scripts/hypercube_paths.py --segment 6_40      # dimensions 3-9, ~1 min
python -I scripts/hypercube_paths.py --segment 140_160 --dims 3 4 5 6
```

Outputs `data/processed/epistasis_audit.json` and `data/processed/hypercube_paths.json`;
tracker runs under `1.1`. The hypercube analysis needs `out_6_40/hypercubes_{3..7}.txt`,
which are gitignored at 856 MB and regenerable per `DATA_REGISTER.md`; dimensions 8 and 9
are committed.

`biodreamer/eval/decompose.py` and `biodreamer/eval/hypercube.py` carry 59 tests between
them, including three landscapes with answers known by construction (additive, pairwise,
pure noise) and a case that reproduces the TEV failure mode directly — an additive
landscape plus heavy noise reporting a 0.8 "non-additive fraction" with no epistasis present.
