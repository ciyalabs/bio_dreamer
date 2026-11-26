# Pre-registration — ProteinDreamer, H1 to H4

**Registered:** 2026-10-27, before any world model exists.
**Deliverable of Step 0.6.** Machine-checked form lives in `plan.yaml`; `bdtrack status`
evaluates every threshold below against logged metrics.

---

## Why this file exists

A negative result is publishable only if the criterion for failure was fixed in advance.
Written afterwards, a threshold is a rationalisation. Every number here is committed and
dated, and **no threshold is edited once Phase 1 begins** — if one must change, the edit is
logged in the amendment table at the end with its reason, and the original stays visible.

The thresholds are grounded in measurements already taken (Steps 0.2–0.5) rather than chosen
for convenience. Where a number is a judgement rather than a measurement, it says so.

---

## Definitions fixed in advance

Ambiguity here is how a threshold gets reinterpreted later.

| Term | Fixed meaning |
|---|---|
| **Greedy** | Hill-climbing with **5 random restarts**, max 20 steps, on the same oracle as the method under test. *Not* single-step. Single-restart greedy finds the optimum on **2 of 5** landscapes and 5-restart greedy on **4 of 5**; on the primary landscape TEV the rate goes from 0.0 at 5 restarts to 0.6 at 100. The restart count decides the comparison, so it is fixed here. *Corrected 2026-11-26: the figure registered here was "0 of 5", measured at max_steps = 5 rather than the 20 fixed in this same row, and with one landscape transcribed wrong. The definition is unchanged.* |
| **Additive null** | `AdditiveNull.fit_on_singles`: a site-independent linear model fitted on the wild type and single mutants only, extrapolated to higher orders. |
| **Strongest baseline** | The **maximum** over {greedy, additive null, stack-top-k, random} on the metric in question. Not a chosen comparator. |
| **Oracle budget** | Number of *true* fitness evaluations a method may request. Headline budget **B = 100**, swept over {30, 100, 300}. |
| **Seed noise** | Pooled standard deviation across seeds of the same configuration. A difference counts only if it exceeds **2 × pooled sd**. |
| **Primary landscape** | **TEV, depth-filtered** (reads ≥ median = 209, n = 79,630). The landscape where greedy failure is *total* — 0 of 10 seeds at 5 restarts, 0.6 at 100, path accessibility 0.042 — with substantial structured epistasis present (out-of-fold pairwise increment +0.257). GB1 (+0.417, greedy 0.7) and TrpB3E (+0.196, greedy 0.4) combine the two as well; TEV is the extreme of the set. *Amended 2026-11-26: Step 0.5 chose TEV because its additive signal was absent, which Step 1.1 showed to be read noise — its reliability ceiling is R² ≤ 0.668. The choice stands; the reason for it does not, and the full-depth landscape must not be used.* |
| **Negative controls** | **ParD2, ParD3, DHFR (protein level)** — near-additive (cv1 0.839, 0.849, 0.784). A planner should show little or no advantage here. *DHFR added 2026-11-26; its nucleotide-level figures are not protein-level figures.* |
| **Oracle cost of a baseline** | Distinct sequences measured, counted by `CountingOracle`; repeat queries are free. The registered greedy costs order 10³ on a four-site landscape — see the amendment table. |

---

## H1 — the reward head is competitive and beats the null

**Step 2.2. Gate: yes.** Failing this stops the project at Phase 2.

| Condition | Threshold | Grounding |
|---|---|---|
| `spearman_mean` on the higher-order extrapolation protocol | **≥ 0.55** | Published band on the same protocol: ProteinNPT 0.56, MULTI-evolve 0.61, Kermut 0.62, Cerebra 0.65. 0.55 puts us at the lower edge of it. |
| `margin_over_additive_null` | **> 0.05** Spearman | A margin smaller than this is indistinguishable from the null given assay-to-assay spread. |
| `frac_assays_beating_null` | **≥ 0.70** | A mean can be carried by a few assays; most must beat the null. |
| `spearman_mean_vs_frozen_regression` | **> 0** | Must beat a plain regression head on the frozen encoder, or the world-model training added nothing. |

**Protocol:** train on <300 single and double mutants, test on unseen 3+ mutants — the setting
Cerebra and MULTI-evolve report, so the numbers are comparable. Uses the **`benchmark`
instance** (Step 0.3 two-instance rule); H4 must not use it.

**If missed:** do not proceed to Phase 3. The reward model is the foundation; a planner on a
weak oracle generates confident nonsense. Diagnose in this order: encoder (D2 result), adapter
capacity, loss (ranking vs regression), per-assay normalisation.

---

## H2 — flow matching beats deterministic dynamics

**Step 4.2. Gate: no** — this chooses an architecture, it does not stop the project.

| Condition | Threshold | Grounding |
|---|---|---|
| `error_growth_slope_ratio` (A ÷ B) | **≤ 0.75** | A must cut error compounding by ≥25% to justify generative dynamics. |
| `error_growth_gap_over_seed_sd` | **≥ 2.0** | Below this the gap is seed noise. |
| `n_seeds` | **≥ 3** | |
| `candidate_diversity_ratio` (A ÷ B) | **≥ 1.10** | The second claimed benefit of distributional dynamics. |

**Cost override.** If A's sampling cost exceeds **5×** B's *and* the accuracy gain is **< 10%**,
**B is declared the better architecture** regardless of the thresholds above. Measured on
wall-clock per rollout, not FLOPs. Step 4.1 already requires `few_step_fidelity ≥ 0.95`, so
A is only eligible for this comparison with few-step sampling working.

**If missed:** report B as sufficient and simplify the architecture. This is a legitimate,
publishable outcome — Step 0.1 found the flow-matching code already exists, so adopting A is
not a sunk cost that needs defending.

---

## H3 — policy classes, and whether calibration explains the ranking

**Step 6.2. Gate: no.**

H3 as stated in the proposal is deliberately weak: *no policy class will dominate when
uncertainty is miscalibrated*. It is a test that can come out either way.

| Condition | Threshold | Grounding |
|---|---|---|
| `policy_gap_over_seed_sd` | **≥ 2.0** to call any difference real | |
| `n_seeds` | **≥ 5** | Low-data regime is high-variance. |
| `spearman_calibration_vs_policy_rank` | **≥ 0.5** to support the stated mechanism | |
| `eta_zero_reported` | **== 1** | The η = 0 ablation is mandatory, not optional. |

**Expected outcome is explicitly allowed to be negative.** Champion et al. found an
expected-free-energy agent collapsing onto a single action while the reward-maximising variant
of the same architecture explored successfully. **If PPO with an entropy bonus matches or beats
Active Inference, that is the reported result.** No re-tuning of η to rescue it after seeing
the outcome.

**If missed:** report no dominance. The world model, not the policy, is the thesis.

---

## H4 — multi-step planning beats the strongest baseline

**Step 6.3. Gate: yes.** The project's central claim.

**This hypothesis was reframed on 2026-10-27, before any model existed**, because Step 0.5
measured greedy-with-restarts solving 4 of 5 landscapes outright. Stated as "peaks greedy
cannot reach", H4 would have been false on three landscapes before training began. It is now
stated **per oracle call**, which is also the question a real engineering campaign asks.

| Condition | Threshold | Grounding |
|---|---|---|
| `optimum_hit_rate_margin_over_best_baseline` at B=100, on **TEV** | **> 0.10** | |
| `best_fitness_margin_over_best_baseline` at B=100, on TEV | **> 0.05** (fraction of max) | Hit rate alone is brittle on a single optimum. |
| `correlation_with_additive` | **≤ 0.95** | **Falsifies H4 regardless of absolute performance.** A model that correlates this closely with the null is the null. |
| `n_seeds` | **≥ 5** | |
| `margin_on_negative_controls` (ParD2, ParD3) | **≤ 0.05** | **A planner that also wins on the near-additive controls is suspect**, not impressive — the likely explanation is reward-model exploitation, not epistatic reasoning. |

**Budget sweep.** Report margin at B ∈ {30, 100, 300}. Greedy needs ≈30 true evaluations to
solve TrpB4, so a method that only wins at very large budgets has not shown what H4 claims.

**Secondary landscapes:** TrpB4 and GB1, reported but not decisive — greedy already solves
both at 5 restarts.

**If missed:** H4 is reported as falsified. Per `RESEARCH_PLAN.md` Step 8.1 this is the
"accurate latent dynamics, but planning gains are explained by additivity" paper, which given
the Visani and Wittmann critiques is a genuine contribution. The trajectory benchmark
(Contribution 2) stands in every branch.

---

## Step 1.1 — the precondition gate

Step 1.1 asks whether there is anything to find at all. Its thresholds are **existential, not
universal**: the project needs *at least one* landscape with headroom, not all of them.

| Condition | Aggregate | Threshold |
|---|---|---|
| `nonadditive_variance_fraction` | **max** | ≥ 0.15 — at least one landscape is non-additive |
| `greedy_optimum_hit_rate` (5 restarts, true oracle) | **min** | ≤ 0.8 — at least one landscape where greedy fails |

Measured already in Step 0.5: TEV satisfies both (0.952 non-additive, greedy 0/5). The gate is
expected to pass on TEV alone, and that expectation is recorded here rather than discovered.

**Outcome (2026-11-26): GO.** `nonadditive_variance_fraction` max 0.972, `greedy_optimum_hit_rate`
min 0.00 — both pass. The expectation above was half right: TEV did satisfy both conditions, but
its 0.952 was read noise rather than epistasis, and the gate's registered metric turned out to be
negatively rank-correlated with real epistasis. The substantive answer comes from elsewhere —
greedy solves 91% of 3-site HIS3 hypercubes and 24% of 9-site ones. See `diagnostics/epistasis_audit.md`.

**Read the dead fraction before the R².** GB1 is 92% dead, so much of its apparent
non-additivity is a floored distribution. Any non-additive figure quoted without its dead
fraction is inadmissible.

---

## Standing rules

1. **Every epistasis or planning claim reports the additive null**, its margin, and the
   model–null correlation. `report_additive_comparison()` emits all three; it is not optional.
2. **No headline number from a single seed.** ProteinJEPA's gains failed to replicate across
   seeds (p = 0.059); single-seed results are not evidence.
3. **No headline number from a dirty working tree.** The tracker flags these.
4. **Uncertainty calibration is a primary result**, reported alongside accuracy.
5. **No biological claim rests on a simulated label** (SLIP, FoldX, pseudo-labels).
6. **The `benchmark` instance never produces a Phase 6 result** (Step 0.3 firewall).
7. **No non-additive figure is quoted without its dead fraction and its reliability ceiling.**
   Step 1.1 found all three of noise, floor censoring and codon redundancy masquerading as
   epistasis. The out-of-fold pairwise increment is the figure that separates them.
8. **Every baseline reports its oracle cost** in distinct evaluations, in the same currency as
   the budget the method under test is held to.

---

## Amendments

Any change after Phase 1 begins is logged here with its date and reason. The original
threshold stays visible.

| Date | What | From | To | Reason |
|---|---|---|---|---|
| 2026-11-26 | **TEV data rule** | full landscape | **depth-filtered**, reads ≥ median (209) | Step 1.1: TEV's reliability ceiling is R² ≤ 0.668. At full depth its out-of-fold pairwise increment is +0.030; filtered it is +0.257. The full-depth figure measures read noise |
| 2026-11-26 | **TEV's justification as primary** | "additive signal absent, null Spearman −0.005" | "total greedy failure with structured epistasis present" | The registered reason was a noise artifact. TEV stays primary; the stated ground for it was wrong |
| 2026-11-26 | **DHFR** | candidate landscape | **third negative control**, protein level only | Step 1.1: at protein level DHFR is 78% additive (cv1 0.784) and greedy solves it 0.9. Its nucleotide-level epistasis is a noise-averaging artifact of 16 synonymous codons per protein |
| 2026-11-26 | **Landscape set** | all 16 SSMuLA landscapes | **T7, TrpB3A/B/C/F/H dropped** | Out-of-fold pairwise increment ≤ 0.043. A method cannot be shown to use epistasis on a landscape whose epistasis is not measurable |
| 2026-11-26 | **Reporting metric for epistasis** | `1 − additive R²` (full fit) | out-of-fold pairwise increment | `1 − additive R²` is rank-correlated with real epistasis at ρ = −0.49 (p = 0.052, n = 16). **Gate thresholds are unchanged**; this governs reporting and landscape selection only |
| 2026-11-26 | **Single-restart greedy's hit rate** | 0 of 5 landscapes | **2 of 5** | The registered figure was measured at max_steps = 5, not the 20 fixed in the same row, and ParD2 was transcribed as a miss when the tracker recorded a hit. TrpB4's greedy walk reaches the optimum from the wild type in 7 steps. The 5-restart definition is unaffected |
| 2026-11-26 | **Greedy's oracle cost** | unspecified | **counted, and must be reconciled with B** | The registered greedy (5 restarts × 20 steps) costs order 10³ evaluations on a four-site landscape, against a registered budget of B = 100. Blocking requirement on Step 6.3 |

**No threshold was edited.** All six entries correct a measurement error or a definition.
None follows from a model result, because no model exists — these come from the step whose
purpose is to find such errors. They are logged as amendments rather than pre-Phase-1
revisions only because Step 1.1 is itself the first step of Phase 1.

**Pre-Phase-1 revisions**, made while this file was being written and therefore not amendments:

| Date | What | Reason |
|---|---|---|
| 2026-10-27 | H4 reframed from "peaks greedy cannot reach" to a per-oracle-budget margin | Step 0.5 measured greedy-with-restarts solving 4/5 landscapes |
| 2026-10-27 | "Greedy" fixed as 5-restart, not single-step | Single-step finds the optimum on 0/5 landscapes; the restart count decides the comparison |
| 2026-10-27 | Primary landscape TrpB4 → TEV | Only landscape with both absent additive signal and failing greedy |
| 2026-10-27 | Negative-control condition added to H4 | A planner winning on near-additive controls indicates reward exploitation |
| 2026-10-27 | H4 seeds 3 → 5 | Consistency with H3 |
| 2026-10-27 | Step 1.1 aggregates `all` → `max`/`min` | The gate is existential: one landscape with headroom suffices |
