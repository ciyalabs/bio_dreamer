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
| **Greedy** | Hill-climbing with **5 random restarts**, max 20 steps, on the same oracle as the method under test. *Not* single-step. Step 0.5 measured single-step greedy finding the optimum on **0 of 5** landscapes and 5-restart greedy on **4 of 5** — the restart count decides the comparison, so it is fixed here. |
| **Additive null** | `AdditiveNull.fit_on_singles`: a site-independent linear model fitted on the wild type and single mutants only, extrapolated to higher orders. |
| **Strongest baseline** | The **maximum** over {greedy, additive null, stack-top-k, random} on the metric in question. Not a chosen comparator. |
| **Oracle budget** | Number of *true* fitness evaluations a method may request. Headline budget **B = 100**, swept over {30, 100, 300}. |
| **Seed noise** | Pooled standard deviation across seeds of the same configuration. A difference counts only if it exceeds **2 × pooled sd**. |
| **Primary landscape** | **TEV.** Chosen in Step 0.5 because it is the only landscape where the additive signal is absent (null Spearman −0.005 on order ≥ 2) *and* greedy fails (0/5 at 5 restarts, 2/5 at 100). |
| **Negative controls** | **ParD2, ParD3** — near-additive (null Spearman 0.844, 0.890). A planner should show little or no advantage here. |

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

---

## Amendments

Any change after Phase 1 begins is logged here with its date and reason. The original
threshold stays visible.

| Date | Threshold | From | To | Reason |
|---|---|---|---|---|
| — | — | — | — | *(none yet)* |

**Pre-Phase-1 revisions**, made while this file was being written and therefore not amendments:

| Date | What | Reason |
|---|---|---|
| 2026-10-27 | H4 reframed from "peaks greedy cannot reach" to a per-oracle-budget margin | Step 0.5 measured greedy-with-restarts solving 4/5 landscapes |
| 2026-10-27 | "Greedy" fixed as 5-restart, not single-step | Single-step finds the optimum on 0/5 landscapes; the restart count decides the comparison |
| 2026-10-27 | Primary landscape TrpB4 → TEV | Only landscape with both absent additive signal and failing greedy |
| 2026-10-27 | Negative-control condition added to H4 | A planner winning on near-additive controls indicates reward exploitation |
| 2026-10-27 | H4 seeds 3 → 5 | Consistency with H3 |
| 2026-10-27 | Step 1.1 aggregates `all` → `max`/`min` | The gate is existential: one landscape with headroom suffices |
