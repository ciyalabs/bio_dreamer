# ProteinDreamer — results summary

_Generated from `plan.yaml` (`ff022d6d97d8`). Do not edit by hand; regenerate with `bdtrack export`._

- Steps complete: **0/31**
- Runs recorded: **1** (0 failed, 1 from a dirty tree)
- Gates open: **8** · steps missing pre-registration: **2**

## Phase 0 — Groundwork

### 0.1 — Project bootstrap and code triage · `partial`
- Pending: `module_triage`

| Run | Experiment | Variant | Metrics | Code |
|---|---|---|---|---|
| `47f9d1c1` | env_bootstrap | cuda=13.0, python=3.12 | cuda_available=1, tests_passing=26, vram_gib=3.62 | 4a7f3b3 ⚠dirty |

### 0.2 — Data acquisition and licence register · `not-started`
- Pending: `hypercube_index`, `mutation_order_histogram`

### 0.3 — Leakage firewall **[GATE]** · `not-started`
- ⬜ `firewall_tests_passing == 1` → observed `—`
- Pending: `firewall_poison_test`

### 0.4 — Encoder harness and embedding cache · `not-started`
- Pending: `cache_throughput`, `per_residue_rank_test`

### 0.5 — Evaluation harness and the additive null **[GATE]** · `not-started`
- Pending: `additive_null_impl`, `metric_validation`

### 0.6 — Pre-register thresholds for H1-H4 **[GATE]** · `not-started`

## Phase 1 — Kill-shot diagnostics

### 1.1 — D3: Epistasis headroom audit **[GATE]** · `not-started`
- ⬜ `nonadditive_variance_fraction >= 0.15 [all]` → observed `—`
- ⬜ `greedy_optimum_hit_rate <= 0.8 [all]` → observed `—`
- Pending: `additive_fit`, `greedy_reachability`, `pairwise_fit`, `rse_count`, `stack_topk_headroom`

### 1.2 — D1: Structure invariance under point mutation · `not-started`
- Pending: `noise_floor`, `struct_displacement`, `substitution_stratified`

### 1.3 — D2: Latent capacity and dimension · `not-started`
- Pending: `backbone_sweep`, `dim_sweep`, `displacement_vs_ddg`

### 1.4 — D4: Inverse-dynamics feasibility probe · `not-started`
- Pending: `mutation_recovery`, `recovery_pooled_vs_residue`, `recovery_vs_length`

## Phase 2 — Reward head

### 2.1 — Implement and train the reward head · `not-started`
- Pending: `censoring_decision`, `reward_training`

### 2.2 — H1: Benchmark against the literature and the null **[GATE]** · `not-started`
- ⬜ `spearman_mean >= 0.5` → observed `—`
- ⬜ `margin_over_additive_null > 0` → observed `—`
- Pending: `higher_order_extrapolation`, `vs_additive_null`, `vs_frozen_regression`, `vs_literature`, `vs_zeroshot`

### 2.3 — Uncertainty calibration · `not-started`
- Pending: `calibration_curves`, `risk_coverage`, `shared_vs_independent_trunk`

## Phase 3 — World model B (deterministic)

### 3.1 — Implement World Model B · `not-started`
- Pending: `wm_b_training`

### 3.2 — Collapse monitoring **[GATE]** · `not-started`
- ⬜ `latent_effective_rank_ratio >= 0.5` → observed `—`
- Pending: `inverse_recovery_during_training`, `latent_rank`, `sigreg_stat`

### 3.3 — One-step accuracy · `not-started`
- Pending: `latent_one_step_error`, `reward_through_predicted`

### 3.4 — Multi-step rollout accuracy · `not-started`
- Pending: `reward_error_growth`, `rollout_2step_gb1`, `rollout_his3`

## Phase 4 — World model A (flow matching)

### 4.1 — Implement World Model A · `not-started`
- ⬜ `few_step_fidelity >= 0.95` → observed `—`
- Pending: `few_step_equivalence`, `flow_training`

### 4.2 — H2: A versus B · `not-started`
- Pending: `candidate_diversity`, `error_compounding_3arm`, `sampling_cost`, `uncertainty_quality`

## Phase 5 — Planner validation on synthetic landscapes

### 5.1 — Stand up SLIP · `not-started`
- Pending: `planner_correctness`, `slip_setup`

### 5.2 — Planning advantage versus ruggedness · `not-started`
- Pending: `horizon_sweep`, `ruggedness_sweep`

## Phase 6 — Policy and planning on real landscapes

### 6.1 — Trust region and model exploitation · `not-started`
- Pending: `admissibility_calibration`, `rejection_rates`, `trust_region_ablation`

### 6.2 — H3: Policy comparison · `not-started`
- Pending: `active_inference`, `gflownet`, `low_data_regime`, `mcts`, `policy_vs_calibration`, `ppo_entropy`

### 6.3 — H4: Multi-step planning **[GATE]** · `not-started`
- ⬜ `optimum_hit_rate_margin_over_additive > 0.1 [all]` → observed `—`
- ⬜ `correlation_with_additive <= 0.95 [max]` → observed `—`
- ⬜ `n_seeds >= 3 [min]` → observed `—`
- Pending: `h4_additive_correlation`, `h4_vs_additive_null`, `h4_vs_greedy_same_reward`, `h4_vs_published`, `h4_vs_random`, `h4_vs_stack_topk`

### 6.4 — Longer horizon · `not-started`
- Pending: `his3_dim5_8`

## Phase 7 — Ablations and robustness

### 7.1 — Required ablations · `not-started`
- Pending: `abl_backbone`, `abl_dynamics`, `abl_ema`, `abl_eta`, `abl_inverse_head`, `abl_pooling`, `abl_sampling_steps`, `abl_sigreg`, `abl_static_context`, `abl_trust_region`

### 7.2 — Seeds and variance **[GATE]** · `not-started`
- ⬜ `min_seeds_on_headline >= 3` → observed `—`
- Pending: `reproducibility_check`, `seed_replication`

## Phase 8 — Paper

### 8.1 — Decide the claim from the outcomes · `not-started`

### 8.2 — Venue · `not-started`

### 8.3 — Rescope options · `not-started`

### 8.4 — Reproducibility package · `not-started`
- Pending: `licence_clearance`, `repro_package`
