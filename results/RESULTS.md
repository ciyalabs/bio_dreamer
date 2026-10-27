# ProteinDreamer — results summary

_Generated from `plan.yaml` (`f9585ee5523c`). Do not edit by hand; regenerate with `bdtrack export`._

- Steps complete: **6/31**
- Runs recorded: **34** (0 failed, 34 from a dirty tree)
- Gates open: **5** · steps missing pre-registration: **0**

## Phase 0 — Groundwork

### 0.1 — Project bootstrap and code triage · `complete`

| Run | Experiment | Variant | Metrics | Code |
|---|---|---|---|---|
| `e5ff3848` | module_triage | scope=protein_dreamer/legacy | delete=3, earlier_triage_errors=3, extend=3, frac_surviving=0.4, lines_total=5998, modules_total=19, new_components=3, reuse=9, rewrite=4 | 804b0cf ⚠dirty |
| `47f9d1c1` | env_bootstrap | cuda=13.0, python=3.12 | cuda_available=1, tests_passing=26, vram_gib=3.62 | 4a7f3b3 ⚠dirty |

### 0.2 — Data acquisition and licence register · `complete`

| Run | Experiment | Variant | Metrics | Code |
|---|---|---|---|---|
| `b2a28cdd` | hypercube_index | dataset=HIS3, segment=80_100 | dim_1=1196, dim_2=2003, dim_3=1209, dim_4=238, genotypes=242, hypercubes_total=4646, max_dimension=4, multiplication_factor=19.2 | b63ad0f ⚠dirty |
| `25655521` | hypercube_index | dataset=HIS3, segment=60_80 | dim_1=1211, dim_2=1954, dim_3=1266, dim_4=241, dim_5=30, dim_6=1, genotypes=264, hypercubes_total=4703, max_dimension=6, multiplication_factor=17.81 | b63ad0f ⚠dirty |
| `659042b4` | hypercube_index | dataset=HIS3, segment=160_180 | dim_1=4185, dim_2=8085, dim_3=7324, dim_4=3241, dim_5=682, dim_6=51, genotypes=799, hypercubes_total=2.357e+04, max_dimension=6, multiplication_factor=29.5 | b63ad0f ⚠dirty |
| `b5d4ab20` | hypercube_index | dataset=HIS3, segment=140_160 | dim_1=4681, dim_2=9191, dim_3=8190, dim_4=3154, dim_5=446, dim_6=18, genotypes=916, hypercubes_total=2.568e+04, max_dimension=6, multiplication_factor=28.03 | b63ad0f ⚠dirty |
| `cbafeb86` | mutation_order_histogram | landscape=TrpB4, source=SSMuLA | additive_r2=0.2568, completeness_vs_20=0.9946, frac_above_wild_type=0.5676, frac_dead=0, n_genotypes=1.591e+05, n_sites=4, nonadditive_variance_fraction=0.7432 | d1884c4 ⚠dirty |
| `83fe4372` | mutation_order_histogram | landscape=TrpB3I, source=SSMuLA | additive_r2=0.5349, completeness_vs_20=0.973, frac_above_wild_type=0.0084, frac_dead=0.0001, n_genotypes=7784, n_sites=3, nonadditive_variance_fraction=0.4651 | d1884c4 ⚠dirty |
| `7da894f7` | mutation_order_histogram | landscape=TrpB3H, source=SSMuLA | additive_r2=0.0277, completeness_vs_20=0.9645, frac_above_wild_type=0.6109, frac_dead=0.0001, n_genotypes=7716, n_sites=3, nonadditive_variance_fraction=0.9723 | d1884c4 ⚠dirty |
| `6f2a61a8` | mutation_order_histogram | landscape=TrpB3G, source=SSMuLA | additive_r2=0.1453, completeness_vs_20=0.9864, frac_above_wild_type=0.7263, frac_dead=0.0003, n_genotypes=7891, n_sites=3, nonadditive_variance_fraction=0.8547 | d1884c4 ⚠dirty |
| `2a30c2d6` | mutation_order_histogram | landscape=TrpB3F, source=SSMuLA | additive_r2=0.1227, completeness_vs_20=0.9671, frac_above_wild_type=0.5961, frac_dead=0.0001, n_genotypes=7737, n_sites=3, nonadditive_variance_fraction=0.8773 | d1884c4 ⚠dirty |
| `e15003b9` | mutation_order_histogram | landscape=TrpB3E, source=SSMuLA | additive_r2=0.053, completeness_vs_20=0.9955, frac_above_wild_type=0.4282, frac_dead=0.0001, n_genotypes=7964, n_sites=3, nonadditive_variance_fraction=0.947 | d1884c4 ⚠dirty |
| `d9dd6157` | mutation_order_histogram | landscape=TrpB3D, source=SSMuLA | additive_r2=0.2507, completeness_vs_20=0.9704, frac_above_wild_type=0.9728, frac_dead=0.0003, n_genotypes=7763, n_sites=3, nonadditive_variance_fraction=0.7493 | d1884c4 ⚠dirty |
| `c8f4d947` | mutation_order_histogram | landscape=TrpB3C, source=SSMuLA | additive_r2=0.1663, completeness_vs_20=0.9992, frac_above_wild_type=0.0135, frac_dead=0.0001, n_genotypes=7994, n_sites=3, nonadditive_variance_fraction=0.8337 | d1884c4 ⚠dirty |
| `f4d7943a` | mutation_order_histogram | landscape=TrpB3B, source=SSMuLA | additive_r2=0.1002, completeness_vs_20=0.9995, frac_above_wild_type=0.1096, frac_dead=0.0001, n_genotypes=7996, n_sites=3, nonadditive_variance_fraction=0.8998 | d1884c4 ⚠dirty |
| `223b3841` | mutation_order_histogram | landscape=TrpB3A, source=SSMuLA | additive_r2=0.2202, completeness_vs_20=0.9964, frac_above_wild_type=0.7769, frac_dead=0.0001, n_genotypes=7971, n_sites=3, nonadditive_variance_fraction=0.7798 | d1884c4 ⚠dirty |
| `42c14d1f` | mutation_order_histogram | landscape=TEV, source=SSMuLA | additive_r2=0.048, completeness_vs_20=0.9946, frac_above_wild_type=0.5053, frac_dead=0.0026, n_genotypes=1.591e+05, n_sites=4, nonadditive_variance_fraction=0.952 | d1884c4 ⚠dirty |
| `615fba2e` | mutation_order_histogram | landscape=T7, source=SSMuLA | additive_r2=0.0406, completeness_vs_20=0.8406, frac_above_wild_type=0.3405, frac_dead=0.17, n_genotypes=6725, n_sites=3, nonadditive_variance_fraction=0.9594 | d1884c4 ⚠dirty |
| `ab974c27` | mutation_order_histogram | landscape=ParD3, source=SSMuLA | additive_r2=0.8517, completeness_vs_20=0.9852, frac_above_wild_type=0.5211, frac_dead=0.0001, n_genotypes=7882, n_sites=3, nonadditive_variance_fraction=0.1483 | d1884c4 ⚠dirty |
| `12f0f12f` | mutation_order_histogram | landscape=ParD2, source=SSMuLA | additive_r2=0.8419, completeness_vs_20=0.9852, frac_above_wild_type=0.0488, frac_dead=0.0003, n_genotypes=7882, n_sites=3, nonadditive_variance_fraction=0.1581 | d1884c4 ⚠dirty |
| `6a788b6e` | mutation_order_histogram | landscape=GB1, source=SSMuLA | additive_r2=0.1944, completeness_vs_20=0.9335, frac_above_wild_type=0.4518, frac_dead=0.9195, n_genotypes=1.494e+05, n_sites=4, nonadditive_variance_fraction=0.8056 | d1884c4 ⚠dirty |
| `8757bcf0` | mutation_order_histogram | landscape=DHFR, source=SSMuLA | additive_r2=0.2771, completeness_vs_20=0, frac_above_wild_type=0.5699, frac_dead=0, n_genotypes=2.613e+05, n_sites=9, nonadditive_variance_fraction=0.7229 | d1884c4 ⚠dirty |
| `d1dfd021` | mutation_order_histogram | source=ProteinGym, version=v1.3 | assays_ge_10k_multimutants=14, assays_with_multimutants=69, frac_multiple=0.7176, multiple_mutants=1.769e+06, n_assays=217, single_mutants=6.963e+05, total_mutants=2.466e+06 | d1884c4 ⚠dirty |

### 0.3 — Leakage firewall **[GATE]** · `complete`
- Verdict: **go** (2026-10-27T02:06:58+00:00) 7 leaks found and excluded (3 unanticipated); 23 firewall tests pass; every poison case blocked; 0 residual sequence-level leaks in the 210-assay training split
- ✅ `firewall_tests_passing == 1` → observed `1`

| Run | Experiment | Variant | Metrics | Code |
|---|---|---|---|---|
| `84f2ee03` | firewall_poison_test | scope=ProteinGym+FLIP2, threshold=0.7 | clean_cases_passed=2, firewall_tests_passing=1, leaks_found=7, leaks_unanticipated=3, poison_cases_blocked=6, proteingym_excluded=7, proteingym_train_assays=210, residual_sequence_leaks=0 | 4b2db6f ⚠dirty |

### 0.4 — Encoder harness and embedding cache · `complete`

| Run | Experiment | Variant | Metrics | Code |
|---|---|---|---|---|
| `db237201` | cache_throughput | L=56, backbone=saprot, batch_size=32 | backbone_dim=1280, bytes_per_sequence=1.435e+05, cache_speedup=1449, cold_seq_per_s=2.1, forward_pass_frac_of_time=0.98, gpu_power_limit_w=50, sustained_tflops=0.27, warm_seq_per_s=3036 | d4a37c1 ⚠dirty |
| `04ef48db` | per_residue_rank_test | backbone=saprot, guard=Cache.put + encode | backbone_dim=1280, per_residue_enforced=1, tests_passing=22, tests_total=22 | eb28191 ⚠dirty |
| `0f8e4ca3` | tooling_verified | host=local | tools_required=3, tools_working=3 | 69d9b9d ⚠dirty |

### 0.5 — Evaluation harness and the additive null **[GATE]** · `complete`
- Verdict: **go** (2026-10-27T03:22:09+00:00) eval harness and additive null built before any model; 34 metric tests against analytic cases; measuring baselines on real landscapes showed greedy-with-restarts solves 4/5, which reframed H4 around oracle budget and made TEV primary

| Run | Experiment | Variant | Metrics | Code |
|---|---|---|---|---|
| `92aac288` | metric_validation | scope=synthetic cases with analytic answers | additive_null_r2_on_additive_landscape=1, disguised_additive_model_caught=1, greedy_trapped_by_constructed_valley=1, tests_passing=34, tests_total=34 | 90ba0e8 ⚠dirty |
| `f6872bcd` | additive_null_impl | landscape=ParD3 | additive_r2_full=0.8517, additive_spearman_higher_order=0.8902, greedy_hit_restarts_1=0, greedy_hit_restarts_100=1, greedy_hit_restarts_20=1, greedy_hit_restarts_5=1 | 90ba0e8 ⚠dirty |
| `82e4de28` | additive_null_impl | landscape=ParD2 | additive_r2_full=0.8419, additive_spearman_higher_order=0.8439, greedy_hit_restarts_1=1, greedy_hit_restarts_100=1, greedy_hit_restarts_20=1, greedy_hit_restarts_5=1 | 90ba0e8 ⚠dirty |
| `b8d21899` | additive_null_impl | landscape=GB1 | additive_r2_full=0.1944, additive_spearman_higher_order=0.476, greedy_hit_restarts_1=0, greedy_hit_restarts_100=1, greedy_hit_restarts_20=1, greedy_hit_restarts_5=1 | 90ba0e8 ⚠dirty |
| `d18f3324` | additive_null_impl | landscape=TEV | additive_r2_full=0.048, additive_spearman_higher_order=-0.005, greedy_hit_restarts_1=0, greedy_hit_restarts_100=0.4, greedy_hit_restarts_20=0, greedy_hit_restarts_5=0 | 90ba0e8 ⚠dirty |
| `6a2443a0` | additive_null_impl | landscape=TrpB4 | additive_r2_full=0.2568, additive_spearman_higher_order=0.0696, greedy_hit_restarts_1=0, greedy_hit_restarts_100=1, greedy_hit_restarts_20=1, greedy_hit_restarts_5=1 | 90ba0e8 ⚠dirty |

### 0.6 — Pre-register thresholds for H1-H4 **[GATE]** · `complete`
- Verdict: **go** (2026-10-27T03:32:13+00:00) H1-H4 registered with measurement-grounded thresholds; 4.2 and 6.2 had none before; greedy/baseline/budget ambiguities fixed in advance; 19 drift tests tie the document to plan.yaml

| Run | Experiment | Variant | Metrics | Code |
|---|---|---|---|---|
| `8811276a` | preregistration | scope=H1-H4 | amendments_after_phase1=0, hypotheses_registered=4, integrity_tests=19, pre_phase1_revisions_logged=6, steps_missing_thresholds=0, steps_with_thresholds=9 | 719b349 ⚠dirty |

## Phase 1 — Kill-shot diagnostics

### 1.1 — D3: Epistasis headroom audit **[GATE]** · `not-started`
- ⬜ `nonadditive_variance_fraction >= 0.15 [max]` → observed `—`
- ⬜ `greedy_optimum_hit_rate <= 0.8 [min]` → observed `—`
- Pending: `additive_fit`, `greedy_reachability`, `negative_control`, `pairwise_fit`, `rse_count`, `stack_topk_headroom`

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
- ⬜ `spearman_mean >= 0.55` → observed `—`
- ⬜ `margin_over_additive_null > 0.05` → observed `—`
- ⬜ `frac_assays_beating_null >= 0.7` → observed `—`
- ⬜ `spearman_mean_vs_frozen_regression > 0` → observed `—`
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
- ⬜ `error_growth_slope_ratio <= 0.75` → observed `—`
- ⬜ `error_growth_gap_over_seed_sd >= 2` → observed `—`
- ⬜ `candidate_diversity_ratio >= 1.1` → observed `—`
- ⬜ `n_seeds >= 3 [min]` → observed `—`
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
- ⬜ `policy_gap_over_seed_sd >= 2` → observed `—`
- ⬜ `n_seeds >= 5 [min]` → observed `—`
- ⬜ `spearman_calibration_vs_policy_rank >= 0.5` → observed `—`
- ⬜ `eta_zero_reported == 1` → observed `—`
- Pending: `active_inference`, `gflownet`, `low_data_regime`, `mcts`, `policy_vs_calibration`, `ppo_entropy`

### 6.3 — H4: Multi-step planning **[GATE]** · `not-started`
- ⬜ `optimum_hit_rate_margin_over_best_baseline > 0.1` → observed `—`
- ⬜ `best_fitness_margin_over_best_baseline > 0.05` → observed `—`
- ⬜ `correlation_with_additive <= 0.95 [max]` → observed `—`
- ⬜ `margin_on_negative_controls <= 0.05 [max]` → observed `—`
- ⬜ `n_seeds >= 5 [min]` → observed `—`
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
