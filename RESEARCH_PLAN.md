# ProteinDreamer — Research Plan to First Paper

Internal working document. Ciya Labs / BioDreamer research line.
Spec of record: `proposal.tex` (revised October 2026). This plan executes that spec.

**Created:** 2026-10-26

---

## How to read this

Steps are ordered by **de-risking**, not by architecture. The cheapest experiments that could
kill the project come first. Every step has the same five fields:

| Field | Meaning |
|---|---|
| **Why** | What question this answers, and what it protects against |
| **Work** | Engineering to be done |
| **Experiments** | What is actually run and measured |
| **Go / no-go** | The condition for proceeding, and what to do if it fails |
| **Outputs** | Artifacts that must exist when the step is closed |

**Size** tags are rough relative effort: `S` ≈ days, `M` ≈ 1–3 weeks, `L` ≈ 1–2 months.
They are for sequencing decisions, not commitments.

Two rules that govern the whole plan:

1. **The additive null is reported on every epistasis or planning claim.** Not as a courtesy —
   as the thing that decides whether the claim survives. See §0.5 and §1.1.
2. **No biological claim rests on a simulated label.** Synthetic landscapes answer questions
   about the algorithm; experimental landscapes answer questions about proteins.

---

## Tracking this plan

`plan.yaml` is the machine-readable version of everything below; `bdtrack` reads it
and the result ledger together. Step ids are identical in both files.

```bash
pip install -e .                 # installs the bdtrack entry point
bdtrack init                     # create results/tracking.db, validate plan.yaml
bdtrack status                   # plan tree: gates, thresholds, flags, next steps
bdtrack show 1.1                 # every run and variant for one step
bdtrack compare 1.1 -m nonadditive_variance_fraction
bdtrack verdict 1.1 go -m "..."  # record a gate decision (refuses 'go' over a failing threshold)
bdtrack export                   # results/summary.json + results/RESULTS.md  <- committed
bdtrack report --open            # results/tracker.html dashboard             <- ignored
```

Record a run from experiment code. Provenance (git SHA, dirty flag, config hash,
dataset hash, seed, plan version) is captured automatically, and a run that crashes
is still written with its traceback rather than vanishing:

```python
from biodreamer import tracking

with tracking.run("1.1", "additive_fit",
                  variant={"landscape": "GB1-4site"},
                  config=cfg, seed=0,
                  datasets=["data/gb1_4site.csv"]) as r:
    r.log_metric("nonadditive_variance_fraction", 0.31)
    r.log_artifact("figures/gb1_epistasis.png", kind="figure")
    r.attach_wandb()     # links the W&B training curves, if a run is active
```

W&B still owns per-step training curves. The ledger owns the plan, the variants,
the provenance, the summary metrics, the threshold outcomes and the decisions —
so "what did step 3.4 conclude, and on which variant" is one command, not a search.

A step's `thresholds` in `plan.yaml` are the pre-registration from Step 0.6 in
executable form. Set `aggregate: all` on any step that sweeps landscapes or seeds,
or the check silently reflects whichever variant ran last.

---

# Phase 0 — Groundwork

Nothing is learned in this phase. Its purpose is to make the later phases interpretable.
Skipping any of it produces numbers that cannot be trusted later.

---

## Step 0.1 — Code triage `S`

**Why.** `biodreamer/` is deleted in the working tree but ~8,000 LOC survive in `HEAD`.
That code was written against the **pre-revision** architecture: mean-pooled latents, a GVP
structure branch, DDPM dynamics, an evidential uncertainty head. Resurrecting it wholesale
would silently reintroduce exactly the three things the revised proposal removed. Triage it
deliberately instead of letting `git checkout` make the decision.

**Work.**
- [ ] Recover `HEAD` version to a `legacy/` reference directory (read-only, not imported).
- [ ] Classify every module:

| Module (LOC at HEAD) | Verdict | Note |
|---|---|---|
| `data/download.py` (330), `data/preprocessing.py` (489), `data/loaders.py` (197), `data/dataset.py` (445) | **Reuse** | Highest-value salvage; audit for the new datasets |
| `model_loader.py` (258), `utils.py` (128) | **Reuse, audit** | Backbone loading needs ProSST/SaProt/ESM-C |
| `blocks.py` (1210), `tokenizers.py` (757) | **Partial reuse** | Keep generic blocks; drop GVP blocks |
| `environment.py` (660) | **Partial reuse** | MDP mechanics survive; state definition changed |
| `encoder.py` (271) | **Rewrite** | Mean-pool + GVP → per-residue, single PLM. The largest single piece of work |
| `dynamics.py` (250) | **Extend** | *Correction:* flow matching already exists in `blocks.py::FlowMatching` (real, Euler/Heun). Needs the inverse-dynamics head; DDPM demotes to an ablation |
| `reward.py` (116) | **Rewrite** | 3 heads → 2 targets + ranking loss |
| `world_model.py` (71), `policy.py` (99) | **Rewrite** | Thin; policy becomes pluggable |
| `uncertainty.py` (279) | **Delete evidential path** | Keep ensemble machinery only |
| `decoder.py` (198) | **Delete** | No decoder in a JEPA world model |

- [x] Write `ARCHITECTURE_DELTA.md`: what changed from the deleted code and why, one line each.

**✅ DONE (2026-10-27).** See `ARCHITECTURE_DELTA.md`. The table above was written from
`git show` line counts without reading the code, and was **wrong in three places**:
`blocks.py` already has a real `FlowMatching` implementation, `utils.py` already has a real
`SIGReg`, and `utils.py` already records the EMA-or-SIGReg-but-not-both conclusion that the
proposal revision reached independently months later. Roughly **40% of the legacy code
survives unchanged**. Consequences: Step 4.1 is smaller than scheduled, Step 3.1 is larger
(the inverse-dynamics head and per-residue plumbing are entirely new), and the GVP blocks
must be kept until D1 has used them.

**Experiments.** None.

**Go / no-go.** Proceed once no module is unclassified.

**Outputs.** `legacy/` reference tree; `ARCHITECTURE_DELTA.md`; a clean `biodreamer/protein_dreamer/` skeleton with stubs.

---

## Step 0.2 — Data acquisition and licence register `M`

**Why.** Two separate risks. First, licensing: the Rocklin 2026 stability release and the
ESM3/Cambrian weights have terms that may block open release of a derived model — and that
must be known **before** anything is trained on them, not after. Second, and more important:
**nobody has told you how much multi-mutant data you actually have.** The mutation-order
histogram per assay is the single number this whole project's feasibility rests on, and it has
to be computed, not assumed.

**Work.**
- [x] Acquire and checksum (**the list collapsed once sources were checked**):
  - **ProteinGym v1.3** (MIT) — 217 assays, 2,465,767 variants. **Also supplies** Olson 2014
    GB1, Wu 2016 GB1 and HIS3 as standardised assays, so those three are *not* separate
    acquisitions from paper supplements with unclear licences
  - **SSMuLA** (CC-BY-4.0) — 16 combinatorial landscapes incl. TrpB4, TEV, GB1, ParD2/3
  - **FLIP2** (CC-BY-4.0) — number splits: single→double, 2→many, 3→many
  - **Tsuboyama** (CC-BY-4.0) — 725 domains with double scans, 36 with triples
  - **SLIP** (Apache-2.0) — synthetic Potts landscapes, for Step 5
  - **HypercubeME** (GPL-3.0)
  - ~~Rocklin 2026~~ — **not needed**: mostly wild-type domains, no new mutant data
  - ~~PhoQ~~ — **withdrawn**: 403 behind AAAS login, and only 1,659/160,000 functional
    under a binary selection, so not a quantitative landscape
  - **Tooling**: hmmer 3.4, mmseqs2 18-8cc5c (apt); foldseek 10-941cd33 (static binary,
    `tools/foldseek/bin`) — all three verified functionally
- [x] Write `DATA_REGISTER.md`, one row per source: licence, redistribution terms, variant
      count, **mutation-order histogram**, measurement floor / censoring behaviour, assay type.
- [x] Run HypercubeME on HIS3; record the hypercube count **by dimension**, not just the total.
      Must be run **per segment** — genotypes from different mutagenesis segments cannot form
      a hypercube together, and a combined run wastes nearly all its effort.

**Experiments.** Descriptive only — but the mutation-order histograms and the HIS3 hypercube
dimension breakdown are inputs to Step 1.1 and belong in the paper.

**Go / no-go.** If the Rocklin release is licence-incompatible or predominantly singles,
drop it and proceed on Tsuboyama — note this in the register rather than quietly leaving it
in the proposal.

**Outputs.** `data/` with checksums; `DATA_REGISTER.md`; HIS3 hypercube index by dimension.

**✅ DONE (2026-10-27).** See `DATA_REGISTER.md`. Three findings changed later steps:
ProteinGym is **72% multi-mutant** (not predominantly single, as first written); it already
contains Olson 2014 GB1, Wu 2016 GB1 and HIS3 as MIT-licensed standardised assays, removing
three separate acquisitions; and Rocklin 2026 is mostly **wild-type** domains, so it was
**not** downloaded. HypercubeME reproduced: 2,221 HIS3 genotypes → 58,597 complete
sub-landscapes reaching **dimension 6**. Acquired ProteinGym, SSMuLA (16 landscapes),
FLIP2, Tsuboyama, SLIP. **ProSST has no licence** → SaProt (MIT) is now the default encoder.
PhoQ withdrawn (only 1,659 of 160,000 functional, binary selection).

---

## Step 0.3 — Leakage firewall `S` ⚠️

**Why.** **This is the step most likely to be forgotten and most destructive if it is.**
GB1 appears in ProteinGym *and* is the valley-crossing benchmark. TrpB and PhoQ may appear in
curated epistasis collections. If the reward head trains on ProteinGym's GB1 assays and
planning is then "tested" on the GB1 four-site landscape, every H4 number is meaningless —
and nothing in the pipeline will warn you. There is published precedent for this being a real
hazard: the Cerebra atlas paper had to perform overlap removal, leaving only 19 ProteinGym
assays.

A second, subtler trap: the **FLIP GB1 protocol requires training on GB1** singles and doubles,
which directly conflicts with holding GB1 out. The resolution is two model instances that are
never cross-reported:

| Instance | Trained on | Used for |
|---|---|---|
| **`benchmark`** | FLIP protocol, GB1 included | Literature comparability only (Step 2.2) |
| **`planning`** | GB1, TrpB, PhoQ fully held out | Everything in Phase 6, including H4 |

**Work.**
- [ ] Freeze the held-out set: **GB1, TrpB4, TEV, ParD2, ParD3** — excluded from reward-head
      *and* dynamics training. (PhoQ dropped; see 0.2.)
- [ ] **Hold out the ProteinGym assay ids too**, since 0.2 established that the landscapes live
      *inside* the training corpus rather than in separate files:
      `SPG1_STRSG_Olson_2014`, `SPG1_STRSG_Wu_2016`, `HIS7_YEAST_Pokusaeva_2019`.
      This makes the firewall more necessary, not less — the leak is now a silent one.
- [ ] Write `splits/` manifests with content hashes.
- [ ] Add a **CI test that fails** if a held-out protein's sequence (or a close homolog, by
      sequence identity threshold) appears in any training split.
- [ ] Add a CI test that fails if a `benchmark`-instance checkpoint is loaded by a Phase 6 script.

**Experiments.** The CI tests are the experiment. They must be seen to fail on a deliberately
poisoned split before being trusted.

**Go / no-go.** Hard gate. No training before the firewall tests pass and have been seen to fail correctly.

**Outputs.** `splits/*.json` + hashes; two CI tests; a one-paragraph statement in the paper's methods.

**✅ DONE (2026-10-27) — verdict: go.** The leakage surface was larger than this step assumed.
mmseqs found **7 leaks, 3 of them unanticipated**:

| Held-out | Leaks into | Identity | Anticipated? |
|---|---|---|---|
| GB1 | `SPG1_STRSG_Olson_2014`, `SPG1_STRSG_Wu_2016` | 100% | yes |
| GB1 | `SPG2_STRSG_Tsuboyama_2023_5UBS` | 89.2% | **no — homolog** |
| GB1, TrpB4 | FLIP2 `gb1`, `trpb` | 100%, 99.2% | **no — a source this plan added** |
| ParD3 | `F7YBW8_MESOW_Aakre_2015`, `..._Ding_2023` | 100% | **no — UniProt name matches neither landscape** |
| ParD2 | `F7YBW8_MESOW_*` | 71.2% | **no — homolog** |
| DHFR | `DYR_ECOLI_Nguyen_2023`, `..._Thompson_2019` | 100% | **no — only matched after translation** |

Three lessons worth carrying forward:

1. **Name matching would have missed four of seven.** `F7YBW8_MESOW` resembles neither
   "ParD2" nor "ParD3". Conversely `TRPC_SACS2` *looks* like TrpB and is unrelated.
2. **FLIP2, which this plan added as a training source, contains two held-out landscapes.**
   Adding a dataset is adding leakage surface.
3. **A protein-level search silently misses a nucleotide-coded landscape.** SSMuLA's DHFR is
   DNA; it only matched `DYR_ECOLI` once translated. The firewall translates before checking.

Result: 217 ProteinGym assays → **210 trainable, 7 excluded**; 0 residual sequence-level
leaks. 23 tests in `tests/unit/test_firewall.py`, every poison case blocked. Registry in
`splits/heldout.yaml`, hashes in `splits/manifest.json`.

---

## Step 0.4 — Encoder harness and embedding cache `M`

**Why.** Every later phase reads from this. Without a cache, repeated backbone forward passes
dominate runtime and make iteration painfully slow.

**Work.**
- [ ] One interface over ProSST (default), SaProt, ESM-C, ESM-2, returning **per-residue**
      `(L, d_h)` — never pooled at this layer.
- [ ] `Φ(WT)` pipeline: MSA retrieval + per-position conservation, reference structure,
      solvent exposure, per-site PLM marginals. Computed **once per protein**.
- [ ] Mutation-history encoding `E(m_1:t)`.
- [ ] Embedding cache keyed by `(backbone, sequence_hash)`; eviction policy.
- [ ] **Budget storage before committing.** Worked example: GB1 doubles at per-residue
      `d=128`, fp16 → `536k × 56 × 128 × 2 B ≈ 7.7 GB`. Fine. Recompute this for every dataset
      *before* launching a cache fill — per-residue latents scale with `L`, and this is where a
      naive run fills the disk overnight.

**Experiments.** Throughput and cache-hit benchmarks; verify per-residue outputs are not
accidentally pooled anywhere (unit test on tensor rank).

**Go / no-go.** Proceed when a full pass over the smallest dataset runs from cache.

**✅ DONE (2026-10-27).** SaProt fits comfortably — 1,243 MiB of weights, 1,363 MiB peak even
for T7 at 883 aa, against 3,707 MiB. Per-residue output enforced in both `encode` and
`Cache.put`. Cache gives a **1,449× speedup** on a warm pass.

**⚠ Hardware constraint discovered, and it changes Phase 2 scheduling.** The GPU is a
**laptop GTX 1650 with a 50 W limit** sustaining **0.27 TFLOP/s**; profiling puts 98% of
encode time in the forward pass, so this is compute, not code. At **2–3.4 seq/s**:

| Job | Sequences | Time |
|---|---:|---|
| Phase 1 diagnostics (D1/D2/D4) | ~1,000–5,000 | **5–25 min — fine** |
| GB1 all-doubles | 536,000 | ~50 h |
| ProteinGym full corpus | 2,465,767 | **~9.5 days** |

Phase 0 and Phase 1 are unaffected — **Step 1.1 needs no encoder at all**. The constraint
bites at **Step 2.1**, where the reward head wants the full corpus. Decide then between
encoding once into the cache, using a smaller backbone for iteration, or renting a GPU for
the one-off pass. Do not discover this at Step 2.1.

**Outputs.** `encoder/` module; cache; storage budget table in `DATA_REGISTER.md`.

---

## Step 0.5 — Evaluation harness **and the additive null**, before any model `M` ⚠️

**Why.** This ordering is deliberate and is the most important process decision in the plan.
If the model is built first, it will be tuned against baselines that do not exist yet, and the
additive null will be implemented late, under pressure, by someone who wants it to lose.
Visani, Verma & DeWitt showed a *Science* paper's epistasis claims dissolving against exactly
this baseline. Build the judge before the defendant.

**Work.**
- [ ] Metrics module: per-assay Spearman; top-*k* recovery under fixed oracle budget;
      global-optimum hit rate; reciprocal-sign-epistasis trap escape rate; path-accessibility
      statistics; calibration curves; risk–coverage.
- [ ] **Additive null**: linear model over single-mutant effects, fit per landscape.
- [ ] **"Stack top-*k* beneficial singles"** heuristic.
- [ ] Greedy single-step optimiser (will later run on the project's own reward head).
- [ ] Experiment tracking: config hash, git SHA, seed, dataset manifest hash on every run.
- [ ] A `report_additive_comparison()` helper that every planning experiment must call, emitting
      both the null's score **and** the correlation between model and null predictions.

**Experiments.** Validate metrics on synthetic cases with known answers (e.g. a landscape that
is additive by construction must give the null ~perfect score).

**Go / no-go.** Hard gate before Phase 2.

**Outputs.** `eval/` module; additive null implementation; tracking configured.

---

## Step 0.6 — Pre-register thresholds for H1–H4 `S`

**Why.** You asked not to forget anything. The thing researchers forget is **what they said in
advance would count as failure.** Writing it down now, timestamped and committed, is what makes
a later negative result publishable rather than embarrassing.

**Work.** Write `PREREGISTRATION.md`, committed and dated, fixing:
- [ ] **H1** — the Spearman range that counts as "competitive" against Kermut / Cerebra /
      ProSST zero-shot, and the margin over the additive null that counts as real.
- [ ] **H2** — the difference in multi-step error growth (A vs B) that counts as A winning,
      and the cost ratio at which B's simplicity wins anyway.
- [ ] **H3** — what counts as a policy-class difference versus seed noise.
- [ ] **H4** — the global-optimum hit-rate margin over the additive null and over greedy that
      counts as support; **and the model–null prediction correlation above which H4 is declared
      falsified regardless of absolute performance.**
- [ ] For each: the action taken if the threshold is missed.

**Go / no-go.** Hard gate. Thresholds are not edited after Phase 1 begins; if they must be,
the edit is logged with a reason and the original kept visible.

**Outputs.** `PREREGISTRATION.md`.

---

# Phase 1 — Kill-shot diagnostics

Days of work, no training, each one capable of redirecting the project. Run **1.1 first** — it
is the one that can invalidate the premise.

---

## Step 1.1 — D3: Epistasis headroom audit `S` ⚠️ **RUN THIS FIRST**

**Why.** This is the experiment the whole project should be gated on, and it was not in the
original plan at all. If the target landscapes are close to additive, then **there is no valley
to cross**, H4 is vacuous, and no architecture can fix it. Finding this out in days instead of
months is worth more than any other single result here. It also produces, as a by-product, the
exact additive-null numbers the paper will need in every later table.

**Work.**
- [ ] Fit additive models to: **TrpB4 and TEV (primary), GB1 (third)**, GB1 all-doubles, the
      nine TrpB 3-site landscapes, and a stratified sample of HIS3 hypercubes (dimensions 3–6).
- [ ] **Negative controls: ParD2 and ParD3** (non-additive fraction 0.148–0.158, near-additive).
      A planner whose advantage is real should show little or none on these.
- [ ] A first pass is already done in 0.2 (`scripts/audit_landscapes.py`,
      `data/processed/landscape_audit.json`); 1.1 extends it with path-level analysis.
- [ ] Fit pairwise (and where data allows, third-order) models.
- [ ] Handle censoring explicitly — see Step 2.1.

**Experiments.**
- [ ] Fraction of fitness variance explained by additive / pairwise / higher-order terms, per landscape.
- [ ] Count of reciprocal-sign-epistasis pairs.
- [ ] **Fraction of global optima reachable by greedy hill-climbing from wild type**, versus
      requiring an indirect path — i.e. how often the phenomenon H4 targets actually occurs.
- [ ] How much of the achievable fitness gain the "stack top-*k* singles" heuristic already captures.

**⚠ Read the dead fraction before the R².** 0.2 measured GB1 at **92% dead** — a large part of
what its additive model fails to explain is a floor-dominated distribution, not exploitable
epistasis. TrpB4 (0% dead) and TEV (0.3% dead) do not have this problem, which is why they
are primary. Never quote a non-additive fraction without the dead fraction beside it.

**Go / no-go.**
- **Go** if a material fraction of variance is non-additive **and** greedy fails to reach the
  optimum often enough for planning to have room.
- **Rescope** if landscapes are near-additive or greedy nearly always succeeds. The honest
  pivots are in Step 8.3 — and this outcome is itself a publishable finding given the
  Visani and Wittmann critiques.

**Outputs.** `diagnostics/epistasis_audit.md` + figures; additive-null reference numbers for
every landscape, frozen and reused everywhere downstream.

---

## Step 1.2 — D1: Structure invariance under point mutation `S`

**Why.** Confirms or refutes the removal of the per-step structure stream — the largest
architectural change in the revision. Produces the figure that preempts the most predictable
objection, and if it comes out the other way, tells you which mutation classes deserve a
conditional structural update.

**Work.**
- [ ] Select ~50 proteins spanning length and fold diversity.
- [ ] ~20 single mutants each, deliberately including known strongly destabilising substitutions.
- [ ] Fold wild type and mutants; extract structural embeddings under both a GVP encoder and
      structure tokens.
- [ ] **Establish the encoder's numerical noise floor** (same sequence, repeated passes) — the
      comparison is meaningless without it.

**Experiments.**
- [ ] TM-score and Cα RMSD, mutant vs wild-type prediction.
- [ ] `‖h_struct_WT − h_struct_mut‖`, against the noise floor.
- [ ] Stratify by substitution type to find any class that *does* move the structure.

**Go / no-go.** If displacement is at noise level, the removal is empirically justified — keep
the figure. If a identifiable class moves, document it and consider a conditional update rather
than reinstating per-step folding wholesale.

**Outputs.** `diagnostics/d1_structure.md`; paper figure.

---

## Step 1.3 — D2: Latent capacity and dimension `S`

**Why.** Decides the pooling question by measurement rather than argument, and fixes `d`.
If neither pooled nor per-residue displacement tracks stability, the frozen-backbone strategy
itself needs revisiting before anything is trained.

**Work.** Compute latent displacements on Tsuboyama under each candidate backbone.

**Experiments.**
- [ ] Correlation of `‖z_WT − z_mut‖` with `|ΔΔG|`, **per-residue versus mean-pooled**.
- [ ] Sweep `d ∈ {64, 128, 256, 512}`.
- [ ] Sweep backbone: ProSST, SaProt, ESM-C, ESM-2 650M.

**Go / no-go.** Per-residue should track `|ΔΔG|` where pooled does not — that is the revision's
central claim about the latent. If both fail, stop and reconsider the encoder before Phase 2.

**Outputs.** `diagnostics/d2_latent.md`; chosen `d` and backbone, with the numbers behind the choice.

---

## Step 1.4 — D4: Inverse-dynamics feasibility probe `S`

**Why.** The inverse-dynamics anchor is the most load-bearing new component in the revised
architecture. Before building it into a training loop, check the precondition: is the applied
mutation even recoverable from frozen embeddings? If not, the latent carries no action signal
and the anchor cannot do its job.

**Work.** Small head trained on frozen `(Z_WT, Z_mut)` pairs to predict `(i, AA_new)`.

**Experiments.**
- [ ] Top-1 / top-5 accuracy for position `i`, and for amino acid `j` given `i`.
- [ ] Per-residue versus pooled latents (pooled should be much worse — a useful confirmation of D2).
- [ ] Accuracy as a function of protein length `L`.

**Go / no-go.** Position recovery must be well above chance (`1/L`). If not, revisit the
adapter and latent dimension before Phase 3.

**Outputs.** `diagnostics/d4_inverse.md`.

---

**Phase 1 gate.** Write a single consolidated diagnostics report and re-read
`PREREGISTRATION.md` against it. Decide explicitly: proceed, adjust, or rescope. Log the decision.

---

# Phase 2 — Reward head

The reward head is the foundation. A planner on top of a weak reward model is a machine for
generating confident nonsense, so this phase must clear its bar before Phase 3 starts.

---

## Step 2.1 — Implement and train `M`

**Why.** Grounds the latent in experimental measurement and supplies the signal every later
phase optimises.

**Work.**
- [ ] Shared trunk on pooled latent (pooling happens **here and only here**).
- [ ] Stability target: SmoothL1 on ΔΔG / ΔG.
- [ ] Fitness target: **per-assay standardisation + pairwise ranking loss** — never regress raw
      fitness across assays, the values are not comparable.
- [ ] **Censoring, decided and documented.** The GB1 doubles have a measurement floor affecting
      at least 30% of variants, and HIS3's growth assay has one too. Choose censored regression
      or principled exclusion, write down which and why. Silently regressing on floored values
      will corrupt every epistasis estimate downstream.

**Experiments.** Training curves; per-assay performance distribution, not just the mean.

**Go / no-go.** Proceed to 2.2 once stable.

**Outputs.** `reward/` module; trained `benchmark` and `planning` instances (Step 0.3).

---

## Step 2.2 — Benchmark against the literature and the null `M`

**Why.** H1. Establishes whether the reward head is competitive — and whether it beats the
additive null, which is the claim that actually matters.

**Work.** Comparison harness; obtain or reimplement baselines.

**Experiments.** On ProteinGym singles and multiples folds, and the FLIP protocol
(**`benchmark` instance only** — see the two-instance rule in Step 0.3):
- [ ] vs **additive null** ← the decisive one
- [ ] vs Kermut, ProteinNPT, MULTI-evolve, Cerebra atlas
- [ ] vs ProSST / VenusREM zero-shot
- [ ] vs a regression head trained directly on the frozen encoder (isolates what the world-model
      training actually adds)
- [ ] Higher-order extrapolation: train on <300 singles+doubles, test on unseen 3+ mutants —
      the protocol the Cerebra and MULTI-evolve numbers use, so results are comparable.

**Go / no-go.** Must beat the additive null and land in range of Kermut / Cerebra. If it does
not, **do not proceed to Phase 3** — fix the reward model. A planner cannot rescue a bad oracle.

**Outputs.** `results/reward_benchmark.md`; H1 verdict against `PREREGISTRATION.md`.

---

## Step 2.3 — Uncertainty calibration `S`

**Why.** The policy consumes uncertainty directly. An uncalibrated epistemic estimate is worse
than none, and the proposal commits to reporting calibration as a primary result rather than an
appendix afterthought.

**Work.** Ensemble or dropout over the reward head; calibration measurement.

**Experiments.**
- [ ] Error-vs-predicted-uncertainty curves on held-out assays.
- [ ] Risk–coverage analysis.
- [ ] **Shared-trunk versus independent-trunk ensembles** — quantify how much the cheap variant
      under-estimates epistemic uncertainty, since that is the quantity the policy uses.

**Go / no-go.** Record the calibration quality; it conditions the interpretation of H3.

**Outputs.** `results/calibration.md`.

---

# Phase 3 — World model B (deterministic)

Simpler of the two architectures. Establishes the pipeline, the collapse diagnostics, and the
multi-step evaluation machinery that Phase 4 will reuse.

---

## Step 3.1 — Implement `M`

**Work.**
- [ ] Per-residue latent `Z_t ∈ R^(L×d)`; trainable adapter over frozen backbone.
- [ ] SIGReg over the batch × residue sample (not over a pooled vector).
- [ ] **Inverse-dynamics head** — factorised cross-entropy over position and amino acid
      (simpler and stronger than batch-contrastive here, because the action space factorises as
      `L × 20`). Keep the action-contrastive variant behind a flag for the ablation.
- [ ] **No EMA target**; stop-gradient on the target instead.
- [ ] Warm-up schedule: reward + inverse-dynamics losses only, then ramp the dynamics loss —
      mitigates the moving-target interaction named in the proposal's limitations.
- [ ] Deterministic predictor; ensemble of `M = 3–5` over the shared frozen encoder.

**Why no EMA.** Follows the LeWorldModel design the architecture is based on, where SIGReg
replaces teacher–student machinery. The risk is real and monitored in 3.2.

---

## Step 3.2 — Collapse monitoring `S` ⚠️

**Why.** ProteinJEPA found that latent prediction **without** a masked-language-modelling anchor
collapses in nearly every experiment on protein data, with gains that failed to replicate across
seeds. Removing the EMA target is the right call per the cited design, but it must be watched,
and the remedy ladder decided in advance rather than improvised.

**Work.** Instrument, don't just hope.

**Experiments, logged every run.**
- [ ] Latent rank / eigenspectrum.
- [ ] SIGReg statistic over training.
- [ ] **Inverse-dynamics recovery accuracy** — the most direct signal that action-dependent
      dynamics are being learned, and more informative than the prediction loss.

**Remedy ladder, in order, if collapse appears.**
1. Add an explicit masked-token loss at the mutated site.
2. Reinstate an EMA target encoder.
3. Switch to the action-contrastive (AC-MTM) anti-collapse mechanism.

**Outputs.** Collapse dashboard; whichever remedy was needed, logged with evidence.

---

## Step 3.3 — One-step accuracy `S`

**Experiments.**
- [ ] Latent prediction error against encoder-computed ground truth.
- [ ] Reward error when passing the *predicted* latent versus the *true* latent through the
      reward head — this is what actually matters for planning.

---

## Step 3.4 — Multi-step rollout accuracy `M`

**Why.** The first real test of whether multi-step planning is viable at all.

**Experiments.**
- [ ] Error growth over 2 steps on GB1 all-doubles (**held out** — use the `planning` instance
      and a non-GB1 training set; the GB1 doubles are evaluation data here, not training data).
- [ ] Error growth over 3–8 steps on HIS3 hypercubes, stratified by dimension.
- [ ] Reward-error growth alongside latent-error growth; they can diverge.

**Go / no-go.** If 2-step error already exceeds the useful range, flow matching will not rescue
it — diagnose (encoder? adapter? action encoding?) before Phase 4.

**Outputs.** `results/rollout_B.md`.

---

# Phase 4 — World model A (flow matching)

---

## Step 4.1 — Implement `M`

**Work.**
- [ ] Conditional flow matching: Gaussian source, linear path, vector-field regression.
- [ ] Shortcut / self-consistency objective for few-step sampling.
- [ ] **Verify `K = 4` sampling matches many-step sampling** before trusting any downstream
      result — the whole feasibility argument for an imagination planner rests on few-step
      sampling, since a rollout costs `N × H × K` denoiser passes.
- [ ] DDPM variant behind a flag, for the ablation.
- [ ] Trajectory-level prediction where trajectory data permits (attacks error compounding
      directly rather than hoping per-step accuracy composes).

---

## Step 4.2 — H2: A versus B `M`

**Experiments.** Three arms — flow matching, DDPM, deterministic:
- [ ] Multi-step error compounding (same protocol as 3.4).
- [ ] Diversity among top-ranked candidates.
- [ ] **Wall-clock and sampling cost per rollout** — part of the hypothesis, not a footnote.
- [ ] Uncertainty quality from flow-sample variance versus ensemble disagreement.

**Go / no-go.** If B matches A at a fraction of the cost, **report that** — it is a legitimate
finding, it simplifies the system, and the pre-registration already allows for it.

**Outputs.** `results/h2_dynamics.md`; H2 verdict.

---

# Phase 5 — Planner validation on synthetic landscapes

## Step 5.1–5.2 — SLIP `M`

**Why this comes before the real H4 test.** Two reasons. You cannot debug a planner and a world
model at the same time — synthetic landscapes give exact ground truth and unlimited queries, so
planner bugs are separable from model error. And the real combinatorially complete landscapes
are precious held-out data: burning them on planner debugging would compromise the one honest
test of H4. Also, this is the only place a horizon beyond ~8 can be examined at all.

**Work.**
- [ ] Stand up SLIP (Potts model fit to real alignments, tunable epistatic horizon).
- [ ] Wire it as an oracle behind the same interface as the learned reward head.

**Experiments.**
- [ ] Does planning beat greedy and the additive null, and **how does the margin scale with the
      ruggedness parameter?** This is the cleanest available statement of when planning helps.
- [ ] Horizon sweep well past 4.
- [ ] Planner correctness checks: does it find known optima in landscapes where they are known?

**Hard rule.** No biological claim from this phase. Note the known limitations in the write-up:
direct-coupling models underestimate strong long-range epistasis, Potts energies correlate only
partially with measured epistasis, and Potts-based simulated evolution tends toward non-neutral
sequences.

**Outputs.** `results/synthetic_planner.md`.

---

# Phase 6 — Policy and planning on real landscapes

This is where the paper's central claim is won or lost. The held-out landscapes are spent here.

---

## Step 6.1 — Trust region and model exploitation `M`

**Why.** Planning inside a learned reward model with no new experimental queries is offline
model-based optimisation, whose characteristic failure is that the optimiser finds precisely the
out-of-distribution points where the surrogate is over-optimistic. DynaPPO had safeguards; much
later work dropped them. These are reinstated.

**Work.**
- [ ] Three-part admissibility predicate: SIGReg-statistic latent plausibility; model
      disagreement threshold; hard mutational budget from the nearest training sequence.
- [ ] Calibrate thresholds on training data.
- [ ] Action masking in the policy; candidate rejection at inference.

**Experiments.**
- [ ] Rejection rates, and where rejected candidates lie relative to the data manifold.
- [ ] **Ablate the trust region** — how much does predicted fitness inflate without it, and how
      much does true fitness fall? This quantifies the exploitation the safeguard prevents and
      is a result worth reporting in its own right.

**Outputs.** `results/trust_region.md`.

---

## Step 6.2 — H3: Policy comparison `M`

**Experiments.** Common world model, four policy classes:
- [ ] PPO + entropy bonus (the reference the others must beat)
- [ ] Active Inference / expected free energy, **with η swept including η = 0**
- [ ] MCTS
- [ ] GFlowNet
- [ ] Low-data regime (<100 labelled variants) specifically
- [ ] Correlate policy ranking with the uncertainty calibration measured in 2.3 — that
      correlation *is* H3 as stated

**Expect a possible negative result for Active Inference.** Champion et al. found an
expected-free-energy agent collapsing onto a single action where the reward-maximising variant
of the same architecture explored successfully. H3 is phrased so this is reportable rather than
awkward.

**Outputs.** `results/h3_policy.md`; H3 verdict.

---

## Step 6.3 — H4: Multi-step planning `L` ⚠️ **the main event**

**Why.** The project's central claim, and the one the additive-null critique is aimed squarely at.

**Work.** Confirm the firewall (Step 0.3) one more time before running. Use the `planning`
instance only.

**Experiments.** On GB1 four-site, TrpB four-site, PhoQ four-site, each fully held out:
- [ ] vs **additive null** ← decisive
- [ ] vs **"stack top-k beneficial singles"**
- [ ] vs **greedy single-step on the project's own reward head** ← isolates planning from the reward model
- [ ] vs random within the same mutational budget, and wild type
- [ ] vs DynaPPO, µProtein / µSearch, StructEvo, ProSpero
- [ ] **Correlation between ProteinDreamer's predictions and the additive null's** — reported
      always; a high correlation falsifies H4 regardless of absolute performance
- [ ] Metrics: global-optimum hit rate; trap escape rate; path-accessibility versus ground truth
- [ ] ≥3–5 seeds with variance reported

**Go / no-go.** Compare against `PREREGISTRATION.md` and accept the answer. If H4 fails, go to
Step 8.1's decision table — a well-executed negative result here is publishable and useful.

**Outputs.** `results/h4_planning.md`; H4 verdict.

---

## Step 6.4 — Longer horizon `M`

**Experiments.** HIS3 hypercubes at dimension 5–8: does the planner's advantage persist past
the four-site regime? Caveat it properly — one protein, segment-local hypercubes, heavily
overlapping, measurement floor.

---

# Phase 7 — Ablations and robustness

## Step 7.1 — Required ablations `M`

**Why.** These are what make the architectural claims credible; each corresponds to a decision
the revision made, and an unablated decision is an assertion.

- [ ] Per-residue versus pooled latent
- [ ] With / without inverse-dynamics head
- [ ] With / without SIGReg; and the action-contrastive alternative
- [ ] Flow matching vs DDPM vs deterministic
- [ ] Few-step vs many-step sampling
- [ ] η in the expected free energy, including 0
- [ ] Static structural context present vs absent
- [ ] Encoder backbone swap
- [ ] EMA target reinstated (does it help after all?)
- [ ] Trust region on / off (from 6.1)

## Step 7.2 — Seeds and variance `S`

**Why.** ProteinJEPA's gains replicated inconsistently across seeds. Single-seed results in
this area are not evidence.

- [ ] ≥3–5 seeds on every headline number; report variance, not just means.
- [ ] Verify end-to-end reproducibility from a committed config + manifest hash.

---

# Phase 8 — Paper

## Step 8.1 — Decide the claim from the outcomes `S`

The paper is written **after** the results decide what it says. Decision table:

| Outcome | Paper |
|---|---|
| H1 ✓, H2 ✓, H4 ✓ | First latent world model for protein engineering; planning beats the additive null on three held-out landscapes. Strongest version. |
| H1 ✓, H2 ✓, H4 ✗ | "Accurate latent dynamics, but planning gains are explained by additivity." A genuine contribution given the Visani and Wittmann critiques, and more useful to the field than a fourth paper claiming epistasis it cannot demonstrate. Lead with the benchmark. |
| H1 ✓, H2 ✗, H4 ✓ | Planning works; generative dynamics unnecessary. Simplifies the architecture — report it. |
| H1 ✗ | Do not write the paper. Return to Phase 2. |
| Step 1.1 rescope | Benchmark + negative-result paper, or pivot (8.3). |

In every branch, the **trajectory benchmark** (Contribution 2) stands on its own: the field
currently has no shared way to test whether planning helps, and supplying one is valuable
independent of who wins on it.

## Step 8.2 — Venue `S`

- [ ] Realistic first target: **MLCB / MLSB workshop** — fast feedback, correct audience, low cost
      of a negative result. Then a full conference submission (NeurIPS / ICML / ICLR) with the
      benchmark as the anchor contribution.
- [ ] **Be honest about the ceiling:** there is no wet-lab validation in this plan, so
      Nature-family venues are out of reach for this paper. Note it rather than discovering it
      at submission.
- [ ] Check the Cerebra / StructEvo / MULTI-evolve publication timeline before choosing — the
      competitive landscape moved substantially in the six months the proposal sat idle, and it
      will move again.

## Step 8.3 — Rescope options `S`

Decided in advance so a bad Phase-1 result produces a pivot rather than a stall:
- [ ] **(a)** Benchmark + negative result: the honest paper if epistasis headroom is small.
- [ ] **(b)** Pivot to the distribution/library formulation (RQ5) — answers Wittmann's cost
      critique directly and is arguably more practically valuable.
- [ ] **(c)** Pivot to MolDreamer or CellDreamer, where the dynamics are genuinely temporal and
      a world model has more to predict than a frozen encoder's response to a substitution.

## Step 8.4 — Reproducibility package `M`

- [ ] Code, configs, seeds, data manifests with hashes.
- [ ] `PREREGISTRATION.md` published alongside — this is what makes the thresholds credible.
- [ ] **Licence clearance before release.** ESM3/Cambrian weights and the Rocklin release may
      restrict redistribution; the open-source framework is Contribution 4, so this is not optional.
- [ ] Model card; `DATA_REGISTER.md`.

---

# Cross-cutting: things that are easy to forget

The checklist to re-read at the start of every phase.

| # | Item | Why it bites |
|---|---|---|
| 1 | **Leakage firewall + the two-instance rule** | GB1 is both training data and the benchmark. Nothing warns you. Every H4 number depends on this. |
| 2 | **Additive null on every epistasis claim** | The documented failure mode in this exact subfield. |
| 3 | **Model–null prediction correlation, not just scores** | A model can beat the null on score while being the null in disguise. |
| 4 | **Measurement floors / censoring** | ≥30% of GB1 doubles affected; HIS3 too. Corrupts epistasis estimates silently. |
| 5 | **Per-assay normalisation; never pool raw fitness** | ProteinGym values are not comparable across assays. |
| 6 | **≥3–5 seeds, variance reported** | ProteinJEPA's gains failed to replicate across seeds. |
| 7 | **Uncertainty calibration as a primary result** | The policy consumes it; uncalibrated is worse than absent. |
| 8 | **Licences before training, not before release** | Rocklin release, ESM3/Cambrian. Blocks Contribution 4 retroactively. |
| 9 | **Storage budget before every cache fill** | Per-residue latents scale with `L`. |
| 10 | **No biological claim from synthetic labels** | Collapses the project's credibility if violated once. |
| 11 | **Never label multi-mutants with FoldX/Rosetta-style tools** | Additive by construction — would teach additivity and make H4 vacuous. |
| 12 | **Pseudo-labels from other predictors are distillation** | Ceiling becomes the teacher's; valleys may be its artifacts. |
| 13 | **Decision log, maintained continuously** | The reason this revision was needed: six months later, the *why* behind each choice was gone. Log every non-obvious decision with its reason and date. |
| 14 | **Keep `proposal.tex` and this plan in sync** | The proposal is the spec; divergence means one of them is lying. |
| 15 | **Negative results are publishable here — if pre-registered** | `PREREGISTRATION.md` is what converts a disappointment into a contribution. |
| 16 | **Re-check the literature quarterly** | Six months of silence cost two of four original contributions. Set a recurring reminder. |

---

# Critical path summary

```
0.1 code triage ─┐
0.2 data ────────┼─► 0.3 FIREWALL ─► 0.4 encoder ─► 0.5 EVAL+NULL ─► 0.6 PREREG
                 ┘                                        │
                                                          ▼
                              1.1 EPISTASIS AUDIT ◄── gate: is there headroom?
                              1.2 D1  1.3 D2  1.4 D4
                                                          │
                                                          ▼
                              2.1 reward ─► 2.2 H1 ◄── gate: beats additive null?
                              2.3 calibration
                                                          │
                                                          ▼
                              3.x World Model B ─► 3.2 collapse watch
                              4.x World Model A ─► 4.2 H2
                                                          │
                                                          ▼
                              5.x SLIP: does the planner work at all?
                                                          │
                                                          ▼
                              6.1 trust region ─► 6.2 H3 ─► 6.3 H4 ◄── the claim
                                                          │
                                                          ▼
                              7.x ablations + seeds ─► 8.x paper
```

**Three hard gates:** the firewall (0.3), the epistasis audit (1.1), and H1 (2.2).
Each can stop or redirect the project, and each is cheap relative to what it protects.
