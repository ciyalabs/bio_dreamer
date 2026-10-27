# Architecture Delta — legacy code vs the revised proposal

Deliverable of **Step 0.1 / `module_triage`** in `RESEARCH_PLAN.md`.

**Compiled:** 2026-10-27, by reading the modules rather than their filenames.

Legacy code lives at `biodreamer/protein_dreamer/legacy/` — 5,998 lines across 19 modules,
restored in commit `4a7f3b3`. This file classifies each one against the revised
`proposal.tex` and records what must change and why.

---

## Correction to the earlier triage

`RESEARCH_PLAN.md` Step 0.1 states that the legacy code "was written against the
pre-revision architecture" and lists `dynamics.py` as a rewrite from DDPM to flow matching.
**That was written from `git show` line counts without reading the code, and it is wrong in
three places:**

- `blocks.py` already contains a **real `FlowMatching` implementation** (rectified flow,
  Euler and Heun solvers, `x_t = (1-t)·x₀ + t·x₁`) — not a stub. Architecture A's dynamics
  does not need writing from scratch.
- `utils.py` already contains a **real `SIGReg` module**.
- `utils.py` already records the conclusion that *"we will use only one of the EMAUpdater or
  the SIGReg loss, not both together"* — the same decision the proposal revision reached
  independently months later, after `why.txt` raised it.

The legacy code is therefore **further along than the plan assumed**. What is genuinely
missing is narrower and more specific than "rewrite the dynamics".

---

## Module classification

### Reuse as-is

| Module | Lines | Why |
|---|---:|---|
| `blocks.py::FlowMatching` | ~55 | Rectified flow matching, Euler/Heun. **This is Architecture A's core** |
| `utils.py::SIGReg` | ~40 | Real implementation; the regulariser the revision kept |
| `uncertainty.py::EnsembleUncertainty` | ~40 | The M=3–5 ensemble the revision specifies for Architecture B |
| `uncertainty.py::MCDropoutUncertainty` | ~45 | Cheap alternative, worth keeping as an ablation |
| `data/download.py` | 330 | Dataset acquisition; already proven in Step 0.2 |
| `data/loaders.py` | 197 | Generic |
| `data/dataset.py` | 445 | Audit for the per-residue change, but the structure holds |
| `blocks.py::ResidualMLP`, `DiffTransformer` | ~100 | Generic building blocks |
| `config.py` | 170 | Config plumbing |

### Extend

| Module | Lines | Change needed |
|---|---:|---|
| `model_loader.py` | 258 | Knows only the ESM family (`esm`, `esm_if`, `esm_lm`, `esmfold`). **Add SaProt** — now the default encoder per Step 0.2's licence finding. ProSST deliberately not added |
| `dynamics.py` | 250 | Has both `DiffusionDynamics` (DDPM) and flow-matching paths, and `EnergyBasedDynamics` for Architecture B. **Add the inverse-dynamics head**; demote DDPM to an ablation flag |
| `data/preprocessing.py` | 489 | Add Foldseek 3Di tokenisation (SaProt's structure channel) and the MSA profile for Φ(WT) |

### Rewrite

| Module | Lines | Why |
|---|---:|---|
| `encoder.py` | 271 | **The central change.** Mean-pools at `seq_emb[:, 1:-1, :].mean(dim=1)`, runs a second GVP stream on a Cα contact graph, and fuses via `[seq_pooled ǀ struct_pooled ǀ pTM] → MLP`. The revision requires per-residue `Z_t ∈ ℝ^(L×d)`, one structure-aware backbone, and no fusion MLP |
| `reward.py` | 116 | Three heads (ΔΔG, K_d, k_cat) → two targets with a ranking loss |
| `world_model.py` | 71 | Thin; rewire once the encoder and dynamics settle |
| `policy.py` | 99 | Becomes the pluggable interface over PPO / AIF / MCTS / GFlowNet |

### Delete

| Module | Lines | Why |
|---|---:|---|
| `decoder.py` | 198 | A JEPA world model has **no decoder**. Dead by construction |
| `uncertainty.py::EvidentialUncertainty` | ~105 | Withdrawn in the revision: evidential epistemic estimates are under sustained criticism and the policy consumes this quantity directly |
| `blocks.py::GVP`, `GVPConv` | ~210 | The GVP stream is removed from the architecture. **Keep the files** — diagnostic D1 (Step 1.2) explicitly compares a GVP encoder against structure tokens, so they are needed once more before being retired |

### Write from scratch

| Component | Where | Note |
|---|---|---|
| Inverse-dynamics head | new | **No trace in the legacy code.** The most load-bearing new component in the revision |
| Trust region / admissibility | new | No trace. Guards against model exploitation (Step 6.1) |
| Per-residue latent plumbing | throughout | `per_residue` appears nowhere in the legacy code |

---

## Summary

| Verdict | Modules | Approx. lines |
|---|---:|---:|
| Reuse | 9 | ~1,400 |
| Extend | 3 | ~1,000 |
| Rewrite | 4 | ~560 |
| Delete | 3 | ~510 |
| New | 3 components | — |

Roughly **40% of the legacy code survives unchanged**, against the plan's implicit assumption
that most of it was obsolete. The two highest-value survivals are `FlowMatching` and `SIGReg`,
both of which the revision would otherwise have had to write.

The single largest piece of work is `encoder.py`, which is also the change the whole revision
turns on: per-residue latents, one backbone, no fusion MLP, no per-step structure.

## Consequences for the plan

1. **Step 4.1 is smaller than scheduled.** `FlowMatching` exists; the work is the shortcut /
   few-step sampling objective, not the flow matching itself.
2. **Step 3.1 is larger than scheduled.** The inverse-dynamics head and per-residue plumbing
   are entirely new, and `encoder.py` is a rewrite rather than an edit.
3. **Do not delete the GVP blocks yet** — D1 needs them to produce the figure that justifies
   removing the structure stream.
4. `model_loader.py` needs SaProt before Step 0.4 can proceed.
