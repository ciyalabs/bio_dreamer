# Data Register — ProteinDreamer

Deliverable of **Step 0.2** in `RESEARCH_PLAN.md`. Records what each source is, what it
is licensed under, what it actually contains, and what it can and cannot train.

**Compiled:** 2026-10-27 · verified against the live sources, not from memory.
Machine-readable companions: `data/processed/landscape_audit.json`,
`data/processed/proteingym_composition.json`,
`data/processed/proteingym_order_histograms.json`.

`data/` is gitignored. Reproduce with `scripts/audit_landscapes.py` and the URLs below.

---

## Three corrections this step produced

These contradict what `proposal.tex` and `RESEARCH_PLAN.md` currently say. Both documents
need updating; see **Actions** at the end.

### 1. ProteinGym is 72% multi-mutant, not "predominantly single mutants"

The proposal describes ProteinGym as *"predominantly single mutants against one wild type"*
and uses that to argue trajectory data is scarce. Measured from the v1.3 reference file:

| | count | share |
|---|---:|---:|
| Single mutants | 696,311 | 28% |
| **Multiple mutants** | **1,769,456** | **72%** |
| Total | 2,465,767 | |

69 of 217 assays contain multi-mutants; 14 contain ≥10,000 of them. The distribution is
skewed — the median assay has none — but the claim as written is wrong.

### 2. ProteinGym already contains three datasets the plan hunts separately

`RESEARCH_PLAN.md` lists Olson 2014, Wu 2016 and HIS3 as separate acquisitions from paper
supplements with unclear licences. All three are already in ProteinGym, standardised and
MIT-licensed:

| Plan entry | ProteinGym assay | Multi-mutants |
|---|---|---:|
| GB1 all-doubles (Olson 2014) | `SPG1_STRSG_Olson_2014` | 535,917 |
| GB1 four-site (Wu 2016) | `SPG1_STRSG_Wu_2016` | 149,284 |
| HIS3 (Pokusaeva 2019) | `HIS7_YEAST_Pokusaeva_2019` | 495,969 |

This removes three licence problems and three bespoke parsers.

### 3. The Rocklin 2026 release is mostly wild-type domains, not mutants

The proposal implies it is a larger ΔΔG training source than Tsuboyama. It is not. The
1.8 million figure counts **distinct metagenomic domains measured once each** (MGnify
stability), not mutants of a protein. Its point-mutant content is the *existing* Megascale
data, used as a benchmark. It gives breadth across domains, not depth of mutants per domain
— much less useful for a reward head predicting mutation effects than it sounded.

---

## Sources acquired

### ProteinGym v1.3 — **primary**

| | |
|---|---|
| Source | `https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/` · [repo](https://github.com/OATML-Markslab/ProteinGym) |
| **Licence** | **MIT** — permissive, redistribution fine |
| Acquired | `DMS_ProteinGym_substitutions.zip` 42 MB (1,015 MiB unzipped, 217 assays) · `sha256:3a83766254ac` |
| | `cv_folds_multiples_substitutions.zip` 33 MB · `sha256:4f1453ee8ccf` |
| | `DMS_substitutions_reference.csv` 204 KB · `sha256:a8f498011532` |
| Composition | 217 assays, 2,465,767 variants, 28% single / 72% multiple |
| Role | Reward head; multi-mutant supervised evaluation; **and** the source for GB1 and HIS3 |

**Mutation-order histograms** (the number Step 0.2 exists to produce), top assays:

| Assay | variants | order 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SPG1_STRSG_Olson_2014` | 536,962 | 1,045 | 535,917 | — | — | — | — |
| `HIS7_YEAST_Pokusaeva_2019` | 496,137 | 168 | 1,475 | 7,627 | 25,927 | 60,263 | 98,869+ |
| `SPG1_STRSG_Wu_2016` | 149,360 | 76 | 2,091 | 26,019 | 121,174 | — | — |
| `GFP_AEQVI_Sarkisyan_2016` | 51,714 | 1,084 | 12,777 | 12,336 | 9,387 | 6,825 | 4,298+ |
| `CAPSD_AAV2S_Sinai_2021` | 42,328 | 532 | 10,833 | 6,906 | 6,646 | 6,298 | 5,396+ |
| `PHOT_CHLRE_Chen_2023` | 167,529 | 2,122 | 176 | 978 | 3,565 | 9,603 | 19,193+ |
| `GRB2_HUMAN_Faure_2021` | 63,366 | 1,034 | 62,332 | — | — | — | — |
| `PABP_YEAST_Melamed_2013` | 37,708 | 1,187 | 36,521 | — | — | — | — |

**HIS3 spans orders 1–6+ with ~500k variants** — a long-horizon trajectory source in its
own right, before any hypercube extraction.

### SSMuLA — combinatorial landscapes, **primary for H4**

| | |
|---|---|
| Source | Zenodo [10.5281/zenodo.13910506](https://doi.org/10.5281/zenodo.13910506) · code [fhalab/SSMuLA](https://github.com/fhalab/SSMuLA) (GPL-3.0) |
| **Licence** | **CC-BY-4.0** (data record) |
| Acquired | `data.zip` 279 MB · `sha256:6acdf302d412` (`results.zip`, 5.9 GB, not needed) |
| Role | 16 combinatorial landscapes in one consistent format |

Audited with `scripts/audit_landscapes.py` (de-duplicated; additive model = one-hot per
site-residue, the null Step 1.1 must beat):

| Landscape | genotypes | sites | complete vs 20ⁿ | additive R² | **non-additive** | dead | >WT |
|---|---:|---:|---:|---:|---:|---:|---:|
| TrpB3H | 7,716 | 3 | 96.5% | 0.028 | **0.972** | 0% | 61.1% |
| T7 | 6,725 | 3 | 84.1% | 0.041 | **0.959** | 17.0% | 34.1% |
| TEV | 159,132 | 4 | 99.5% | 0.048 | **0.952** | 0.3% | 50.5% |
| TrpB3E | 7,964 | 3 | 99.6% | 0.053 | 0.947 | 0% | 42.8% |
| TrpB3B | 7,996 | 3 | 100.0% | 0.100 | 0.900 | 0% | 11.0% |
| TrpB3F | 7,737 | 3 | 96.7% | 0.123 | 0.877 | 0% | 59.6% |
| TrpB3G | 7,891 | 3 | 98.6% | 0.145 | 0.855 | 0% | 72.6% |
| TrpB3C | 7,994 | 3 | 99.9% | 0.166 | 0.834 | 0% | 1.4% |
| **GB1** | 149,361 | 4 | 93.3% | 0.194 | 0.806 | **92.0%** | 45.2% |
| TrpB3A | 7,971 | 3 | 99.6% | 0.220 | 0.780 | 0% | 77.7% |
| TrpB3D | 7,763 | 3 | 97.0% | 0.251 | 0.749 | 0% | 97.3% |
| **TrpB4** | 159,129 | 4 | **99.5%** | 0.257 | 0.743 | **0%** | 56.8% |
| DHFR | 261,333 | 9 (nt) | n/a | 0.277 | 0.723 | 0% | 57.0% |
| TrpB3I | 7,784 | 3 | 97.3% | 0.535 | 0.465 | 0% | 0.8% |
| ParD2 | 7,882 | 3 | 98.5% | 0.842 | **0.158** | 0% | 4.9% |
| ParD3 | 7,882 | 3 | 98.5% | 0.852 | **0.148** | 0% | 52.1% |

Notes that change how these should be used:

- **GB1 is 92% dead.** Its high non-additive fraction is partly the additive model failing
  to fit a floor-dominated distribution, not exploitable epistasis. Do not read 0.806 as
  headroom without inspecting the live fraction first.
- **TrpB4 is the cleanest 4-site landscape**: 99.5% complete, 0% dead, 0.743 non-additive.
  Better suited as the primary H4 landscape than GB1.
- **TEV** is an equally good second: 159k genotypes, 99.5% complete, 0.3% dead.
- **ParD2 and ParD3 are nearly additive (0.15).** These are the natural **negative
  controls** for H4 — a planner should show no advantage on them. The plan has no negative
  control; it should.
- **DHFR is nucleotide-level** (3 codons), not amino-acid. Different semantics; do not mix
  it with the others without translating.

### FLIP2 — splits

| | |
|---|---|
| Source | Zenodo [10.5281/zenodo.18433203](https://doi.org/10.5281/zenodo.18433203) · [flip.protein.properties](https://flip.protein.properties) |
| **Licence** | **CC-BY-4.0** (authors redistribute all data under it) |
| Acquired | 81 files, 109 MB |
| Content | Amylase, IRED, NucB, TrpB, hydrophobic core, rhodopsin, PDZ3 (>200k double-mutant pairs) |
| Role | **Number splits** (single→double, 2→many, 3→many) are exactly the higher-order extrapolation protocol H1/H2 need |

Note: FLIP2 does **not** include GB1 or PhoQ, despite original FLIP having GB1.

### Tsuboyama 2023 — stability

| | |
|---|---|
| Source | Zenodo [10.5281/zenodo.7992926](https://doi.org/10.5281/zenodo.7992926) |
| **Licence** | **CC-BY-4.0** |
| Size | 1.9 GB total; `Processed_K50_dG_datasets.zip` 1.0 GB |
| Content | ~776k ΔG measurements. **Contains `Double_DMS_list.csv` and `Triple_DMS_list.csv`** inside the processed archive |
| Role | Reward head (stability). The doubles/triples lists matter — this is not singles-only |
| Status | **Not yet downloaded** (1.0 GB); needed for Step 2.1 |

### Rocklin 2026 — wild-type domain stability

| | |
|---|---|
| Source | Zenodo [10.5281/zenodo.19411306](https://doi.org/10.5281/zenodo.19411306) |
| **Licence** | **CC-BY-4.0**, **plus a request to register use** via a Google Form |
| Size | 8.0 GB; key file `230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv` 2.2 GB |
| Authors | Cho, Tsuboyama, Litberg, Jung, Obisesan, Wang, Phoumyvong, Thibeault, Ovchinnikov, Rocklin |
| Content | MGnify stability for ~1.8M domains (60–80 aa) measured once each; Megascale point mutants reused as a benchmark; ins/del benchmarks; ThermoMut S1724; TED domains |
| Role | **Limited.** Breadth across domains, not depth of mutants. Useful for a wild-type stability prior, not for ΔΔG of mutations |
| Status | **Not downloaded.** Registration is a courtesy request, not a licence condition — but it is the authors' explicit ask and **should be done by a human before use** |

### HypercubeME

| | |
|---|---|
| Source | [github.com/ivankovlab/HypercubeME](https://github.com/ivankovlab/HypercubeME) |
| **Licence** | **GPL-3.0** |
| Acquired | Cloned, 316 KB |
| Input | Tab-separated genotype file; first column colon-separated mutations (`0C:2T`), wild type `0Z` or empty |
| Status | Cloned, **not yet run** — see Blocked below |

### Wu 2016 GB1 (original supplement)

| | |
|---|---|
| Source | eLife [10.7554/eLife.16965](https://doi.org/10.7554/eLife.16965), `elife-16965-supp1-v4.xlsx` |
| **Licence** | **CC-BY-4.0** (eLife) |
| Acquired | 6.4 MB · `sha256:ac9a16295cd2` |
| Role | **Redundant** — superseded by ProteinGym's `SPG1_STRSG_Wu_2016`. Kept as provenance cross-check |

---

## Blocked or outstanding

| Item | Status | Action needed |
|---|---|---|
| **PhoQ** (Podgornaia & Laub 2015) | `podgornaia-table-s1.xlsx` returns **HTTP 403** behind AAAS login. Copyright © AAAS; supplementary terms unclear | Download manually if you have access — or **drop it**. SSMuLA's TEV and TrpB4 are better: quantitative, cleanly licensed, and not 99% dead |
| **PhoQ suitability** | Of 160,000 variants only **1,659 are functional** (~1%), and the assay is a binary selection, not quantitative fitness | Reconsider its role; it is not a quantitative landscape like GB1/TrpB |
| **HIS3 GitHub repo** | `Lcarey/HIS3InterspeciesEpistasis` has **no LICENSE file** (3.5 GB) | Not needed — use ProteinGym's `HIS7_YEAST_Pokusaeva_2019` (MIT) instead. Paper is CC-BY; raw data at GEO `GSE99990` |
| **HypercubeME run** | Not executed. Running third-party downloaded code was declined by the sandbox | Run manually, or approve: `python -I data/raw/hypercubeme/repo/HypercubeME.py -g <genotypes> -d <out>` |
| **Tsuboyama** | Not downloaded (1.0 GB) | Fetch before Step 2.1 |
| **Rocklin 2026** | Not downloaded (8.0 GB) | Register use first (authors' request), then decide if the limited role justifies 8 GB |
| **Measurement floors** | GB1 92% dead quantified. Floors for Tsuboyama/HIS3 **not yet characterised** | Required before Step 2.1's censoring decision |

## Non-pip tooling still required

`foldseek` (3Di tokens) · `hmmer`/`MMseqs2` (MSA retrieval for Φ(WT)) · SLIP (synthetic
Potts landscapes, Step 5). Recorded in the `pyproject.toml` header.

---

## Licence summary

| Licence | Sources | Redistribution |
|---|---|---|
| MIT | ProteinGym | Yes |
| CC-BY-4.0 | SSMuLA, FLIP2, Tsuboyama, Rocklin 2026, Wu 2016 | Yes, with attribution |
| GPL-3.0 | HypercubeME (code) | Code only; copyleft if modified and distributed |
| Unclear / restricted | PhoQ (AAAS), HIS3 repo (no licence) | **Do not redistribute** |

Nothing acquired so far blocks the open-source framework promised as Contribution 4.
The two unclear items are both avoidable via ProteinGym equivalents.

---

## Actions arising

1. **`proposal.tex` §4.9 and Table 1** — correct the ProteinGym characterisation (72%
   multi-mutant, not predominantly single) and the Rocklin 2026 role (breadth, not depth).
2. **`RESEARCH_PLAN.md` Step 0.2 / Table 1** — collapse Olson, Wu and HIS3 into ProteinGym;
   add SSMuLA and FLIP2 as primary sources; demote or drop PhoQ.
3. **Step 1.1** — use TrpB4 and TEV as primary landscapes, GB1 with the 92%-dead caveat
   stated, and **add ParD2/ParD3 as low-epistasis negative controls**.
4. **Step 0.3 (leakage firewall)** — the held-out set must now name the ProteinGym assay ids
   `SPG1_STRSG_Olson_2014`, `SPG1_STRSG_Wu_2016`, `HIS7_YEAST_Pokusaeva_2019`, since they
   are inside the training corpus rather than separate files. This makes the firewall more
   necessary, not less.
5. **Register use of the Rocklin 2026 data** before downloading it.
