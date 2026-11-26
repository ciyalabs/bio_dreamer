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

**Re-audited 2026-11-26 by Step 1.1** (`scripts/epistasis_audit.py`). The table below is
sorted by the column that matters, and it is **not** the column the first audit produced.

`1−R² (old)` is the Step 0.2 figure: one minus the R² of an additive model fitted to all the
data. It conflates genuine epistasis with measurement noise and floor censoring, and across
these sixteen landscapes it is rank-correlated with exploitable epistasis at **ρ = −0.49**
(p = 0.052). `cv1` and `cv2` are out-of-fold R² at interaction order 1 and 2, and
**`structured` = cv2 − cv1** is predictable variance that site-independence cannot reach —
the figure that separates the three causes.

| Landscape | genotypes | sites | complete | dead | 1−R² *(old)* | cv1 | cv2 | **structured** | greedy@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DHFR | 261,333 | 9 (nt) | n/a | 0% | 0.723 | 0.277 | 0.729 | **+0.452** | 0.0 |
| GB1 | 149,361 | 4 | 93.3% | 92% | 0.806 | 0.193 | 0.610 | **+0.421**† | 0.7 |
| TrpB4 | 159,129 | 4 | 99.5% | 0% | 0.743 | 0.256 | 0.600 | **+0.344** | 1.0 |
| TEV | 159,132 | 4 | 99.5% | 0% | 0.952 | 0.047 | 0.077 | **+0.257**† | 0.0 |
| TrpB3I | 7,784 | 3 | 97.3% | 0% | 0.465 | 0.527 | 0.782 | **+0.256** | 1.0 |
| TrpB3D | 7,763 | 3 | 97.0% | 0% | 0.749 | 0.235 | 0.456 | **+0.221** | 0.7 |
| TrpB3E | 7,964 | 3 | 99.6% | 0% | 0.947 | 0.041 | 0.237 | **+0.196** | 0.4 |
| TrpB3G | 7,891 | 3 | 98.6% | 0% | 0.855 | 0.130 | 0.247 | **+0.117** | 0.8 |
| ParD2* | 7,882 | 3 | 98.5% | 0% | 0.158 | 0.839 | 0.949 | **+0.110** | 1.0 |
| ParD3* | 7,882 | 3 | 98.5% | 0% | 0.148 | 0.849 | 0.937 | **+0.088** | 1.0 |
| TrpB3H | 7,716 | 3 | 96.5% | 0% | 0.972 | 0.016 | 0.059 | **+0.043** | 1.0 |
| TrpB3F | 7,737 | 3 | 96.7% | 0% | 0.877 | 0.109 | 0.128 | **+0.019** | 1.0 |
| T7 | 6,725 | 3 | 84.1% | 17% | 0.959 | 0.023 | 0.027 | **+0.009**† | 0.0 |
| TrpB3A | 7,971 | 3 | 99.6% | 0% | 0.780 | 0.205 | 0.211 | **+0.006** | 0.4 |
| TrpB3B | 7,996 | 3 | 100.0% | 0% | 0.900 | 0.086 | 0.092 | **+0.005** | 0.4 |
| TrpB3C | 7,994 | 3 | 99.9% | 0% | 0.834 | 0.150 | 0.152 | **+0.002** | 0.2 |

*\* negative control. † measured on the above-median-read-depth subset, which is how these
landscapes must be used. `greedy@5` = optimum hit rate at 5 restarts, 10 seeds, max 20 steps.*

### What changed, and how these must be used

- **Six landscapes are dropped**: T7, TrpB3A, TrpB3B, TrpB3C, TrpB3F, TrpB3H, all with
  structured epistasis ≤ 0.043. T7 is the clearest case — 0.959 by the old metric, **+0.003**
  by the new one, with a reliability ceiling of 0.73 and a median 11 reads per variant. A
  method cannot be shown to use epistasis on a landscape whose epistasis is not measurable.
- **TEV: use the depth-filtered subset only** (reads ≥ median = 209, n = 79,630). Its
  published per-variant variance caps any model's R² at **0.668**. Structured epistasis is
  +0.030 at full depth and +0.257 filtered. Its floor-removed 1−R² is 0.955 against 0.952
  raw, so none of the inflation is censoring — all of it is read noise. Still the primary H4
  landscape, now for the right reason: greedy failure is total (0.0 at 5 restarts across 10
  seeds, 0.6 at 100; path accessibility 0.042) with structured epistasis present. GB1 and
  TrpB3E combine the two as well — TEV is the extreme, not the only case.
- **GB1 is 92% dead, and its epistasis is still real.** 1−R² falls from 0.806 to **0.634**
  with floored variants removed — so a fifth of the raw figure is censoring — but the
  structured figure barely moves under depth filtering (+0.417 → +0.421). Median **4**
  selected reads per variant, 19.7% at zero, so fitness is coarse even where not floored.
- **DHFR is a negative control, not a candidate.** Its 9 sites are nucleotides. 13.5% of
  variants carry a stop codon; of the rest only 6.3% of variance is synonymous, so the
  epistasis is not codon usage — but averaging the median 16 synonymous codons per protein
  removes ~16× of noise, and the translated landscape (8,000 variants, **100% complete**,
  0.1% dead) is **78% additive with greedy solving it 0.9**. Never quote its nucleotide
  figures as protein figures.
- **TrpB4 remains the cleanest 4-site landscape**: 99.5% complete, 0% dead, +0.344 structured.
  Greedy solves it from the wild type in 7 steps at **one** restart, so it cannot be decisive
  for H4 on hit rate alone.
- **ParD2, ParD3 and protein-level DHFR are the negative controls.** Note they are not
  epistasis-free: +0.110, +0.088 and +0.187 structured. They are *near-additive*, which is
  what a control needs to be.

Full write-up and the dimension 3–9 depth curve: `diagnostics/epistasis_audit.md`.

### ⚠ Wild-type provenance (2026-11-26)

`Landscape.from_csv` inferred the wild type as the **modal residue per site**. On a
combinatorially complete landscape every residue appears at every site by design, so the
mode is close to arbitrary — and on all three landscapes where it could be checked it
returned a **dead** variant (GB1 `WLVS` at 0.0039 instead of `VDGV` at 1.0000; TEV `FLPP`
at −0.0162 instead of `TDHS` at 0.4144; TrpB4 `GSHL` at 0.0099 instead of `VFVS` at
0.4081). Resolution order is now: explicit argument → a substitution-distance column's
zero row → `KNOWN_WILD_TYPES` → modal **with a warning**, and `wild_type_source` records
which was used.

| landscape | wild type | source | status |
|---|---|---|---|
| GB1 | `VDGV` | `HD==0` row of the shipped CSV | verified |
| TEV | `TDHS` | `Mutation Distance==0` row | verified |
| TrpB4 | `VFVS` | Johnston 2024; present at fitness 0.4081 | verified |
| ParD2, ParD3 | — | no distance column, CSV sorted alphabetically | **unresolved** |
| DHFR, T7, TrpB3A–I | — | same | **unresolved** |

**Open item, blocking H4:** ParD2 and ParD3 are the negative controls, and a control with
the wrong starting point cannot do its job. Their ProteinGym counterpart
`F7YBW8_MESOW` covers 4 and 10 variable positions against SSMuLA's 3, so it does not map
directly. Source needed from Ding et al. 2023.

**Diagnostic for the next case:** `frac_above_wild_type`. It is 0.4–2.4% on the verified
landscapes and 34–97% on the unresolved ones. A functional wild type is not beaten by
most of its own neighbourhood.

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
| Status | **Run on five HIS3 segments** — see HypercubeME run below |

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
| **Tsuboyama** | Not downloaded (1.0 GB) | Fetch before Step 2.1 |
| **Rocklin 2026** | Not downloaded (8.0 GB) | Register use first (authors' request), then decide if the limited role justifies 8 GB |
| **Measurement floors** | GB1 92% dead quantified. Floors for Tsuboyama/HIS3 **not yet characterised** | Required before Step 2.1's censoring decision |

## Non-pip tooling still required

`foldseek` (3Di tokens) · `hmmer`/`MMseqs2` (MSA retrieval for Φ(WT)) · SLIP (synthetic
Potts landscapes, Step 5). Recorded in the `pyproject.toml` header.

---

---

## Model weights and tooling (added 2026-10-27)

### Encoder weights — **a licence problem with the proposed default**

| Model | HF repo | Gated | Licence | Verdict |
|---|---|---|---|---|
| **SaProt 650M** | `westlake-repl/SaProt_650M_AF2` | No | **MIT** (HF card and GitHub) | **Use as default** |
| ESM-2 650M | `facebook/esm2_t33_650M_UR50D` | No | **MIT** | Baseline, fine |
| **ProSST-2048** | `AI4Protein/ProSST-2048` | No | **NONE** — no LICENSE file on HF; GitHub reports `NOASSERTION`, `LICENSE` is 404 | **Do not make it the default** |
| ESM-C 600M / 300M | `biohub/esmc-600m-2024-12` | No | `mit, other` (dual) | Usable; read the "other" (Cambrian) terms before commercial use |
| ESM-3 open 1.4B | `biohub/esm3-sm-open-v1` | No | unspecified on card | Read terms before use |

**`proposal.tex` §4.3 names ProSST as the default encoder.** It has no licence at all,
which is incompatible with the open-source framework promised as Contribution 4. SaProt is
MIT, clean, and only modestly behind on ProteinGym (0.457 vs 0.507 average Spearman).

Recommendation: **make SaProt the documented default**, keep ProSST as an optional backend
with the licence gap stated, and re-check whether ProSST adds a licence later.

Note: EvolutionaryScale's HuggingFace org now redirects to **`biohub`**. Neither ESM-C nor
the open ESM-3 is gated any more, so no account or terms acceptance is needed to download
them — this supersedes the "API-only / non-commercial" caution in the proposal, which should
be softened to "dual-licensed; read the terms".

Downloaded to `models/hf` (gitignored): SaProt 650M, ESM-2 650M. ProSST deliberately skipped.

### SLIP — synthetic landscapes (Step 5)

| | |
|---|---|
| Source | [google-research/slip](https://github.com/google-research/slip) |
| **Licence** | **Apache-2.0** |
| Status | Cloned, 135 KB |
| Note | Ships `epistasis_selection.py`, directly relevant to Step 1.1 and Step 5 |

### Non-pip tools — **not installed, and blocked**

`foldseek`, `hmmer` (`jackhmmer`), `mmseqs` are all absent. I attempted the foldseek static
binary and the sandbox refused: downloading and executing an external binary is outside what
I may do, regardless of permission granted in chat.

**These need a human.** The simplest route on Ubuntu 26.04:

```bash
sudo apt install hmmer mmseqs2          # MSA retrieval for Φ(WT)
# foldseek: conda install -c conda-forge -c bioconda foldseek
#       or: download the static build from https://github.com/steineggerlab/foldseek/releases
```

Needed for Step 0.4 (structure tokens, MSA profile), not before.

---

## Decision: Rocklin 2026 is **not needed** — do not register

The question was whether to register and download the 8 GB record. Having read its README
in full, the answer is no, and the reasoning is worth recording so it is not revisited:

- The record's **only new measurements** are MGnify wild-type domain stability (ΔG of ~1.8M
  distinct domains, each measured once) and MGnify insertion/deletion benchmarks. The plan
  excludes indels.
- The files named "Megascale point mutants" are **predictions**, i.e. model outputs
  evaluated on the existing Tsuboyama 2023 measurements. No new mutant data.
- The reward head needs ΔΔG **of mutations**. Tsuboyama 2023 (already downloaded) is that
  source. Rocklin 2026 adds breadth across unrelated domains, which the architecture has no
  current use for.

**No registration is required of you.** The only item with any future pull is
`S1724_thermomutdb_cleaned_withseq.csv` — a curated ThermoMutDB set that would make a decent
*independent* stability benchmark for Step 2.2, since it is not Tsuboyama-derived. But it
sits inside an 871 MB archive and **ThermoMutDB is freely available on its own**, so fetch it
directly if Step 2.2 wants it.

Revisit only if a wild-type stability prior becomes useful for the static context Φ(WT).

---

## HypercubeME run (2026-10-27)

**Validated first:** reproduced the shipped `test_expected` output exactly (the apparent diff
was CRLF line endings only).

**Method note.** HIS3 was mutagenised in ~12 **separate segments**; variants span 25–30
positions within one window. A combined run over all 496,138 genotypes was started, reached
7 GB RSS, and was abandoned after ~10 minutes still on dimension 1 — genotypes from different
segments cannot form a hypercube together, so nearly all of that work was wasted. The
published method is per-segment, and that is what is reported here.

| HIS3 segment | genotypes | hypercubes | × | max dim |
|---|---:|---:|---:|---:|
| 140–160 | 915 | 25,680 | 28.1 | **6** |
| 160–180 | 799 | 23,568 | 29.5 | **6** |
| 60–80 | 264 | 4,703 | 17.8 | **6** |
| 80–100 | 242 | 4,646 | 19.2 | 4 |
| 6–40 | 58,008 | 13,464,442 | 232.1 | **9** |
| **Total (5 segments)** | **60,229** | **13,523,039** | **224.5** | **9** |

Dimension breakdown for segment 140–160: d1 4,681 · d2 9,191 · d3 8,190 · d4 3,154 ·
d5 446 · **d6 18**.

Dimension breakdown for segment 6–40: d1 497,062 · d2 1,792,772 · d3 3,478,667 ·
d4 3,898,512 · d5 2,576,913 · d6 988,823 · d7 209,135 · d8 21,741 · **d9 817**.
Dimension 10 returned zero.

**The multiplication factor is strongly superlinear in segment size.** The four small
segments average 26×; segment 6–40, 26× larger in genotypes, multiplies 232× and reaches
three dimensions deeper. This is expected — a hypercube needs every corner present, so
density matters more than count — but it was not obvious in advance, and it means the four
small segments *understated* what the method yields.

**What this means for the plan.** Path-level ground truth now exists to **horizon 9**, not
4 or 6. Three plan steps were widened accordingly: the Phase 2 stratified sample (3–6 → 3–9),
the Phase 4 error-growth curve (3–8 → 3–9), and the Phase 7 depth check (5–8 → 5–9). The
"path-level data remains narrow" caution in the proposal's limitations was too strong and has
been rewritten; what remains true is that this is **one protein**, segment-local, with heavily
overlapping hypercubes and a growth-assay measurement floor.

**Run cost:** 6 h 22 m (22,896 s) wall clock on 8 cores of the laptop, peak ~0.6 GiB RSS
across 12 workers. Within the 2 h–10 d band the paper reports for 16k–82k genotypes.

**Storage.** The segment 6–40 output is **856 MB**, too large for git. Dimensions 8 and 9 are
committed (1.9 MB — they are the scarce ones, and the only ≥8-step ground truth the project
has); dimensions 1–7 are gitignored and regenerable in 6 h 22 m with:

```
python data/raw/hypercubeme/repo/HypercubeME.py \
  -g data/processed/hypercubeme/his3_seg_6_40.txt \
  -d data/processed/hypercubeme/out_6_40 -c 8
```

Note the output directory must **not** exist beforehand; HypercubeME creates it and refuses
to run if it is already there.

Outputs: `data/processed/hypercube_index.json`; per-segment files under
`data/processed/hypercubeme/`. Recorded in the tracker as `0.2 / hypercube_index`.

---

## Tsuboyama 2023 — acquired and characterised

Downloaded `Processed_K50_dG_datasets.zip`, 1,013,678,473 bytes, valid, 18 entries.

The `*_DMS_list.csv` files are lists of **experiments**, not variants:

| File | Rows | Meaning |
|---|---:|---|
| `Single_DMS_list.csv` | 983 | domains with a single-mutant scan |
| `Double_DMS_list.csv` | 725 | **domains with a double-mutant scan** |
| `Triple_DMS_list.csv` | 36 | domains with a triple-mutant scan |

725 domains carry systematic double-mutant ΔG data — substantial 2-step transition data with
a calibrated biophysical readout. `Double_DMS_list.csv` even carries a `max_ep_dG` column
(maximum epistasis in ΔG), which is directly usable in Step 1.1.

---

## Embedding storage budget (Step 0.4)

Measured with `scripts/bench_encoder.py` against SaProt 650M (`d_h = 1280`).
A cached per-residue embedding costs **140.1 KiB per sequence at L=56**, and cost
scales with both length and dimension.

| Workload | n | L | d | Size |
|---|---:|---:|---:|---:|
| GB1 doubles, raw backbone dim | 536,000 | 56 | 1280 | **71.6 GiB** |
| GB1 doubles, after adapter | 536,000 | 56 | 128 | 7.2 GiB |
| TrpB4, raw backbone dim | 159,129 | 390 | 1280 | **148.0 GiB** |
| TrpB4, after adapter | 159,129 | 390 | 128 | 14.8 GiB |
| ProteinGym all, raw (L≈300) | 2,465,767 | 300 | 1280 | **1.7 TiB** |

**Rule that follows: cache `Z_t` after the adapter, never the raw backbone output.**
At `d_h = 1280` ProteinGym alone needs 1.7 TiB, against 521 GiB free on this machine.
At `d = 128` the same corpus is ~174 GiB, which fits. `Cache.budget_report()` prints this
for a planned workload and flags an over-budget fill before it starts.

### Throughput, and what it means for scheduling

Measured on 512 GB1 single mutants, batch 32:

| | |
|---|---|
| Cold pass | 244 s → **2.1 seq/s** |
| Warm pass | 0.17 s → 3,036 seq/s |
| **Cache speedup** | **1,449×** |

**The bottleneck is GPU compute, not the code.** Profiling puts 98% of the time in the
forward pass; GPU→CPU transfer is 2%. The hardware is a **laptop GTX 1650 with a 50 W
limit** (chassis type 10, PCIe gen 1), which sustains **0.27 TFLOP/s** on fp16 matmul at
1860 MHz and 100% utilisation with no throttle flags. SaProt 650M at batch 32 needs about
2.4 TFLOP per batch, which predicts ~3 seq/s — matching what is observed.

Practical consequence, at 2–3.4 seq/s:

| Encoding job | Sequences | Time |
|---|---:|---|
| D1/D2/D4 diagnostics (Phase 1) | ~1,000–5,000 | **5–25 min — fine** |
| GB1 all-doubles | 536,000 | ~50 hours |
| ProteinGym full corpus | 2,465,767 | **~9.5 days** |

So **the near-term plan is unaffected**: Step 1.1 needs no encoder at all, and the Phase 1
diagnostics need only a few thousand sequences. The constraint bites at Phase 2, where the
reward head wants the full corpus. Options at that point, in order of preference: encode once
and rely on the cache (the 1,449× speedup is why the cache exists); use a smaller backbone for
development iterations; or rent a GPU for the one-off full-corpus pass.

The cache pays for itself on the first repeat, which matters because Phase 1's diagnostics
make several passes over the same sequences.

## Licence summary

| Licence | Sources | Redistribution |
|---|---|---|
| MIT | ProteinGym, SaProt, ESM-2 | Yes |
| CC-BY-4.0 | SSMuLA, FLIP2, Tsuboyama, Wu 2016 | Yes, with attribution |
| Apache-2.0 | SLIP | Yes |
| GPL-3.0 | HypercubeME (code) | Code only; copyleft if modified and distributed |
| Dual (mit, other) | ESM-C | Read the "other" terms before commercial use |
| **None stated** | **ProSST** | **Not safe to depend on; use SaProt** |
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
5. **Rocklin 2026: decided against.** No registration needed; see the decision section.
6. **`proposal.tex` §4.3** — change the default encoder from ProSST (no licence) to SaProt
   (MIT), and soften the ESM-3 "API-only / non-commercial" caution, since the open
   checkpoint is now ungated under `biohub`.
7. **Install `foldseek`, `hmmer`, `mmseqs2`** before Step 0.4 — needs a human with sudo.
