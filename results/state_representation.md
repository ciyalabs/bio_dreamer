# Phase 3 — what the state has to be, measured rather than assumed

**Runs:** 2026-11-28, local GTX 1650, frozen `westlake-repl/SaProt_650M_AF2`.

```
python -I scripts/state_rank_ceiling.py
python -I scripts/mutation_propagation.py --assay PHOT_CHLRE_Chen_2023 -n 128 --batch 8
python -I scripts/mutation_propagation.py --assay CAPSD_AAV2S_Sinai_2021 -n 64 --batch 2
```

Step 3.1 registers a per-residue state `Z ∈ R^(L×d)` and Step 3.2 gates on effective rank.
Two things had been assumed and are now measured. One assumption held, one was refuted —
**the one that was refuted is the one that would have saved money.**

---

## 1. Per-residue raises the rank ceiling ~3×, and that is not nearly enough

Step 3.2's registered floor is `latent_effective_rank_ratio >= 0.5` of `d_model`. The
pooled run failed at 0.022, and the recorded reason was that the frozen encoder's *own*
pooled states have effective rank 6.6–21.8 of 1280.

The obvious objection: pooling is what destroyed the rank, and the per-residue state Step
3.1 actually registers would not have that problem. Measured on the eleven cached assays,
with no new encoding — `positional_cache` already holds each assay's full wild-type `L×d`
plus every cached variant's rows at its own mutated positions:

| assay | L | pooled | wt rows | mut rows | **all rows** | row displacement |
|---|---:|---:|---:|---:|---:|---:|
| `PHOT_CHLRE_Chen_2023` | 118 | 7.6 | 69.1 | 22.5 | **26.1** | 19.1 |
| `CAPSD_AAV2S_Sinai_2021` | 735 | 14.1 | 65.0 | 41.4 | **49.4** | 25.4 |
| `GFP_AEQVI_Sarkisyan_2016` | 238 | 25.8 | 55.0 | 57.9 | **59.4** | 33.0 |
| `HIS7_YEAST_Pokusaeva_2019` | 220 | 12.6 | 52.8 | 57.0 | **58.9** | 33.3 |
| `SPG1_STRSG_Wu_2016` | 448 | 6.0 | 77.5 | 36.9 | **49.2** | 25.9 |
| `D7PM05_CLYGR_Somermeyer_2022` | 235 | 37.3 | 56.0 | 67.6 | **68.6** | 34.8 |
| `F7YBW8_MESOW_Ding_2023` | 93 | 16.6 | 52.3 | 66.1 | **69.5** | 39.4 |
| `GCN4_YEAST_Staller_2018` | 281 | 22.9 | 56.0 | 62.7 | **65.6** | 40.1 |
| `Q6WV12_9MAXI_Somermeyer_2022` | 222 | 27.7 | 50.5 | 58.4 | **59.5** | 30.6 |
| `Q8WTC7_9CNID_Somermeyer_2022` | 238 | 28.3 | 53.7 | 54.3 | **55.9** | 28.5 |
| `F7YBW8_MESOW_Aakre_2015` | 93 | 11.9 | 52.3 | 28.1 | **31.2** | 18.1 |

All against d = 1280. **The objection is partly right and does not rescue the gate.** Going
per-residue raises the ceiling by 1.8×–8.2×, to 26–70 of 1280. The registered floor needs
**640**. Clearing it would take another ~9×.

So `latent_effective_rank_ratio >= 0.5` is unreachable on the state Step 3.1 registers, not
only on the pooled stand-in. It is a property of single-protein variant data under a frozen
pretrained encoder: all the variants of one protein are near-identical sequences, and the
encoder maps them into a few dozen directions no matter how the state is read out. This is
the evidence the gate amendment rests on, and it is now measured on both state formats
rather than argued from one.

The `row displacement` column — a mutated row minus the wild-type row at the *same*
position, which removes "which position is this" from the comparison — is 18–40. That is
the dimensionality of the signal the world model's dynamics head actually has to model.

---

## 2. A windowed cache would have cut storage 3×. It does not work.

The per-residue cache for the seven `planning` assays is **216.7 GB**. The existing
`positional_cache` stores only each variant's mutated rows (radius 0) and keeps **5.2%** of
the pooled displacement on CAPSD, which is why Step 3.1 cannot use it.

The hypothesis worth testing before paying for the full tensor: 5.2% at radius 0 does not
say whether the missing 95% is diffuse or merely *just outside* the window. If a window of
radius r captured most of it, the cache would store `wild type + rows within r of each
mutation` and reconstruct the rest.

`scripts/mutation_propagation.py` measures it against the true per-residue encoding.
`pooled |·|` is the fraction of the pooled displacement recovered when wild-type rows are
used outside the window; `cos` is its direction agreement.

**`PHOT_CHLRE`, L = 118, n = 128 — the hypothesis holds:**

| radius | rows kept | pooled \|·\| | cos |
|---:|---:|---:|---:|
| 0 | 6.7% | 25.4% | 0.552 |
| 8 | 67.3% | 77.5% | 0.973 |
| 16 | 89.9% | **93.6%** | 0.993 |
| 32 | 98.1% | **99.3%** | 0.999 |

…but it holds for a reason that makes it useless: with ~7.9 substitutions on 118 residues,
a radius-16 window already covers **89.9%** of the chain. The fidelity is bought by not
windowing.

**`CAPSD_AAV2S`, L = 735, n = 64 — it fails:**

| radius | rows kept | pooled \|·\| | cos |
|---:|---:|---:|---:|
| 0 | 0.6% | 18.0% | 0.324 |
| 8 | 4.4% | 32.4% | 0.635 |
| 16 | 6.6% | 37.6% | 0.697 |
| 32 | 10.9% | 43.8% | 0.771 |
| 64 | 19.6% | 52.3% | 0.860 |
| 128 | 37.0% | **69.7%** | 0.946 |

On CAPSD, keeping 37% of the rows recovers 70% of the displacement with cosine 0.946 — and
at the radius where the cache would actually shrink 15× (79.6 GB → 5.3 GB at radius 16),
it recovers 37.6% with cosine 0.697. **Not usable.**

**Windowing is strictly dominated.** Where it saves storage it does not preserve the
signal, and where it preserves the signal it does not save storage. `rows kept` is the
union of one window per substituted site, counted per variant, not `(2r+1)/L` — the mean
variant has 4–8 substitutions and their windows overlap by an amount that depends on how
clustered the sites are, so it has to be counted rather than assumed.

### Why, and why the two columns disagree

Note CAPSD's radius 0 row: `rowwise` 63.5% against `pooled` 18.0%. Most of the *squared row
displacement* is at the mutated sites, and yet most of the *pooled* displacement is not.
Those are consistent: large displacements at a handful of mutated rows point in
idiosyncratic directions that partly cancel when summed, while the thousands of small
displacements across the rest of the chain are mutually aligned and survive averaging.

**The mutation's effect on the pooled vector is carried by a diffuse, chain-wide shift, not
by the substituted residues.** On a 118-residue protein with ~8 substitutions per variant
there is no "rest of the chain" to speak of, so the window works. On a 735-residue protein
with ~5 substitutions, the window misses the mechanism entirely.

This also explains the earlier 5.2% figure properly: the radius-0 reconstruction fails
because the effect is *delocalised*, not because the window is slightly too tight.

### Consequence for the cache

**There is no cheap per-residue format. Store the full `L×d`.** Windowing is refused for
the same reason the radius-0 reconstruction was: it scores well on cosine against the
pooled vector while discarding most of what distinguishes a variant. The measured cost of
the honest format, at fp16 d = 1280:

| | full | radius 32 | radius 16 |
|---|---:|---:|---:|
| seven `planning` assays, every variant | **216.7 GB** | 108.4 GB | 81.9 GB |
| fidelity on CAPSD | 100% | 43.8% | 37.6% |

Windowing was measured because it looked like a 3× saving, and the measurement is what
rules it out. Recorded as a negative result rather than dropped.

### A fixed random projection would work, and is not needed

The other way to shrink the cache: the row displacement has effective rank 19–40 of 1280
(§1), so a Johnson–Lindenstrauss projection to a few hundred dimensions should be close to
isometric. Measured on real cached rows, projecting to **256**:

| assay | median distance error | p95 | effective rank, 1280 → 256 |
|---|---:|---:|---:|
| `PHOT_CHLRE` | 3.2% | 9.0% | 18.9 → 17.9 |
| `CAPSD_AAV2S` | 2.8% | 8.3% | 25.2 → 23.4 |
| `GFP_AEQVI` | 2.9% | 8.2% | 32.8 → 29.0 |

It works: 5× smaller for ~3% median distortion, and unlike projecting through the trained
reward adapter it is label-free and adds no dependency on Phase 2. **It is declined anyway**
— rented block storage is ~$0.02/hr for 200 GB, so the 5× buys pennies, while the
projection is a deviation from the per-residue state Step 3.1 registers and costs ~8% of
the effective rank that Step 3.2's gate is measured on. Spending accuracy on the gate metric
to save pennies is the wrong trade. Kept here so the option is on the record with a number
against it rather than being re-proposed later.

---

## 3. Where the encode has to run

Measured on this machine, SaProt 650M, per-residue:

| | throughput | 43.6M residues |
|---|---:|---:|
| local GTX 1650 (4 GB) | **270 res/s** | **45 hours** |
| rented A40, measured 2026-11-27 | ~26,000 res/s | **28 min** (~$0.30) |

Storage is not the constraint — `/home` has 464 GB free. **Throughput is, by 96×.** The
per-residue encode belongs on the rented box, and because the cache it produces is tens of
GB against a measured ~10 MB/s link, the training belongs there too: bring home metrics and
a checkpoint, not the cache.
