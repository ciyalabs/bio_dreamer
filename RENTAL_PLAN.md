# GPU rental plan — ProteinDreamer, Phases 1–7

**Written:** 2026-11-26. Prices are list, **excl. VAT**, and move — re-check before ordering.
Model: `scripts/cost_model.py --gpu hetzner-gex45`.

---

## The one number that decides everything

The whole study needs **5.6 EFLOP** of compute. On a 24 GB card that is **52 GPU-hours**.
The cost model bills **810 instance-hours**, so the implied GPU utilisation is **6%**.

**You are not buying FLOPs, you are buying availability.** 94% of the rental is the machine
sitting idle while you read output, fix a script, or think. Every euro of optimisation is in
*reducing hours the box exists*, not in buying a faster card. A card twice as fast halves 52
hours and changes nothing about the other 758.

This is why the plan below is "develop locally, rent to execute". Today is the evidence it
works: Step 1.1 and half of Step 1.2 ran to completion on a laptop GTX 1650.

---

## What Hetzner actually offers

**The console link in your account is Hetzner Cloud, which has no GPU instances at all.** GPUs
are a separate product — *dedicated root servers*, the GEX line, ordered from
`hetzner.com/dedicated-rootserver/matrix-gpu/` and managed in Robot, not the Cloud console.
Different product, different interface, different billing.

| Model | GPU | VRAM | RAM | Disk | €/h | €/mo cap | Setup fee |
|---|---|---:|---:|---|---:|---:|---:|
| **GEX45** | RTX PRO 4000 Blackwell SFF | **24 GB** | 64 GB | 2× 512 GB NVMe | **0.3429** | 214 | **209** |
| GEX63 | RTX PRO 6000 Blackwell Max-Q | 96 GB | 128 GB | 2× 1.92 TB | 1.6009 | 999 | 999 |
| GEX131 | RTX PRO 6000 Blackwell Max-Q | 96 GB | 256 GB+ | 2× 960 GB+ | 2.4022 | 1,499 | 1,499 |
| GEX430 | 4× RTX PRO 6000 Max-Q | 384 GB | 128 GB | 2× 1.92 TB | 5.7676 | 3,599 | 3,599 |

**GEX45 is the right tier.** The largest thing this study runs is ESMFold inference, which
needs ~16 GB; everything else is a 650M frozen encoder plus small heads. GEX63 costs **4.8×
the setup fee** for VRAM nothing here uses.

Four properties of the billing that shape the plan:

1. **Hourly, capped at the monthly price.** No minimum term, cancellation immediate.
2. **The setup fee is charged per order — €209 every time you order a GEX45.** That fee equals
   **610 hours** of runtime, so re-ordering is the most expensive mistake available.
3. **Billing starts when the server is assigned, not when it is powered on.** Powering off does
   *not* pause it. Only cancellation stops the meter.
4. **GEX45 is Helsinki (HEL1) only.**

One third-party source claims setup fees were dropped for some models in Dec 2025, while
Hetzner's own pages still list them. **Verify on the live configurator** — if the fee is gone,
the strategy below relaxes a great deal.

---

## Cost, by how disciplined you are

| Scenario | instance-h | Total |
|---|---:|---:|
| **1 order, develop locally, rent to execute** | 810 | **€487** |
| 2 orders (one gap > 3 weeks) | 810 | €696 |
| 3 orders | 810 | €905 |
| 1 order, but living on the box interactively | 2,835 | **€1,065** |

**Discipline is worth ~€580** — more than the entire compute bill. The failure mode is not
picking the wrong card, it is treating the rented box as a workstation.

For comparison, at the same 810 instance-hours: an RTX 4090 on RunPod/Vast on-demand is ≈€721
with no setup fee, and ≈€396 on spot. Spot is cheapest on paper but gets preempted, which is
bad for long training runs, and gives no persistent disk.

**Honest caveat for a first rental:** a Hetzner GEX is **unmanaged bare metal**. The €209 buys
provisioned hardware, not a working environment — you install the OS via `installimage`, then
the NVIDIA driver and CUDA yourself. Budget 2–4 billed hours for that the first time. RunPod
and Vast hand you a prebuilt PyTorch+CUDA container in about two minutes. If the driver setup
is unappealing, take RunPod on-demand for Window A, learn what the workflow needs, and decide
on Hetzner afterwards.

---

## The decision rule for cancelling

An extra order costs €209 = **610 hours** = **~25 days** of just leaving it running.

> **Gap until you next need the GPU < 3 weeks → leave it running. > 4 weeks → cancel and
> re-order.** Between 3 and 4 weeks it is a wash; leave it running, because re-ordering also
> costs you the driver setup again.

---

## Window A — finish Phase 1

Everything here is scripted and non-interactive. **Nothing in this window needs a human
watching it.**

| Job | EFLOP | GPU-h | Why it needs the card |
|---|---:|---:|---|
| Encode ProteinGym (210 assays) | 0.90 | 8.3 | 52 days on the laptop |
| D1 ESMFold folds (~2,000 mutants) | 0.60 | 5.6 | weights exceed 4 GB VRAM |
| D2 backbone sweep (Step 1.3) | 0.50 | 4.6 | three backbones × subset |
| Encode Tsuboyama + held-out landscapes | 0.31 | 2.9 | |
| **Total** | **2.31** | **21.4** | budget **40 billed hours** |

**Cost: €209 setup + €14 runtime ≈ €223.**

### Before you order anything

The meter starts at assignment, so arrive with everything ready:

- [ ] Folding script written, committed, and dry-run on one sequence locally (CPU, slow, just
      to prove the code path).
- [ ] `scripts/d1_noise_floor.py` already done — **the folding noise floor is still needed**:
      fold every wild type **twice** and measure TM-score/RMSD between the two predictions
      before comparing any mutant. See `diagnostics/d1_structure.md`.
- [ ] One entry-point script that runs all four jobs unattended and logs to the tracker.
- [ ] `data/` download script tested — pull data on the box, do not upload it.
- [ ] A measured throughput number: **run `scripts/bench_encoder.py` first**, and replace the
      estimated 30 TFLOP/s in `scripts/cost_model.py`. Step 0.4 did exactly this for the
      laptop; the estimate here is a guess from the card's power budget.

### The storage decision that makes the rest cheap

`DATA_REGISTER.md` measured the cache: **ProteinGym at raw backbone dimension is 1.7 TiB** and
does not fit the GEX45's 1 TB, let alone a laptop.

**After the adapter, at d = 128, the same corpus is ~170 GiB.** So:

> **Never persist raw d=1280 latents.** Encode → project through the adapter → store d=128.
> Keep raw latents only for the small subset needed to fit the adapter.

That single choice is a 10× storage saving, fits the corpus on local disk, and makes most of
Phases 2–4 runnable without the encoder at all — the latent dynamics model never sees a
sequence, only cached latents. Re-encoding the whole corpus from scratch costs ~€4, so losing
the cache is an annoyance, not a disaster.

- [ ] `rsync` the d=128 cache home **before cancelling**. 170 GiB over the 1 Gbit port is
      ~25 min at line rate; budget a few hours.

---

## Window B — Phases 2 to 7

| Phase | Item | GPU-h | billed-h |
|---|---|---:|---:|
| 2 | Reward head train + benchmark | 0.5 | 120 |
| 3 | World model B + collapse diagnostics | 1.4 | 110 |
| 4 | World model A: flow matching + DDPM | 2.8 | 110 |
| 5 | SLIP synthetic planner validation | 0.9 | 80 |
| 6 | Policy comparison + H4 | 11.1 | 200 |
| 7 | Ablations (10 axes) × seeds | 13.9 | 150 |
| | **Total** | **30.6** | **770** |

**Cost if continuous with Window A: €264 runtime, no second setup fee → €487 all-in.**

770 billed hours is **32 days**. Spread across Phases 2–7 that is not 32 days of work, it is
~5 months of work with the box up 20% of the time — which is precisely when the per-order fee
versus the cancel-and-reorder rule above starts to matter. Expect **one or two orders**, so
**budget €500–700** for the study.

Phases 6 and 7 are 350 of the 770 hours and are **embarrassingly parallel** (seeds × policy
classes × ablation axes). They are the one place where a bigger box could pay: GEX430 is 4
GPUs at 2.4× the hourly rate of 4× GEX45, so it is worse per GPU-hour, but it finishes in a
quarter of the wall-clock, and wall-clock is what you are paying for. Decide this with measured
per-seed timings from Phase 6, not in advance.

---

## What stays on the laptop

Proven today, and the reason the bill is €487 rather than €1,065:

- All of Phase 0 and Step 1.1 — done, no GPU touched.
- Step 1.2's noise floor and Step 1.4's inverse-dynamics probe.
- Every data-preparation, audit, firewall and analysis script.
- Reward-head and world-model development against **cached** latents — small models on a 4 GB
  card, slow but correct. Debug here; run the real seeds on the rented box.
- All writing, all figures, the whole tracker.

---

## Order of operations

1. Verify the GEX45 setup fee and price on the live configurator. If the fee is gone, this plan
   gets much more flexible.
2. Finish Steps 1.4 locally. Write the folding script and the Window A entry point.
3. Order the GEX45. Install OS, NVIDIA driver, CUDA. Clone, `uv sync`, download data on the box.
4. Benchmark the card. **Update the 30 TFLOP/s estimate in `scripts/cost_model.py`.**
5. Run Window A unattended. Pull results and the d=128 cache home.
6. Apply the cancel rule. Continue into Window B when Phase 2 is ready to execute.

---

## Sources

Hetzner specs and prices read from Hetzner's own pages on 2026-11-26
([GPU matrix](https://www.hetzner.com/dedicated-rootserver/matrix-gpu/),
[GEX45](https://www.hetzner.com/dedicated-rootserver/gex44/),
[GEX63](https://www.hetzner.com/dedicated-rootserver/gex63/)). Billing mechanics corroborated by
[wz-it](https://wz-it.com/en/blog/hetzner-gpu-server-comparison/),
[fastgpu](https://fastgpu.co/providers/hetzner) and
[LowEndTalk](https://lowendtalk.com/discussion/207697/dedicated-compute-ressources-hetzner-like-with-hourly-billing);
the disputed setup-fee removal is claimed by [bex.co](https://bex.co/blog/2026/08/16/hetzner-rtx-pro-6000-blackwell-self-hosting-ai-inference-cost)
and not confirmed on Hetzner's pages. Comparison rates from
[Aliteq](https://aliteq.com/rtx-4090-price) and
[AltStreet](https://altstreet.investments/tools/gpu/prices/rtx-4090).
