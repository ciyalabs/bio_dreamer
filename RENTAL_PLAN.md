# GPU rental plan

**Revised 2026-11-26** after measuring the work instead of estimating it. Prices are
list, excl. VAT, and move — check before ordering.

---

## The short version

| | what | where | cost |
|---|---|---|---|
| **Window A** | finish Phase 1: fold 1,188 structures | **hourly container** (RunPod / Vast) | **~$10** |
| **Window B** | Phases 2–7 | **Hetzner GEX45** | **~€470** |

Two providers, because the two jobs have opposite shapes. Window A is five hours of
work. Window B is five months of having a GPU around.

---

## Why two providers

The whole study needs **5.6 EFLOP** — about 50 GPU-hours. But the cost model bills
**810 instance-hours**, so the GPU is busy **6%** of the time. The other 94% is the
box sitting idle while you read output and fix scripts.

**So you are mostly paying for availability, not compute.** That is why a monthly
machine wins for the long phases. But it is also why a €209 one-off setup fee is
ruinous for a short job.

### Window A alone

| | 10 hours billed | 16 hours billed |
|---|---:|---:|
| Hetzner GEX45 | €212 — **98% of it the setup fee** | €214 |
| RunPod 4090, on demand | $8.90 | $14.24 |
| Vast 4090, typical | $4.70 | $7.52 |
| RunPod 4090, spot | $3.40 | $5.44 |

**15–25× cheaper on an hourly container.** It is also the easier first rental: a
prebuilt PyTorch container starts in two minutes, where Hetzner bare metal means
installing the NVIDIA driver yourself — budget 2–4 *billed* hours for that.

### The whole study

At 810 instance-hours the picture reverses, because Hetzner caps at €214/month and
the setup fee amortises:

| | total |
|---|---:|
| Hetzner GEX45, one order, disciplined | **€487** |
| RunPod 4090 on demand | €721 |
| Hetzner, one order, but working interactively on the box | €1,065 |

**Discipline is worth ~€580** — more than the compute bill. The failure mode is
treating the rented box as a workstation. Develop at home, rent to execute.

---

## Window A — finish Phase 1

**~4.8 GPU-hours.** Budget 10–16 billed hours.

| job | GPU-h | why it needs a rental |
|---|---:|---|
| D1: fold 1,188 structures | ~3.0 | ESMFold needs ~16 GB; the laptop card has 4 GB |
| D2 re-run with real 3Di | ~1.0 | settles the backbone choice |
| encoding throughput sample | ~0.8 | 200 sequences, to measure residues/second |

This is **seven times smaller** than the first version of this plan, for three
reasons found by measuring:

1. **Encoding shrank 15×.** The firewall bans 8 assays holding 49% of ProteinGym, and
   H1's protocol touches only the 7 unbanned assays with order≥3 variants — 64M
   residues, not 972M.
2. **Steps 1.3 and 1.4 already ran at home.** The laptop was enough. 1.3 only needs a
   re-run with 3Di; 1.4 is finished.
3. **Tsuboyama is already cached** locally by those two steps.

### What the box needs

Almost nothing: a **15 MB repo**, a **236 KB manifest**, and model weights that
download from Hugging Face. **No datasets.** `scripts/provision_container.sh` does
the whole setup.

### What comes home

**Under 100 MB:** the fold PDBs (~60 MB), three JSON result files, and the tracker
ledger.

**Do not bring the embedding cache home.** It is 152 GiB at d=1280, and re-encoding
it later costs 0.8 GPU-hours — far less than moving it. Step 1.3 fixed the latent
dimension at **d=128**, and the adapter that projects down to it does not exist until
Phase 2, so the real cache gets built once, in Window B, at **170 GiB instead of
1.7 TiB**.

---

## Window B — Phases 2 to 7

**~31 GPU-hours, 770 billed hours** (32 days of box time spread over ~5 months).

| phase | item | GPU-h | billed-h |
|---|---|---:|---:|
| 2 | reward head train + benchmark | 0.5 | 120 |
| 3 | world model B + collapse diagnostics | 1.4 | 110 |
| 4 | world model A: flow matching + DDPM | 2.8 | 110 |
| 5 | SLIP synthetic planner validation | 0.9 | 80 |
| 6 | policy comparison + H4 | 11.1 | 200 |
| 7 | ablations × seeds | 13.9 | 150 |

**Order the Hetzner GEX45 when Phase 2 is ready to run, not before.** Billing starts
when the server is assigned, not when you power it on — powering off does **not** pause
it. Only cancelling stops the meter.

**Cancel rule:** the €209 setup fee is worth **610 hours** of runtime, about 25 days.
So if the next GPU job is more than ~4 weeks away, cancel and re-order. Under 3 weeks,
leave it running.

Phases 6 and 7 are 350 of the 770 hours and are embarrassingly parallel (seeds ×
policies × ablation axes). That is the one place a bigger box could pay — decide it
from measured per-seed timings in Phase 6, not now.

---

## What to rent

### Window A: any 24 GB container

RunPod or Vast, RTX 4090 or A10/L4. ESMFold needs ~16 GB, so 24 GB is comfortable.
Spot is cheapest but can be preempted; Window A is short enough that on-demand is
worth the few dollars.

### Window B: Hetzner GEX45

| | |
|---|---|
| GPU | RTX PRO 4000 Blackwell SFF, **24 GB** |
| CPU / RAM | i5-13500, 64 GB |
| Disk | 2× 512 GB NVMe |
| Price | **€0.3429/h**, capped at **€214/month** |
| Setup fee | **€209, per order** |
| Location | Helsinki only |
| Term | none; cancellation immediate |

**Not Hetzner Cloud.** That product has no GPUs at all. The GEX line is *dedicated*
servers, ordered from `hetzner.com/dedicated-rootserver/matrix-gpu/`.

The bigger models are poor value here: GEX63 (96 GB) costs **€999 setup + €999/month**
for VRAM nothing in this study uses.

**Verify the setup fee on the live configurator.** Hetzner's pages list it, but one
source claims it was dropped for some models in Dec 2025. If it is gone, Window A
could go to Hetzner too.

---

## First thing on any rented box

```
python -I scripts/bench_encoder.py
```

Then replace the **estimated** 30 TFLOP/s for `hetzner-gex45` in
`scripts/cost_model.py` with the measured number. It is a guess from the card's power
budget, and every hour and euro above scales with it. Step 0.4 did exactly this for
the laptop.

---

## Order of operations

1. Start a RunPod or Vast container with a 24 GB card.
2. `bash scripts/provision_container.sh --from-host you@laptop:/path/to/bio_dreamer`
3. `python -I scripts/bench_encoder.py` — update the cost model.
4. `bash scripts/window_a.sh` — unattended, ~5 hours.
5. Copy the results home (<100 MB), commit, destroy the container.
6. Build Phase 2 at home. Order the Hetzner GEX45 when it is ready to run.

---

## Sources

Hetzner specs read from
[the GPU matrix](https://www.hetzner.com/dedicated-rootserver/matrix-gpu/) and
[GEX45](https://www.hetzner.com/dedicated-rootserver/gex44/) on 2026-11-26. Billing
mechanics from [wz-it](https://wz-it.com/en/blog/hetzner-gpu-server-comparison/),
[fastgpu](https://fastgpu.co/providers/hetzner) and
[LowEndTalk](https://lowendtalk.com/discussion/207697/dedicated-compute-ressources-hetzner-like-with-hourly-billing).
The disputed setup-fee removal is claimed by
[bex.co](https://bex.co/blog/2026/08/16/hetzner-rtx-pro-6000-blackwell-self-hosting-ai-inference-cost)
and not confirmed on Hetzner's own pages. Hourly rates from
[Aliteq](https://aliteq.com/rtx-4090-price) and
[AltStreet](https://altstreet.investments/tools/gpu/prices/rtx-4090).
