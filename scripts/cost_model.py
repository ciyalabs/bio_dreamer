"""Compute-cost model for ProteinDreamer through the first paper.

Two regimes, because they give answers an order of magnitude apart and only one
of them is what a provider bills you for.

  FLOP floor      what the work irreducibly costs if the GPU were busy 100% of
                  the time. Derived from measured dataset sizes and the 650M
                  backbone. This is small, because the backbone is FROZEN:
                  encoding is a one-off inference pass and everything after it
                  trains small models on cached latents. Not a pretraining budget.

  Instance-hours  what you actually pay. Rental is billed per wall-clock hour of
                  a running instance, and in research the GPU idles while you
                  edit, debug and read results. This dominates, and it is a
                  function of how you work rather than what you compute.

    python -I scripts/cost_model.py [--gpu a100|h100|rtx4090] [--style batch|mixed|interactive]
"""

from __future__ import annotations

import argparse

# ---- measured ----
PROTEINGYM_RESIDUES = 972_380_198   # sum(variants * seq_len) from the reference file
LOCAL_TFLOPS = 0.27                 # laptop GTX 1650, measured in step 0.4

# (effective TFLOP/s for 650M inference, $/GPU-hour on-demand, late Oct 2026)
GPUS = {
    "rtx4090": (50.0, 0.89),
    "a100": (100.0, 1.16),
    "h100": (250.0, 2.90),
}

# EFLOP (1e18). Training uses 6*N*D, inference 2*N*D.
WORKLOAD = {
    ("1", "Encode ProteinGym (210 trainable assays)"): 0.90,
    ("1", "Encode Tsuboyama + held-out landscapes"): 0.31,
    ("1", "D2 backbone sweep (3 backbones, subset)"): 0.50,
    ("1", "D1 ESMFold folds (~2000 mutants)"): 0.60,
    ("2", "Reward head train + benchmark"): 0.05,
    ("3", "World model B + collapse diagnostics"): 0.15,
    ("4", "World model A: flow matching + DDPM ablation"): 0.30,
    ("5", "SLIP synthetic planner validation"): 0.10,
    ("6", "Policy comparison + H4 (4 x 3 x 5 seeds)"): 1.20,
    ("7", "Ablations (10 axes) x seeds"): 1.50,
}

# Instance-hours per phase under a disciplined batch workflow: develop locally,
# rent only to run scripted jobs. `style` scales this.
BATCH_HOURS = {
    "1": 40, "2": 120, "3": 110, "4": 110, "5": 80, "6": 200, "7": 150,
}
STYLE_MULTIPLIER = {
    "batch": 1.0,        # scripted jobs, instance up only while running
    "mixed": 1.8,        # some interactive debugging on the rented box
    "interactive": 3.5,  # live on the instance, idle GPU most of the time
}

STORAGE_USD_PER_TB_MONTH = 90
PROJECT_MONTHS = 9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="a100", choices=sorted(GPUS))
    ap.add_argument("--style", default="batch", choices=sorted(STYLE_MULTIPLIER))
    ap.add_argument("--months", type=int, default=PROJECT_MONTHS)
    ap.add_argument("--cache-gb", type=int, default=30,
                    help="persistent storage kept between sessions")
    ap.add_argument("--spot", action="store_true", help="assume ~45%% spot discount")
    a = ap.parse_args()

    tflops, usd_hr = GPUS[a.gpu]
    if a.spot:
        usd_hr *= 0.55
    mult = STYLE_MULTIPLIER[a.style]

    print(f"GPU {a.gpu} @ ${usd_hr:.2f}/h ({tflops:.0f} TFLOP/s effective)"
          f"{' [spot]' if a.spot else ''} | workflow: {a.style} (x{mult})")
    print(f"Laptop GTX 1650 measured at {LOCAL_TFLOPS} TFLOP/s "
          f"-> rented is ~{tflops / LOCAL_TFLOPS:.0f}x faster\n")

    print(f"{'Ph':<3} {'Item':<46} {'EFLOP':>6} {'FLOP-h':>7} {'inst-h':>7} {'USD':>7}")
    print("-" * 80)
    tot_ef = tot_floph = 0.0
    by_phase: dict[str, float] = {}
    for (ph, item), ef in WORKLOAD.items():
        h = ef * 1e18 / (tflops * 1e12) / 3600
        tot_ef += ef
        tot_floph += h
        by_phase[ph] = by_phase.get(ph, 0.0) + h
        print(f"{ph:<3} {item:<46} {ef:>6.2f} {h:>7.1f} {'':>7} {'':>7}")

    print("-" * 80)
    inst = sum(BATCH_HOURS.values()) * mult
    print(f"{'':<3} {'FLOP floor (GPU busy 100% of the time)':<46} "
          f"{tot_ef:>6.2f} {tot_floph:>7.1f} {'':>7} {tot_floph * usd_hr:>7.0f}")
    print(f"{'':<3} {'Instance-hours actually billed':<46} "
          f"{'':>6} {'':>7} {inst:>7.0f} {inst * usd_hr:>7.0f}")
    print(f"{'':<3} {f'  implied GPU utilisation':<46} "
          f"{'':>6} {'':>7} {100 * tot_floph / inst:>6.0f}% {'':>7}")

    storage = a.cache_gb / 1000 * STORAGE_USD_PER_TB_MONTH * a.months
    print(f"{'':<3} {f'Storage {a.cache_gb} GB x {a.months} mo @ ${STORAGE_USD_PER_TB_MONTH}/TB/mo':<46} "
          f"{'':>6} {'':>7} {'':>7} {storage:>7.0f}")
    print("-" * 80)
    print(f"{'':<3} {'TOTAL':<46} {'':>6} {'':>7} {inst:>7.0f} {inst * usd_hr + storage:>7.0f}")

    # the non-obvious one
    reenc = (0.90 + 0.31) * 1e18 / (tflops * 1e12) / 3600 * usd_hr
    big_cache = 250 / 1000 * STORAGE_USD_PER_TB_MONTH * a.months
    print(f"\n  Re-encoding the whole corpus costs ${reenc:.2f}; keeping a 250 GB cache for "
          f"{a.months} months costs ${big_cache:.0f}.")
    print(f"  Storing regenerable embeddings is ~{big_cache / reenc:.0f}x more expensive than "
          "recomputing them. Keep the cache local, not rented.")
    print("\n  Phase 1 needs ~"
          f"{by_phase['1']:.0f} FLOP-hours; step 1.1 needs no GPU at all and runs on the laptop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
