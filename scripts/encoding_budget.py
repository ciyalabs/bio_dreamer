"""How many sequences must actually be encoded, and what that costs.

`scripts/cost_model.py` prices "Encode ProteinGym (210 trainable assays)" at 0.90
EFLOP, which assumes encoding all 2,465,767 variants. No registered protocol uses
all of them. H1 (Step 2.2) trains on under 300 single and double mutants per assay
and tests on unseen order>=3 variants, so it only touches assays that *have*
order>=3 variants — and the firewall bans several of the largest.

Getting this wrong in either direction is expensive: too small and a rental window
runs out of work half-finished, too large and the bill pays for latents nothing
reads. This measures it from the corpus rather than assuming.

The existing `data/processed/proteingym_order_histograms.json` covers only the 14
assays with >=10k multimutants. This does the full 217.

Run from the repo root (CPU only, a few minutes):

    python -I scripts/encoding_budget.py
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PG_ZIP = Path("data/raw/proteingym/DMS_ProteinGym_substitutions.zip")
OUT = Path("data/processed/encoding_budget.json")
REGISTRY = Path("splits/heldout.yaml")

# SaProt 650M: inference is 2*N*D FLOP, N = 650e6 parameters, D = residues
PARAMS = 650e6
BYTES_PER_VALUE = 2  # float16 cache
H1_TRAIN_PER_ASSAY = 300  # the registered cap in PREREGISTRATION.md


def survey(zip_path: Path) -> list[dict]:
    """Per-assay variant count, order histogram and sequence length."""
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".csv"))
        for i, name in enumerate(names, 1):
            assay = Path(name).stem
            df = pd.read_csv(
                io.BytesIO(z.read(name)), usecols=["mutant", "mutated_sequence"]
            )
            order = df["mutant"].astype(str).str.count(":") + 1
            lengths = df["mutated_sequence"].astype(str).str.len()
            hist = order.value_counts().sort_index()
            rows.append(
                {
                    "assay": assay,
                    "n": int(len(df)),
                    "length": int(lengths.median()),
                    "max_order": int(order.max()),
                    "n_order_1": int((order == 1).sum()),
                    "n_order_2": int((order == 2).sum()),
                    "n_order_ge3": int((order >= 3).sum()),
                    "histogram": {int(k): int(v) for k, v in hist.items()},
                }
            )
            if i % 50 == 0:
                print(f"  ... {i}/{len(names)} assays", flush=True)
    return rows


def cost(residues: float) -> dict[str, float]:
    eflop = 2 * PARAMS * residues / 1e18
    return {
        "residues": residues,
        "eflop": eflop,
        # the two cards that matter: the laptop, and the one being rented
        "gpu_hours_at_30_tflops": eflop * 1e18 / (30e12) / 3600,
        "laptop_hours_at_0.27_tflops": eflop * 1e18 / (0.27e12) / 3600,
        "cache_gb_d1280": residues * 1280 * BYTES_PER_VALUE / 1e9,
        "cache_gb_d128": residues * 128 * BYTES_PER_VALUE / 1e9,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-survey the zip")
    args = ap.parse_args()

    banned = set(yaml.safe_load(REGISTRY.read_text())["banned_proteingym_assays"])

    if OUT.exists() and not args.refresh:
        rows = json.loads(OUT.read_text())["assays"]
        print(f"reusing {OUT} ({len(rows)} assays); --refresh to re-survey")
    else:
        if not PG_ZIP.exists():
            print(f"missing {PG_ZIP}")
            return 1
        print(f"surveying {PG_ZIP} ...")
        rows = survey(PG_ZIP)

    df = pd.DataFrame(rows)
    df["banned"] = df["assay"].isin(banned)
    total_res = float((df["n"] * df["length"]).sum())

    print(f"\n{len(df)} assays, {df['n'].sum():,} variants, {total_res/1e6:,.0f}M residues")
    print(f"median sequence length {df['length'].median():.0f}")
    print(f"banned by the firewall: {df['banned'].sum()} assays, "
          f"{df.loc[df['banned'], 'n'].sum():,} variants")

    # --- what H1 actually needs ---
    h1 = df[(df["max_order"] >= 3) & ~df["banned"]].copy()
    h1_train = (H1_TRAIN_PER_ASSAY * len(h1))
    h1_test = int(h1["n_order_ge3"].sum())
    h1_res = float(
        (h1["length"] * (H1_TRAIN_PER_ASSAY + h1["n_order_ge3"])).sum()
    )
    print(f"\nH1 (step 2.2) -- assays with order>=3 and not banned: {len(h1)}")
    print(f"  train (<=300/assay)      {h1_train:>10,} sequences")
    print(f"  test  (all order>=3)     {h1_test:>10,} sequences")
    if len(h1):
        print("  largest test contributors:")
        for _, r in h1.nlargest(5, "n_order_ge3").iterrows():
            print(f"    {r['assay']:<36} {r['n_order_ge3']:>8,}  L={r['length']:>4}  max_order {r['max_order']}")

    # --- a frozen-encoder regression baseline wants singles+doubles broadly ---
    pool = df[~df["banned"]]
    pool_res = float((pool["length"] * (pool["n_order_1"] + pool["n_order_2"])).sum())

    budgets = {
        "proteingym_all_variants": cost(total_res),
        "h1_train_and_test": cost(h1_res),
        "all_singles_and_doubles_unbanned": cost(pool_res),
    }

    print(f"\n{'workload':<34}{'residues':>12}{'EFLOP':>8}{'GPU-h':>8}{'laptop-h':>10}{'GB d1280':>10}{'GB d128':>9}")
    print("-" * 91)
    for k, v in budgets.items():
        print(f"{k:<34}{v['residues']/1e6:>11,.0f}M{v['eflop']:>8.3f}"
              f"{v['gpu_hours_at_30_tflops']:>8.1f}{v['laptop_hours_at_0.27_tflops']:>10,.0f}"
              f"{v['cache_gb_d1280']:>10,.0f}{v['cache_gb_d128']:>9,.0f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "assays": rows,
                "banned": sorted(banned),
                "budgets": budgets,
                "h1_assays": sorted(h1["assay"].tolist()),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
