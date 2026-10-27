"""Convert a ProteinGym assay into HypercubeME's genotype format.

HypercubeME expects a tab-separated file whose first column is a colon-separated
mutation list in `<position><new_aa>` form, with the wild type as an empty string:

    mut_list	fitness	error
    	0.1564	0.0
    6E:7S	0.1718	0.0

ProteinGym writes `<wt_aa><position><new_aa>` (e.g. `A6E:L7S`), so the leading
wild-type residue is dropped. Positions are kept as published.

Usage:
    python -I scripts/to_hypercubeme.py <proteingym_assay.csv> <out.txt> [--max-order N] [--sample N]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

MUT = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def convert(src: Path, dest: Path, max_order: int | None, sample: int | None, seed: int) -> dict:
    df = pd.read_csv(src, usecols=["mutant", "DMS_score"])
    df["mutant"] = df["mutant"].astype(str)
    order = df["mutant"].str.count(":") + 1

    if max_order is not None:
        df, order = df[order <= max_order], order[order <= max_order]
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=seed)
        order = df["mutant"].str.count(":") + 1

    rows, skipped = [], 0
    for mutant, score in zip(df["mutant"], df["DMS_score"], strict=True):
        parts = []
        bad = False
        for tok in mutant.split(":"):
            m = MUT.match(tok.strip())
            if not m:
                bad = True
                break
            parts.append(f"{m.group(2)}{m.group(3)}")
        if bad:
            skipped += 1
            continue
        # HypercubeME sorts mutations by position; do it here so the keys are canonical
        parts.sort(key=lambda p: int(re.match(r"\d+", p).group(0)))
        rows.append((":".join(parts), score))

    # the wild type must be present or no hypercube has a corner at the origin
    have_wt = any(k == "" for k, _ in rows)
    if not have_wt:
        rows.insert(0, ("", 0.0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        fh.write("mut_list\tfitness\terror\n")
        for key, score in rows:
            fh.write(f"{key}\t{score}\t0.0\n")

    return {
        "source": str(src),
        "written": len(rows),
        "skipped_unparseable": skipped,
        "wild_type_added": not have_wt,
        "max_order": int(order.max()) if len(order) else 0,
        "dest": str(dest),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--max-order", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    info = convert(a.src, a.dest, a.max_order, a.sample, a.seed)
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
