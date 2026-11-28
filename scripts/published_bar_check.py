"""Step 2.5 — is 0.55 the right bar for THIS assay list?

H1 registered `spearman_mean >= 0.55`, grounded as "the published band on the same
protocol: ProteinNPT 0.56, MULTI-evolve 0.61, Kermut 0.62, Cerebra 0.65". Every one of
those is a **mean over its own assay list**. Ours is eleven assays chosen for multi-mutant
depth, which is not a random draw from ProteinGym, and a mean over a harder list is not
comparable to a mean over an easier one.

Adjusting a threshold after seeing your own score is how a result stops meaning anything,
so this script never looks at our numbers. It reads ProteinGym's published **per-assay**
supervised table and asks one question: *do the published methods themselves score lower
on our eleven than on the other 206?* If they do, by how much, and is it consistent across
methods that share nothing but the benchmark?

**This does not license editing 0.55.** It measures an offset. What to do about it is an
amendment, argued in PREREGISTRATION.md, and the honest reading of the offset is bounded
by one caveat that the numbers here cannot remove: ProteinGym's supervised CV trains on
~80% of each assay and tests on the rest, singles included, while H1 trains on <300
order <= 2 variants and tests on unseen order >= 3. The two protocols are not comparable
in level. They are comparable in *relative difficulty across assays*, because the same
method and the same protocol are applied to both groups, and that is the only thing read.

    python -I scripts/published_bar_check.py

Input is `DMS_substitutions_Spearman_DMS_level.csv` from the ProteinGym repo (20 KB,
recorded in DATA_REGISTER.md); pass --table to point at a local copy.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from biodreamer.reward.data import assays_for  # noqa: E402

DEFAULT_TABLE = Path("data/raw/proteingym/DMS_substitutions_Spearman_DMS_level.csv")
OUT = Path("data/processed/published_bar_check.json")
# every supervised baseline in the published table. They are read together on purpose:
# one method scoring low on our list could be a quirk of that method, but Kermut,
# ProteinNPT and a plain one-hot encoding agreeing is a property of the assays.
METHODS = (
    "Kermut",
    "ProteinNPT",
    "MSA Transformer Embeddings",
    "ESM-1v Embeddings",
    "Tranception Embeddings",
    "One-Hot Encodings",
)


def _f(row: dict, col: str) -> float | None:
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(DEFAULT_TABLE))
    ap.add_argument("--instance", default="benchmark")
    args = ap.parse_args()

    path = Path(args.table)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- fetch benchmarks/DMS_supervised/substitutions/Spearman/"
            "DMS_substitutions_Spearman_DMS_level.csv from the ProteinGym repo"
        )
    rows = list(csv.DictReader(path.open()))
    ours = set(assays_for(args.instance))
    present = {r["DMS_id"] for r in rows} & ours
    if len(present) != len(ours):
        print(f"  WARNING: {sorted(ours - present)} absent from the published table")

    out: dict[str, dict[str, float]] = {}
    print(f"{'published method':<30}{'all':>14}{'ours':>14}{'rest':>14}{'offset':>10}")
    for m in METHODS:
        mine = [v for r in rows if r["DMS_id"] in ours and (v := _f(r, m)) is not None]
        rest = [
            v for r in rows if r["DMS_id"] not in ours and (v := _f(r, m)) is not None
        ]
        if not mine or not rest:
            print(f"{m:<30}{'absent from table':>52}")
            continue
        off = st.mean(mine) - st.mean(rest)
        out[m] = {
            "ours": st.mean(mine),
            "rest": st.mean(rest),
            "all": st.mean(mine + rest),
            "offset": off,
            "n_ours": len(mine),
            "n_rest": len(rest),
        }
        print(
            f"{m:<30}{st.mean(mine + rest):>+14.3f}{st.mean(mine):>+14.3f}"
            f"{st.mean(rest):>+14.3f}{off:>+10.3f}"
        )

    offs = [v["offset"] for v in out.values()]
    summary = {
        "offset_mean": st.mean(offs),
        "offset_min": min(offs),
        "offset_max": max(offs),
        "n_methods": len(offs),
        "registered_bar": 0.55,
        "difficulty_adjusted_bar": 0.55 + st.mean(offs),
    }
    print(
        f"\nOffset is negative for {sum(o < 0 for o in offs)}/{len(offs)} published "
        f"methods; mean {st.mean(offs):+.3f}, range {min(offs):+.3f} .. {max(offs):+.3f}."
    )
    print(
        f"A bar of {0.55:.2f} set from means over the WHOLE benchmark corresponds to "
        f"{summary['difficulty_adjusted_bar']:.3f} on this assay list."
    )
    print(
        "\nThis is an OFFSET, not a new threshold. 0.55 stands until an amendment says "
        "otherwise, and the protocols differ in level -- see this file's docstring."
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"per_method": out, "summary": summary}, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
