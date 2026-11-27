"""Step 2.1 — how censored is the training data, and what should the loss do about it?

`RESEARCH_PLAN.md` requires this decided and written down before the reward head is
trained: "Silently regressing on floored values will corrupt every epistasis estimate
downstream." It offers two options, censored regression or principled exclusion. The
measurement says the two targets need different answers, and that neither option is
quite right for the fitness one.

**Fitness (DMS score).** Floor fractions range from 0.0% to 40.9% across the seven H1
assays, and the signature to look for is not the floor fraction but the *modal share*:
D7PM05 has 31.9% of its variants at one single exact value, which is a detection limit
rather than a measurement. Excluding them would throw away real information -- "this
variant is dead" is a fact worth learning. Regressing on them teaches a precise value
that does not exist.

The plan already specifies a **pairwise ranking loss** for fitness, and ranking is what
resolves this: a pair of two floored variants carries no information and is dropped; a
pair of one floored and one live variant carries the real signal and is kept. So the
decision is *neither* censored regression *nor* exclusion -- it is to use the censored
points only for the comparisons they can support.

**Stability (ddG).** Censoring is essentially absent: dG_ML is rail-clipped to
[-1, +5] but only 0.01% of values sit at a rail. The problem is precision, not
truncation -- 9.7% of measurements have a 95% confidence interval wider than
1 kcal/mol. So stability gets **inverse-variance weighting**, not censoring treatment.

Run from the repo root (CPU, a few minutes):

    python -I scripts/censoring_audit.py
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PG_ZIP = Path("data/raw/proteingym/DMS_ProteinGym_substitutions.zip")
TS_ZIP = Path("data/raw/tsuboyama/Processed_K50_dG_datasets.zip")
TS_MEMBER = "Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
BUDGET = Path("data/processed/encoding_budget.json")
OUT = Path("data/processed/censoring_audit.json")

# A value is "at the floor" if it sits within this fraction of the range above the
# minimum. Fixed in advance; the modal share below is the stronger signal anyway.
FLOOR_BAND = 0.01
# Above this share of variants at one exact value, the floor is a detection limit
# rather than a coincidence, and pairs among those variants must be dropped.
MODAL_LIMIT = 0.02
# Stability measurements wider than this 95% CI are down-weighted, not dropped.
CI_LIMIT = 1.0


def assay_censoring(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    vals, counts = np.unique(np.round(y, 6), return_counts=True)
    modal_i = int(np.argmax(counts))
    return {
        "n": float(len(y)),
        "min": lo,
        "max": hi,
        "frac_at_floor": float((y <= lo + FLOOR_BAND * span).mean()) if span > 0 else float("nan"),
        "frac_at_ceiling": float((y >= hi - FLOOR_BAND * span).mean()) if span > 0 else float("nan"),
        "modal_value": float(vals[modal_i]),
        "modal_share": float(counts[modal_i] / len(y)),
        "distinct_fraction": float(len(vals) / len(y)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-stability", action="store_true")
    args = ap.parse_args()

    if not BUDGET.exists():
        print(f"missing {BUDGET} -- run scripts/encoding_budget.py first")
        return 1
    assays = json.loads(BUDGET.read_text())["h1_assays"]

    print("FITNESS TARGET — the seven H1 assays\n")
    print(f"{'assay':<32}{'n':>9}{'floor':>8}{'modal':>8}{'distinct':>10}  verdict")
    print("-" * 82)
    fitness = {}
    with zipfile.ZipFile(PG_ZIP) as z:
        for a in assays:
            df = pd.read_csv(
                io.BytesIO(z.read(f"DMS_ProteinGym_substitutions/{a}.csv")),
                usecols=["DMS_score"],
            )
            rec = assay_censoring(pd.to_numeric(df["DMS_score"], errors="coerce").to_numpy())
            rec["detection_limit"] = bool(rec["modal_share"] >= MODAL_LIMIT)
            fitness[a] = rec
            verdict = (
                f"detection limit at {rec['modal_value']:.4g}"
                if rec["detection_limit"]
                else "no detection limit"
            )
            print(f"{a:<32}{rec['n']:>9,.0f}{rec['frac_at_floor']:>8.1%}"
                  f"{rec['modal_share']:>8.1%}{rec['distinct_fraction']:>10.1%}  {verdict}")

    censored = [a for a, r in fitness.items() if r["detection_limit"]]
    print(f"\n  {len(censored)} of {len(fitness)} assays carry a detection limit: {censored}")

    stability = {}
    if not args.skip_stability:
        print("\nSTABILITY TARGET — Tsuboyama single substitutions\n")
        keep = []
        with zipfile.ZipFile(TS_ZIP) as z, z.open(TS_MEMBER) as fh:
            want = {"ddG_ML", "dG_ML", "mut_type", "deltaG_t_95CI"}
            for ch in pd.read_csv(
                fh, usecols=lambda c: c in want, chunksize=400_000, low_memory=False
            ):
                keep.append(ch[ch["mut_type"].astype(str).str.fullmatch(r"[A-Z]\d+[A-Z]")])
        d = pd.concat(keep, ignore_index=True)
        g = pd.to_numeric(d["dG_ML"], errors="coerce").dropna()
        ci = pd.to_numeric(d.get("deltaG_t_95CI"), errors="coerce").dropna()
        rails = float(((g <= g.min() + 1e-6) | (g >= g.max() - 1e-6)).mean())
        stability = {
            "n": float(len(g)),
            "dG_min": float(g.min()),
            "dG_max": float(g.max()),
            "frac_at_rails": rails,
            "n_with_ci": float(len(ci)),
            "ci_median": float(ci.median()) if len(ci) else float("nan"),
            "frac_ci_over_1": float((ci > CI_LIMIT).mean()) if len(ci) else float("nan"),
        }
        print(f"  dG_ML clipped to [{g.min():.3f}, {g.max():.3f}]; "
              f"{rails:.2%} of values sit at a rail")
        print(f"  95% CI: median {stability['ci_median']:.3f} kcal/mol; "
              f"{stability['frac_ci_over_1']:.1%} wider than {CI_LIMIT}")
        print("  -> censoring is negligible; PRECISION is the problem")

    decision = {
        "fitness": {
            "rule": "pairwise ranking loss, dropping pairs where BOTH members are at the "
                    "detection limit",
            "why": "A floored pair carries no ordering information and would teach the model "
                   "a false tie-break. A floored-versus-live pair carries the real signal that "
                   "one variant is dead. Excluding censored variants entirely would discard "
                   "that, and regressing on them would assert a precise value the assay never "
                   "measured.",
            "detection_limit_assays": censored,
            "modal_share_threshold": MODAL_LIMIT,
        },
        "stability": {
            "rule": "inverse-variance weighted SmoothL1; no censoring treatment",
            "why": "dG_ML is rail-clipped but only ~0.01% of values reach a rail, so there is "
                   "nothing meaningful to censor. What varies is precision: ~10% of "
                   "measurements have a 95% CI wider than 1 kcal/mol, and weighting by it uses "
                   "every point in proportion to how much it is worth.",
            "ci_limit_kcal": CI_LIMIT,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"fitness_assays": fitness, "stability": stability, "decision": decision}, indent=2
    ) + "\n")

    print("\nDECISION")
    for target, rec in decision.items():
        print(f"  {target}: {rec['rule']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
