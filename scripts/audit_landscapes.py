"""Characterise combinatorial fitness landscapes: completeness, order, epistasis headroom.

Step 0.2 of RESEARCH_PLAN.md asks for the mutation-order histogram per source; this
produces it, plus the numbers that decide whether step 1.1 has anything to find.

A low additive R^2 is ambiguous on its own: it can mean genuine epistasis, or a
landscape where almost every variant is dead and the signal is a thin tail. The
`frac_dead` and `gini`-style spread columns are reported alongside so the two can
be told apart before step 1.1 draws any conclusion.

Run from the repo root, passing the data directory as an argument:

    python -I scripts/audit_landscapes.py data/raw/ssmula/extracted/data

Deliberately takes paths as arguments and is kept outside the data directory, since
downloaded archives are untrusted input.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEQ_ALIASES = ("variants", "variant", "aas", "muts", "sequences", "sequence", "seq", "genotype")
FIT_ALIASES = ("fitness", "fitness mean", "score", "y", "value", "activity")
NUCLEOTIDES = set("ACGT")


def _pick(cols: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for a in aliases:
        if a in cols:
            return cols[a]
    return None


def audit(csv: Path) -> dict:
    """Reads one landscape and reports its shape, order histogram and additive headroom."""
    df = pd.read_csv(csv)
    # strip BOM and whitespace that the published CSVs carry on the first header
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    seq_col = _pick(cols, SEQ_ALIASES)
    fit_col = _pick(cols, FIT_ALIASES)
    if seq_col is None or fit_col is None:
        return {"name": csv.stem, "error": f"columns not recognised: {list(df.columns)[:8]}"}

    seqs = df[seq_col].astype(str).str.strip().str.upper()
    fit = pd.to_numeric(df[fit_col], errors="coerce")

    keep = fit.notna() & seqs.str.fullmatch(r"[A-Z]+")
    seqs, fit = seqs[keep], fit[keep]
    modal_len = int(seqs.str.len().mode()[0])
    same_len = seqs.str.len() == modal_len
    seqs, fit = seqs[same_len], fit[same_len]

    # replicate rows exist in several of these files; one row per genotype
    n_rows = len(seqs)
    dedup = pd.DataFrame({"seq": seqs.to_numpy(), "fit": fit.to_numpy()}).groupby(
        "seq", as_index=False
    )["fit"].mean()
    seqs = dedup["seq"]
    y = dedup["fit"].to_numpy()
    n_sites = modal_len

    arr = np.array([list(s) for s in seqs])
    alphabet = sorted({a for row in arr for a in row})
    wt = "".join(pd.Series(arr[:, i]).mode()[0] for i in range(n_sites))
    order = (arr != np.array(list(wt))).sum(axis=1)

    # additive (one-hot per site-residue) model: the null step 1.1 must beat
    X = pd.get_dummies(
        pd.DataFrame(arr, columns=[f"p{i}" for i in range(n_sites)]), dtype=float
    ).to_numpy()
    A = np.c_[X, np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    # is a low R^2 epistasis, or a dead landscape with a thin tail?
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    dead = float((y <= lo + 0.01 * span).mean()) if span > 0 else float("nan")
    # fraction of genotypes beating the wild type: "is there anything to find"
    wt_mask = (arr == np.array(list(wt))).all(axis=1)
    wt_fit = float(y[wt_mask][0]) if wt_mask.any() else float(np.median(y))
    frac_above_wt = float((y > wt_fit).mean())

    counts = pd.Series(order).value_counts().sort_index()
    return {
        "name": csv.stem,
        "rows_in_file": int(n_rows),
        "n_genotypes": int(len(y)),
        "n_sites": n_sites,
        "alphabet_size": len(alphabet),
        "theoretical_20": 20**n_sites,
        "completeness_vs_20": round(len(y) / 20**n_sites, 4),
        "wild_type": wt,
        "order_histogram": {int(k): int(v) for k, v in counts.items()},
        "frac_order_ge2": round(float((order >= 2).mean()), 4),
        "additive_r2": round(r2, 4),
        "nonadditive_variance_fraction": round(1.0 - r2, 4),
        "fitness_min": round(lo, 4),
        "fitness_max": round(hi, 4),
        "fitness_median": round(float(np.median(y)), 4),
        "frac_dead": round(dead, 4),
        "wild_type_fitness": round(wt_fit, 4),
        "frac_above_wild_type": round(frac_above_wt, 4),
        "alphabet_kind": "nucleotide" if set(alphabet) <= NUCLEOTIDES else "amino_acid",
    }


def main(root: str) -> int:
    files = sorted(Path(root).rglob("fitness_landscape/*.csv"))
    if not files:
        print(f"no fitness_landscape CSVs under {root}", file=sys.stderr)
        return 1

    out = []
    print(
        f"{'landscape':<10} {'genotypes':>9} {'sites':>5} {'complete':>9} "
        f"{'add_r2':>7} {'nonadd':>7} {'dead':>6} {'>WT':>6} {'kind':>5}"
    )
    for f in files:
        try:
            rec = audit(f)
        except Exception as exc:  # a malformed file must not stop the audit
            rec = {"name": f.stem, "error": f"{type(exc).__name__}: {exc}"}
        out.append(rec)
        if "error" in rec:
            print(f"{rec['name']:<10} ERROR {rec['error'][:60]}")
        else:
            print(
                f"{rec['name']:<10} {rec['n_genotypes']:>9,} {rec['n_sites']:>5} "
                f"{rec['completeness_vs_20']:>8.1%} {rec['additive_r2']:>7.3f} "
                f"{rec['nonadditive_variance_fraction']:>7.3f} {rec['frac_dead']:>6.1%} "
                f"{rec['frac_above_wild_type']:>6.1%} {rec['alphabet_kind'][:3]:>5}"
            )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    dest = Path("data/processed/landscape_audit.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/ssmula/extracted/data"))
