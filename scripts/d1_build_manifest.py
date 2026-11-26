"""Choose the proteins and mutants Step 1.2 will fold, before renting anything.

Step 1.2 needs ~50 proteins spanning length and fold diversity, ~20 single mutants
each, "deliberately including known strongly destabilising substitutions". Tsuboyama
2023 supplies all of it: 983 domains with per-variant ddG from a single consistent
assay, so destabilisation is measured rather than guessed.

This runs on CPU, locally, and writes a manifest. The folding script then does
nothing but consume it. That split is deliberate: selection logic is where the bugs
are, and a bug found while the rented GPU meter is running costs money. Review the
manifest, then fold it.

Stratification per domain, by measured ddG. **In Tsuboyama's convention dG is the
folding free energy and positive means stable, so a destabilising mutation has a
NEGATIVE ddG.** Verified against the data rather than assumed: ddG_ML correlates at
0.987 with (dG_mutant - dG_wildtype) over 326,820 pairs, 78.8% of substitutions are
negative, 13.2% are at or below -2 kcal/mol, and only 0.1% reach +2.

    severe      ddG <= -2.0 kcal/mol   the case most likely to move the structure
    moderate    -2.0 < ddG <= -1.0
    mild        -1.0 < ddG <= 0.0
    stabilising ddG > 0.0              rare, 0.1% of the corpus above +2

An unstratified random sample would be dominated by mild mutations and would make
"structure does not move" easy to conclude for the wrong reason. Getting the sign
backwards does worse: the first version of this script labelled destabilising
mutations "neutral" and found almost no "severe" ones, so the stratification
excluded precisely the cases it existed to include.

A second group is added for length. Tsuboyama's domains are all 36-72 residues, so
the 50 above carry almost no length diversity and cannot answer whether displacement
depends on length -- which Step 1.2's noise-floor measurement suggested it does
(0.132 at L=56 falling to 0.050 at L=389). The landscape parents supply that range
and already have shipped structures.

Run from the repo root (~5 min, streams a 698 MB CSV):

    python -I scripts/d1_build_manifest.py
    python -I scripts/d1_build_manifest.py --n-proteins 20 --n-mutants 10
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ZIP = Path("data/raw/tsuboyama/Processed_K50_dG_datasets.zip")
MEMBER = "Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
OUT = Path("data/processed/d1_fold_manifest.json")

COLS = ["name", "aa_seq", "ddG_ML", "mut_type", "WT_name"]
# (name, low, high) with destabilising NEGATIVE -- see the module docstring
BANDS = [
    ("severe", -np.inf, -2.0),
    ("moderate", -2.0, -1.0),
    ("mild", -1.0, 0.0),
    ("stabilising", 0.0, np.inf),
]

# length probe: the landscape parents, which span 56-389 residues and ship structures
SSMULA = Path("data/raw/ssmula/extracted/data")
LENGTH_PROBE = {
    "GB1": SSMULA / "GB1/GB1.fasta",
    "ParD3": SSMULA / "ParD3/ParD3.fasta",
    "TEV": SSMULA / "TEV/TEV.fasta",
    "TrpB": SSMULA / "TrpB/TrpB.fasta",
}
AMINO = "ACDEFGHIKLMNPQRSTVWY"


def stream_singles(chunksize: int = 500_000) -> pd.DataFrame:
    """Single substitutions only, with a measured ddG and a usable sequence."""
    keep = []
    with zipfile.ZipFile(ZIP) as z, z.open(MEMBER) as fh:
        for i, chunk in enumerate(
            pd.read_csv(fh, usecols=COLS, chunksize=chunksize, low_memory=False), 1
        ):
            c = chunk[chunk["mut_type"].astype(str).str.fullmatch(r"[A-Z]\d+[A-Z]")]
            c = c.dropna(subset=["ddG_ML", "aa_seq", "WT_name"])
            c = c[pd.to_numeric(c["ddG_ML"], errors="coerce").notna()]
            if len(c):
                keep.append(c)
            print(f"  chunk {i}: kept {len(c):,}", flush=True)
    df = pd.concat(keep, ignore_index=True)
    df["ddG_ML"] = df["ddG_ML"].astype(float)
    return df


def pick_proteins(df: pd.DataFrame, n: int, seed: int) -> list[str]:
    """Spread across the length distribution rather than sampling uniformly.

    Tsuboyama's domains cluster near 60-70 residues; a uniform sample would carry
    almost no length diversity, and Step 1.2 wants to know whether displacement
    depends on length.
    """
    per_wt = df.groupby("WT_name").agg(
        length=("aa_seq", lambda s: len(s.iloc[0])), n_mut=("ddG_ML", "size")
    )
    # need enough mutants to stratify at all
    per_wt = per_wt[per_wt["n_mut"] >= 20]
    if per_wt.empty:
        return []
    rng = np.random.default_rng(seed)
    per_wt = per_wt.sort_values("length")
    buckets = np.array_split(per_wt.index.to_numpy(), min(n, len(per_wt)))
    return [str(rng.choice(b)) for b in buckets if len(b)]


def pick_mutants(sub: pd.DataFrame, n: int, seed: int) -> list[dict]:
    """Even coverage of the ddG bands, so severe cases are not crowded out."""
    rng = np.random.default_rng(seed)
    per_band = max(1, n // len(BANDS))
    out: list[dict] = []
    for band, lo, hi in BANDS:
        pool = sub[(sub["ddG_ML"] > lo) & (sub["ddG_ML"] <= hi)]
        if pool.empty:
            continue
        take = pool.iloc[rng.permutation(len(pool))[:per_band]]
        for _, r in take.iterrows():
            out.append(
                {
                    "mut_type": str(r["mut_type"]),
                    "sequence": str(r["aa_seq"]),
                    "ddG": float(r["ddG_ML"]),
                    "band": band,
                }
            )
    # top up with the most destabilising remaining, never with random mild ones
    if len(out) < n:
        chosen = {m["mut_type"] for m in out}
        rest = sub[~sub["mut_type"].isin(chosen)].nsmallest(n - len(out), "ddG_ML")
        for _, r in rest.iterrows():
            band = next(b for b, lo, hi in BANDS if lo < r["ddG_ML"] <= hi)
            out.append(
                {
                    "mut_type": str(r["mut_type"]),
                    "sequence": str(r["aa_seq"]),
                    "ddG": float(r["ddG_ML"]),
                    "band": band,
                }
            )
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proteins", type=int, default=50)
    ap.add_argument("--n-mutants", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not ZIP.exists():
        print(f"missing {ZIP}")
        return 1

    print(f"streaming {MEMBER} ...")
    df = stream_singles()
    print(f"\n{len(df):,} single substitutions with a measured ddG, "
          f"{df['WT_name'].nunique():,} domains")

    wts = pick_proteins(df, args.n_proteins, args.seed)
    print(f"selected {len(wts)} domains\n")

    proteins, folds = [], 0
    for wt in wts:
        sub = df[df["WT_name"] == wt]
        # the wild-type sequence: the modal sequence length tells us nothing here,
        # so take any mutant's sequence and revert its own substitution
        ref = str(sub.iloc[0]["aa_seq"])
        m = str(sub.iloc[0]["mut_type"])
        pos, wt_aa = int(m[1:-1]), m[0]
        wt_seq = ref[: pos - 1] + wt_aa + ref[pos:] if 0 < pos <= len(ref) else ref

        muts = pick_mutants(sub, args.n_mutants, args.seed)
        if not muts:
            continue
        bands = pd.Series([m["band"] for m in muts]).value_counts().to_dict()
        proteins.append(
            {
                "wt_name": wt,
                "wt_sequence": wt_seq,
                "length": len(wt_seq),
                "mutants": muts,
                "bands": bands,
                "ddG_range": [
                    round(min(m["ddG"] for m in muts), 2),
                    round(max(m["ddG"] for m in muts), 2),
                ],
            }
        )
        folds += 2 + len(muts)  # wild type twice, for the folding noise floor
        print(f"  {wt[:44]:<44} L={len(wt_seq):>3}  {len(muts):>2} mutants  "
              f"ddG {proteins[-1]['ddG_range'][0]:>6.2f}..{proteins[-1]['ddG_range'][1]:>5.2f}  {bands}")

    # length probe: the landscape parents, random mutants, no ddG available
    rng = np.random.default_rng(args.seed)
    probe = []
    for name, fasta in LENGTH_PROBE.items():
        if not fasta.exists():
            print(f"  {name:<44} MISSING {fasta}")
            continue
        seq = "".join(
            l.strip() for l in fasta.read_text().splitlines() if not l.startswith(">")
        ).upper()
        seq = "".join(c for c in seq if c in AMINO)
        muts = []
        for _ in range(args.n_mutants):
            pos = int(rng.integers(len(seq)))
            alt = str(rng.choice([a for a in AMINO if a != seq[pos]]))
            muts.append(
                {
                    "mut_type": f"{seq[pos]}{pos + 1}{alt}",
                    "sequence": seq[:pos] + alt + seq[pos + 1 :],
                    "ddG": None,          # not measured for these
                    "band": "unmeasured",
                }
            )
        probe.append(
            {
                "wt_name": name,
                "wt_sequence": seq,
                "length": len(seq),
                "mutants": muts,
                "bands": {"unmeasured": len(muts)},
                "ddG_range": None,
            }
        )
        folds += 2 + len(muts)
        print(f"  [probe] {name:<37} L={len(seq):>3}  {len(muts):>2} random mutants")

    lengths = [p["length"] for p in proteins + probe]
    manifest = {
        "source": "Tsuboyama 2023 Dataset2_Dataset3 + landscape parents for length",
        "n_proteins": len(proteins) + len(probe),
        "n_proteins_ddG_stratified": len(proteins),
        "n_proteins_length_probe": len(probe),
        "n_mutants_total": sum(len(p["mutants"]) for p in proteins + probe),
        "n_folds": folds,
        "wild_type_folds_each": 2,
        "length_min": int(min(lengths)) if lengths else 0,
        "length_max": int(max(lengths)) if lengths else 0,
        "length_median": int(np.median(lengths)) if lengths else 0,
        "seed": args.seed,
        "proteins": proteins + probe,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n{manifest['n_proteins']} proteins "
          f"({len(proteins)} ddG-stratified + {len(probe)} length probe), "
          f"{manifest['n_mutants_total']} mutants, "
          f"{folds} folds (wild types folded twice for the noise floor)")
    print(f"lengths {manifest['length_min']}-{manifest['length_max']} "
          f"(median {manifest['length_median']})")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
