"""Pull published zero-shot scores out of ProteinGym's archive for this project's assays.

Why this exists. Every prior-mean term this project has used so far comes from a
**single-sequence** model -- SaProt or ESM-2 reading one protein in isolation. On the 11
`benchmark` assays that is a weak prior: ProteinGym's own per-assay table puts ESM-2 650M
at 0.256 mean Spearman while GEMME, which reads a multiple sequence alignment, is at
0.445. On `GFP_AEQVI_Sarkisyan_2016` the gap is 0.108 against 0.678. Alignment-based
evolutionary information is a whole category of signal the project had never used, and it
is published per variant, so it is a download rather than a computation.

The archive is one CSV per DMS assay with a column per baseline, NOT a directory per
model -- so a model is a column name here, not a file.

    python -I scripts/extract_pgym_priors.py <zs_scores.zip> [--out data/processed/pgym_priors.npz]

Writes one `<assay>/<model>` array per entry, aligned to that assay's mutant order as
`load_assay` returns it, plus `<assay>/covered` recording how many of this project's
variants the published file actually scores. Coverage is checked rather than assumed:
ProteinGym's file for an assay need not contain exactly the variants this project kept
after its own filtering, and a silently short join would corrupt the prior.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from biodreamer.reward.data import LITERATURE_MULTIMUTANT, assays_for, load_assay

# the published baselines worth carrying. ESM2_650M is included NOT because it is wanted
# as a prior -- this project computes its own -- but as a join check: our independently
# computed ESM-2 score must correlate near 1.0 with theirs on the shared variants, and if
# it does not, the mutant keys are not lining up and every other column is suspect too.
MODELS = ("GEMME", "EVE_ensemble", "MSA_Transformer_ensemble", "TranceptEVE_L", "ESM2_650M")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--out", default="data/processed/pgym_priors.npz")
    args = ap.parse_args()

    names = sorted(set(assays_for("benchmark")) | set(LITERATURE_MULTIMUTANT))
    out: dict[str, np.ndarray] = {}

    with zipfile.ZipFile(args.zip_path) as z:
        members = {Path(n).stem: n for n in z.namelist() if n.endswith(".csv")}
        for n in names:
            if n not in members:
                print(f"  {n:34s} NOT IN ARCHIVE -- skipped")
                continue
            with z.open(members[n]) as fh:
                head = pd.read_csv(fh, nrows=0)
            have = [m for m in MODELS if m in head.columns]
            with z.open(members[n]) as fh:
                df = pd.read_csv(fh, usecols=["mutant", *have])
            pub = df.drop_duplicates("mutant").set_index("mutant")

            a = load_assay(n)
            idx = pd.Index(a.mutant)
            covered = int(idx.isin(pub.index).sum())
            out[f"{n}/covered"] = np.array([covered, len(idx)])
            for m in have:
                out[f"{n}/{m}"] = pub[m].reindex(idx).to_numpy(dtype=np.float64)
            missing = [m for m in MODELS if m not in have]
            flag = f"  (absent: {','.join(missing)})" if missing else ""
            print(f"  {n:34s} {covered:>7,}/{len(idx):<7,} variants scored{flag}")

    if not out:
        raise SystemExit("nothing extracted -- check the archive path")
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(f"\nwrote {dest} ({dest.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
