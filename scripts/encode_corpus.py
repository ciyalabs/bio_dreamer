"""Encode a scoped set of sequences into the per-residue cache.

Scope comes from `scripts/encoding_budget.py`, which measured what the registered
protocols actually need. The headline: `scripts/cost_model.py` prices "Encode
ProteinGym" at 0.90 EFLOP, assuming all 2,465,767 variants. The firewall bans 8
assays holding 1,206,303 of them (49% of the corpus), and H1's protocol touches only
the 7 unbanned assays that have order>=3 variants -- 287,212 sequences, 64M residues,
**0.083 EFLOP**. Encoding the whole corpus would cost 15x what the experiment needs.

Workloads:

    h1          the 7 unbanned assays with order>=3: <=300 order<=2 per assay for
                training, all order>=3 for testing. 0.8 GPU-h, 164 GB at d=1280.
    landscapes  the held-out combinatorial landscapes, needed in full for Phase 6
                planning. ~108M residues.
    probe       a small fixed subset, for benchmarking the rented card on arrival.

Storage: a d=1280 per-residue cache is 2.5 KB per residue. `DATA_REGISTER.md`
measured the whole corpus at 1.7 TiB, which does not fit the GEX45's 1 TB disk. The
project's answer is to project through the trained adapter to d=128 before storing
-- a 10x saving -- but **the adapter does not exist until Phase 2**, so Window A can
only cache raw. That is fine at this scope: 164 GB for h1 fits with room to spare.
Do not widen the scope to the full corpus before the adapter exists.

Run from the repo root:

    python -I scripts/encode_corpus.py --workload probe      # 200 sequences, smoke test
    python -I scripts/encode_corpus.py --workload h1
    python -I scripts/encode_corpus.py --workload landscapes
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from biodreamer.protein_dreamer.cache import Cache, format_bytes
from biodreamer.protein_dreamer.encoder import ProteinEncoder

PG_ZIP = Path("data/raw/proteingym/DMS_ProteinGym_substitutions.zip")
BUDGET = Path("data/processed/encoding_budget.json")
REGISTRY = Path("splits/heldout.yaml")
SSMULA = Path("data/raw/ssmula/extracted/data")
OUT = Path("data/processed/encoding_log.json")

H1_TRAIN_PER_ASSAY = 300

LANDSCAPES = {
    "TrpB4": SSMULA / "TrpB/fitness_landscape/TrpB4.csv",
    "TEV": SSMULA / "TEV/fitness_landscape/TEV.csv",
    "GB1": SSMULA / "GB1/fitness_landscape/GB1.csv",
    "ParD2": SSMULA / "ParD2/fitness_landscape/ParD2.csv",
    "ParD3": SSMULA / "ParD3/fitness_landscape/ParD3.csv",
}


def h1_sequences(limit_assays: int | None = None) -> dict[str, list[str]]:
    """Per assay: up to 300 order<=2 for training, plus every order>=3 for testing."""
    if not BUDGET.exists():
        raise FileNotFoundError(
            f"{BUDGET} missing -- run scripts/encoding_budget.py first, so the scope "
            "is measured rather than assumed"
        )
    budget = json.loads(BUDGET.read_text())
    wanted = set(budget["h1_assays"][:limit_assays] if limit_assays else budget["h1_assays"])
    banned = set(yaml.safe_load(REGISTRY.read_text())["banned_proteingym_assays"])
    overlap = wanted & banned
    if overlap:
        raise RuntimeError(
            f"the firewall bans {sorted(overlap)} but they are in the H1 scope -- "
            "refusing to encode training data for a held-out evaluation set"
        )

    out: dict[str, list[str]] = {}
    with zipfile.ZipFile(PG_ZIP) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
            assay = Path(name).stem
            if assay not in wanted:
                continue
            df = pd.read_csv(
                io.BytesIO(z.read(name)), usecols=["mutant", "mutated_sequence"]
            )
            order = df["mutant"].astype(str).str.count(":") + 1
            train = df.loc[order <= 2, "mutated_sequence"].head(H1_TRAIN_PER_ASSAY)
            test = df.loc[order >= 3, "mutated_sequence"]
            seqs = pd.concat([train, test]).astype(str).str.upper().unique().tolist()
            out[assay] = seqs
    return out


def landscape_sequences() -> dict[str, list[str]]:
    """Full combinatorial landscapes. Phase 6 scores every candidate, so all of them."""
    from biodreamer.eval.landscape import Landscape  # noqa: PLC0415

    out = {}
    for name, path in LANDSCAPES.items():
        if not path.exists():
            print(f"  {name}: MISSING {path}")
            continue
        out[name] = Landscape.from_csv(path, name).sequences()
    return out


def probe_sequences(n: int = 200) -> dict[str, list[str]]:
    """A fixed small batch, to benchmark the card before committing to a long run."""
    from biodreamer.eval.landscape import Landscape  # noqa: PLC0415

    ls = Landscape.from_csv(LANDSCAPES["ParD3"], "ParD3")
    return {"probe": ls.sequences()[:n]}


WORKLOADS = {"h1": h1_sequences, "landscapes": landscape_sequences, "probe": probe_sequences}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit-assays", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="list the work and the storage cost, encode nothing")
    args = ap.parse_args()

    fn = WORKLOADS[args.workload]
    groups = fn(args.limit_assays) if args.workload == "h1" else fn()
    if not groups:
        print("nothing to encode")
        return 1

    n_seq = sum(len(v) for v in groups.values())
    n_res = sum(len(s) for v in groups.values() for s in v)
    est = n_res * 1280 * 2
    print(f"workload {args.workload}: {len(groups)} group(s), {n_seq:,} sequences, "
          f"{n_res/1e6:,.1f}M residues")
    print(f"  cache at d=1280 float16: {format_bytes(est)}")
    for g, seqs in groups.items():
        print(f"    {g:<36} {len(seqs):>9,} seqs  {sum(len(s) for s in seqs)/1e6:>7.1f}M res")
    if args.dry_run:
        print("\ndry run: nothing encoded")
        return 0

    enc = ProteinEncoder("saprot")
    cache = enc.cache
    log, t0 = {}, time.time()
    for g, seqs in groups.items():
        t = time.time()
        enc.encode(seqs, batch_size=args.batch_size, use_cache=True)
        dt = time.time() - t
        res = sum(len(s) for s in seqs)
        log[g] = {
            "n_sequences": len(seqs),
            "residues": res,
            "seconds": round(dt, 1),
            "residues_per_second": round(res / dt, 1) if dt > 0 else None,
        }
        print(f"  {g:<36} {len(seqs):>9,} seqs in {dt:>8.1f}s "
              f"({res/max(dt,1e-9)/1000:>7.1f}k res/s)", flush=True)

    total = time.time() - t0
    stats = cache.stats if hasattr(cache, "stats") else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing[args.workload] = {
        "groups": log,
        "total_seconds": round(total, 1),
        "backbone": enc.name,
        "cache_dir": str(getattr(cache, "root", "")),
    }
    OUT.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"\n{total:.0f}s total. wrote {OUT}")
    if stats is not None:
        print(f"cache: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
