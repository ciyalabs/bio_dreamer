"""Union two pooled-embedding caches, keyed on sequence hash.

A rented box starts with an empty cache, so it encodes only what the job in front of it
needs. `scripts/window_c.sh` asks for the literature protocol's 75,082 sequences, so the
file it brings home for CAPSD holds 14,496 vectors while the local file for the same assay
holds the 31,263 of H1's split. **Copying one over the other silently destroys whichever
rows the other had**, and the failure is invisible: every downstream script would simply
report a KeyError on sequences it used to have, or worse, be handed a smaller test set and
carry on.

Both caches are keyed by `sequence_hash` of the variant sequence and hold float16 vectors
from the same frozen backbone, so the union is well defined and order does not matter.
Overlapping keys are checked rather than assumed: identical inputs through an identical
frozen model must give identical vectors, and if they do not, something is wrong with one
of the two runs and that is worth stopping for.

    python -I scripts/merge_pooled_cache.py data/processed/pooled_cache_pod
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEST = Path("data/processed/pooled_cache")


def merge_one(src: Path, dest: Path, atol: float) -> tuple[int, int, int]:
    with np.load(src) as s:
        skeys, semb = s["keys"], s["emb"]
    if not dest.exists():
        np.savez_compressed(dest, keys=skeys, emb=semb)
        return 0, len(skeys), 0
    with np.load(dest) as d:
        dkeys, demb = d["keys"], d["emb"]
    have = {k: i for i, k in enumerate(dkeys)}
    mismatched = 0
    for i, k in enumerate(skeys):
        j = have.get(k)
        if j is not None and not np.allclose(
            demb[j].astype(np.float32), semb[i].astype(np.float32), atol=atol
        ):
            mismatched += 1
    merged = {k: demb[i] for i, k in enumerate(dkeys)}
    added = 0
    for i, k in enumerate(skeys):
        if k not in merged:
            merged[k] = semb[i]
            added += 1
    order = sorted(merged)
    np.savez_compressed(
        dest, keys=np.array(order), emb=np.stack([merged[k] for k in order])
    )
    return len(dkeys), added, mismatched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory of pooled-cache npz files to merge IN")
    ap.add_argument("--atol", type=float, default=2e-2,
                    help="float16 tolerance when the same sequence is in both")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_dir = Path(args.src)
    files = sorted(src_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz under {src_dir}")
    total_bad = 0
    for f in files:
        dest = DEST / f.name
        if args.dry_run:
            with np.load(f) as s:
                n_src = len(s["keys"])
            n_dest = 0
            if dest.exists():
                with np.load(dest) as d:
                    n_dest = len(d["keys"])
            print(f"  {f.name:66s} src {n_src:>7,}  dest {n_dest:>7,}")
            continue
        before, added, bad = merge_one(f, dest, args.atol)
        total_bad += bad
        flag = f"  !! {bad} MISMATCHED" if bad else ""
        print(f"  {f.name:66s} {before:>7,} + {added:>7,} new{flag}")
    if total_bad:
        raise SystemExit(
            f"{total_bad} overlapping sequences disagree between the two caches -- the "
            "same frozen backbone on the same input must be deterministic, so one of the "
            "two runs is not what it claims to be. Not merging further."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
