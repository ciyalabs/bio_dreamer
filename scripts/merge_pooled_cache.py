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
import os
import tempfile
from pathlib import Path

import numpy as np

DEST = Path("data/processed/pooled_cache")


def save_atomic(dest: Path, keys: np.ndarray, emb: np.ndarray) -> None:
    """Write to a sibling temp file, then rename over `dest`.

    `np.savez_compressed` truncates its target before writing, so an interrupt partway
    through leaves a corrupt file and the original contents are gone. That is not
    hypothetical: killing this script mid-merge destroyed a 494,794-vector HIS7 cache that
    had cost an hour of rented A40 to build, and the loss was silent until the next read
    raised BadZipFile. `rename` within a directory is atomic, so the destination is either
    the old file or the complete new one and never a prefix of either.
    """
    # The suffix MUST end in `.npz`. `np.savez_compressed` appends `.npz` to any path that
    # does not already end in it, so a `.npz.tmp` suffix makes numpy write to
    # `<tmp>.npz.tmp.npz` while the empty file mkstemp created keeps the name being
    # renamed -- which replaces every destination with a zero-byte file. That is not a
    # hypothetical either: the first version of this function did exactly that to ten
    # caches at once, and the only reason nothing was lost is that numpy's real output was
    # still sitting beside them under its mangled name.
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp.npz")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        np.savez_compressed(tmp_path, keys=keys, emb=emb)
        if tmp_path.stat().st_size == 0:
            raise RuntimeError(f"numpy wrote nothing to {tmp_path}")
        os.replace(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def merge_one(src: Path, dest: Path, atol: float) -> tuple[int, int, int]:
    with np.load(src) as s:
        skeys, semb = s["keys"], s["emb"]
    if not dest.exists():
        save_atomic(dest, skeys, semb)
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
    save_atomic(dest, np.array(order), np.stack([merged[k] for k in order]))
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
