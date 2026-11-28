"""Step 3.1 — encode the FULL per-residue state Z in R^(L x d) for transition endpoints.

Step 3.1 registers a per-residue state. Nothing in this project had one for variants:
`pooled_cache/` holds one vector per variant and `positional_cache/` holds only the rows
at each variant's own mutated positions. Both shortcuts to a full tensor were measured and
refused -- see `results/state_representation.md` and
`biodreamer/world_model/transitions.py`. So it has to be encoded.

**Scope comes from the transition index, not from the assay.** A world model needs states
that are transition endpoints, and `transition_index(assay, present=None)` names exactly
those from the mutant strings alone, with no cache involved. Encoding that set gives the
resulting cache 100% coverage by construction -- which is the point, because reading
transitions out of the H1-scoped pooled cache loses most of them (`D7PM05` 55 usable of
20,513, `Q6WV12` 92 of 34,111).

Measured scope for the 7 `planning` assays at `--max-transitions 60000`:

    assay                            L    states      Mres      GB
    PHOT_CHLRE_Chen_2023           118    86,116      10.2    26.0
    CAPSD_AAV2S_Sinai_2021         735    27,771      20.4    52.3
    Q8WTC7_9CNID_Somermeyer_2022   238    19,233       4.6    11.7
    Q6WV12_9MAXI_Somermeyer_2022   222    19,241       4.3    10.9
    GFP_AEQVI_Sarkisyan_2016       238    17,514       4.2    10.7
    D7PM05_CLYGR_Somermeyer_2022   235    12,040       2.8     7.2
    GCN4_YEAST_Staller_2018        281       119       0.0     0.1

about 119 GB and 46M residues. At the rented A40's measured ~26,000 res/s that is ~30
minutes; on this project's local GTX 1650, measured at 270 res/s per-residue, it is 48
hours. **This belongs on the rented box, and so does the training that reads it** -- the
cache is far larger than the ~10 MB/s link home.

Storage is a plain `.npy` memmap per assay, `(n_states, L, d)` fp16, written row by row,
plus a `.keys.npy` of sequence hashes and a `.done.npy` progress mask so an interrupted
run resumes instead of restarting.

    python -I scripts/encode_residue_states.py --assay GCN4_YEAST_Staller_2018   # tiny
    python -I scripts/encode_residue_states.py --instance planning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from biodreamer.protein_dreamer.cache import sequence_hash
from biodreamer.protein_dreamer.encoder import ProteinEncoder
from biodreamer.reward.data import assays_for, detection_limits, load_assay
from biodreamer.world_model.transitions import transition_index

OUT_DIR = Path("data/processed/residue_cache")
LOG = Path("data/processed/residue_encoding_log.json")


def paths(name: str, assay: str) -> tuple[Path, Path, Path]:
    stem = OUT_DIR / f"{name.replace('/', '_')}__{assay}"
    return (
        Path(f"{stem}.npy"),
        Path(f"{stem}.keys.npy"),
        Path(f"{stem}.done.npy"),
    )


def encode_assay(enc: ProteinEncoder, assay: str, max_transitions: int,
                 batch_size: int, seed: int, dry_run: bool) -> dict:
    a = load_assay(assay, detection_limits())
    used, src, _, _, _, _ = transition_index(
        assay, present=None, max_transitions=max_transitions, seed=seed
    )
    seqs = [str(s) for s in a.sequence[used]]
    if not seqs:
        print(f"  {assay}: no transitions, nothing to encode")
        return {"n_states": 0, "n_transitions": 0}
    L = len(seqs[0])
    bad = [i for i, s in enumerate(seqs) if len(s) != L]
    if bad:
        raise ValueError(
            f"{assay}: variant {used[bad[0]]} is {len(seqs[bad[0]])} residues but the "
            f"first is {L}; a ragged assay cannot share one (n, L, d) memmap"
        )
    keys = np.array([sequence_hash(s) for s in seqs])
    gb = len(seqs) * L * 1280 * 2 / 1e9
    print(f"  {assay}: {len(src):,} transitions, {len(seqs):,} states, L={L}, "
          f"{len(seqs) * L / 1e6:.1f}M res, {gb:.1f} GB", flush=True)
    if dry_run:
        return {"n_states": len(seqs), "n_transitions": int(len(src)),
                "length": L, "gb": round(gb, 2), "residues": len(seqs) * L}

    data_p, keys_p, done_p = paths(enc.name, assay)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Resume only against an IDENTICAL key list. A different --max-transitions samples a
    # different state set, and reusing rows across the two would pair row i of one scope
    # with key i of another -- silently mismatched states and actions.
    fresh = True
    if data_p.exists() and keys_p.exists() and done_p.exists():
        if np.array_equal(np.load(keys_p), keys):
            fresh = False
        else:
            print(f"    existing cache was built for a different state set; rewriting")
    if fresh:
        out = np.lib.format.open_memmap(
            data_p, mode="w+", dtype=np.float16, shape=(len(seqs), L, 1280)
        )
        done = np.zeros(len(seqs), dtype=bool)
        np.save(keys_p, keys)
    else:
        out = np.lib.format.open_memmap(data_p, mode="r+")
        done = np.load(done_p)
        print(f"    resuming: {int(done.sum()):,}/{len(done):,} already encoded")

    todo = np.flatnonzero(~done)
    t0, n_res = time.time(), 0
    for s in range(0, len(todo), batch_size):
        b = todo[s : s + batch_size]
        zs = enc.encode([seqs[i] for i in b], batch_size=batch_size, use_cache=False)
        for i, z in zip(b, zs):
            if z.shape != (L, 1280):
                raise ValueError(f"{assay}: encoder returned {z.shape}, expected "
                                 f"{(L, 1280)}")
            out[i] = z.astype(np.float16)
            n_res += L
        done[b] = True
        if (s // batch_size) % 20 == 0:
            el = time.time() - t0
            np.save(done_p, done)
            out.flush()
            rate = n_res / max(el, 1e-9)
            left = (len(todo) - s - len(b)) * L / max(rate, 1e-9)
            print(f"    {int(done.sum()):,}/{len(done):,}  {rate:,.0f} res/s  "
                  f"~{left / 60:.0f} min left", flush=True)
    np.save(done_p, done)
    out.flush()
    secs = time.time() - t0
    print(f"  {assay}: done, {n_res:,} residues in {secs / 60:.1f} min "
          f"({n_res / max(secs, 1e-9):,.0f} res/s), {gb:.1f} GB", flush=True)
    return {"n_states": len(seqs), "n_transitions": int(len(src)), "length": L,
            "residues": n_res, "seconds": round(secs, 1), "gb": round(gb, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="planning", choices=["planning", "benchmark"])
    ap.add_argument("--assay", action="append", default=None,
                    help="encode only these; repeatable")
    ap.add_argument("--backbone", default="saprot")
    ap.add_argument("--max-transitions", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the scope and cost, encode nothing")
    args = ap.parse_args()

    names = args.assay if args.assay else list(assays_for(args.instance))
    # largest last: a dry run then shows the cheap assays' real numbers before the
    # expensive one commits any GPU time
    print(f"{args.instance}: {len(names)} assays, "
          f"max_transitions={args.max_transitions:,}\n")
    enc = None if args.dry_run else ProteinEncoder(args.backbone)
    name = "westlake-repl/SaProt_650M_AF2" if enc is None else enc.name

    log, t0 = {}, time.time()
    for n in names:
        log[n] = encode_assay(enc, n, args.max_transitions, args.batch_size,
                              args.seed, args.dry_run)
    tot_gb = sum(v.get("gb", 0) for v in log.values())
    tot_res = sum(v.get("residues", 0) for v in log.values())
    print(f"\ntotal {tot_gb:.1f} GB, {tot_res / 1e6:.1f}M residues, "
          f"{(time.time() - t0) / 60:.1f} min")
    if not args.dry_run:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps(
            {"backbone": name, "max_transitions": args.max_transitions,
             "total_gb": round(tot_gb, 2), "per_assay": log}, indent=2) + "\n")
        print(f"wrote {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
