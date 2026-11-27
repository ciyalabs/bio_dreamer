"""Step 2.2 follow-up — cache latents AT THE MUTATED POSITIONS, not just a mean.

H1 failed and the diagnosis landed on the representation: a plain
`RidgeCV(pooled_embedding ++ mutation_count)` scores +0.390 against the trained head's
+0.409. Every configuration tried reads the same 1280-d **mean-pooled** SaProt vector,
which cannot express *which* position changed -- and Step 1.4 measured that per-residue
structure is recoverable and load-bearing (position top-1 0.999 against chance 0.019).

`pool_first` was adopted on a measurement taken on **stability**, 1,000 Tsuboyama single
mutants of 36-72 residue domains, and then generalised to multi-mutant fitness on
proteins up to 735 residues. That generalisation was never tested. This script makes
testing it cheap.

**It does not cache whole per-residue tensors.** Those would be ~170 GiB for the corpus.
The hypothesis is about the mutated positions, so for each variant this stores the
latents at its own mutated positions (k x 1280, mean k ~ 5) plus one full wild-type
latent per assay. That is ~3 GB, and the GPU cost is unchanged either way -- pooling
ever only saved storage, never compute.

Wild types are reconstructed by reverting each variant's substitutions, and the
reconstruction is cross-checked: every assay must yield exactly one wild type, and each
substitution's stated mutant residue must match the sequence it came from. Both held for
all 11 assays when this was written.

    python -I scripts/encode_positional.py --instance benchmark --test-cap 30000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biodreamer.protein_dreamer.cache import sequence_hash
from biodreamer.protein_dreamer.encoder import ProteinEncoder
from biodreamer.reward.data import assays_for, detection_limits, h1_split, load_assay

OUT_DIR = Path("data/processed/positional_cache")
LOG = Path("data/processed/positional_encoding_log.json")


def parse_subs(mut: str) -> list[tuple[str, int, str]]:
    """'K3R:V55A' -> [('K', 3, 'R'), ('V', 55, 'A')]; positions are 1-based."""
    out = []
    for sub in mut.split(":"):
        out.append((sub[0], int(sub[1:-1]), sub[-1]))
    return out


def wild_type(seq: str, mut: str, assay: str) -> str:
    """Revert a variant's substitutions. Raises if the mutant residue disagrees."""
    s = list(seq)
    for wt, pos, mt in parse_subs(mut):
        if s[pos - 1] != mt:
            raise ValueError(
                f"{assay}: substitution {wt}{pos}{mt} expects {mt} at position {pos}, "
                f"sequence has {s[pos-1]}"
            )
        s[pos - 1] = wt
    return "".join(s)


def resolve_wild_type(a, assay: str, probes: int = 60) -> str:
    """One wild type per assay, cross-checked across spaced-out variants."""
    step = max(1, len(a.sequence) // probes)
    seen = {wild_type(a.sequence[i], a.mutant[i], assay)
            for i in range(0, len(a.sequence), step)}
    if len(seen) != 1:
        raise ValueError(f"{assay}: reverting gave {len(seen)} distinct wild types")
    return seen.pop()


def encode_assay(enc: ProteinEncoder, assay: str, limits, test_cap: int,
                 batch_size: int, seed: int) -> dict:
    a = load_assay(assay, limits)
    tr, te = h1_split(a.order, seed=seed)
    if len(te) > test_cap:
        te = np.sort(np.random.default_rng(seed).choice(te, test_cap, replace=False))
    idx = np.concatenate([tr, te])
    split = np.concatenate([np.zeros(len(tr), np.int8), np.ones(len(te), np.int8)])

    wt = resolve_wild_type(a, assay)
    wt_lat = enc.encode([wt], batch_size=1, use_cache=False)[0].astype(np.float16)

    pooled = np.empty((len(idx), wt_lat.shape[1]), np.float16)
    pos_lat, offsets, positions = [], [0], []
    t0 = n_res = 0
    t0 = time.time()
    for s in range(0, len(idx), batch_size):
        b = idx[s : s + batch_size]
        zs = enc.encode(list(a.sequence[b]), batch_size=batch_size, use_cache=False)
        for r, (z, i) in enumerate(zip(zs, b)):
            n_res += z.shape[0]
            pooled[s + r] = z.mean(axis=0).astype(np.float16)
            p = np.array([pos for _, pos, _ in parse_subs(a.mutant[i])], np.int32)
            if p.max() > z.shape[0] or p.min() < 1:
                raise ValueError(f"{assay}: position {p} outside length {z.shape[0]}")
            pos_lat.append(z[p - 1].astype(np.float16))
            positions.append(p)
            offsets.append(offsets[-1] + len(p))
        if s % (batch_size * 40) == 0:
            print(f"    {s + len(b):,}/{len(idx):,}", flush=True)

    dest = OUT_DIR / f"{enc.name.replace('/', '_')}__{assay}.npz"
    np.savez_compressed(
        dest,
        keys=np.array([sequence_hash(s) for s in a.sequence[idx]]),
        pooled=pooled,
        pos_latents=np.concatenate(pos_lat),
        offsets=np.array(offsets, np.int64),
        positions=np.concatenate(positions),
        wt_latent=wt_lat,
        order=a.order[idx].astype(np.int32),
        test_cap=np.int64(test_cap),   # the resume check reads this; see main()
        score=a.score[idx].astype(np.float32),
        censored=a.censored[idx],
        split=split,          # 0 = H1 train, 1 = H1 test
    )
    secs = time.time() - t0
    print(f"  {assay}: {len(idx):,} variants, {n_res:,} residues, {secs:.0f}s "
          f"({n_res/max(secs,1e-9):,.0f} res/s), {dest.stat().st_size/1e6:.0f} MB",
          flush=True)
    return {"n_variants": len(idx), "n_train": int(len(tr)), "n_test": int(len(te)),
            "residues": int(n_res), "seconds": round(secs, 1),
            "mb": round(dest.stat().st_size / 1e6, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="benchmark", choices=["planning", "benchmark"])
    ap.add_argument("--backbone", default="saprot")
    ap.add_argument("--test-cap", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-separated assay subset, for smoke tests")
    args = ap.parse_args()

    names = assays_for(args.instance)
    if args.only:
        want = set(args.only.split(","))
        names = [n for n in names if n in want]
        if not names:
            raise SystemExit(f"--only matched nothing; choose from {assays_for(args.instance)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    limits = detection_limits()
    enc = ProteinEncoder(args.backbone)
    print(f"{args.instance}: {len(names)} assays, test capped at {args.test_cap:,}\n")

    log, t0 = {}, time.time()
    for n in names:
        dest = OUT_DIR / f"{enc.name.replace('/', '_')}__{n}.npz"
        if dest.exists():
            # A file written with a different --test-cap holds a different test set, and
            # silently skipping it would mix caps across assays. A smoke run at cap 200
            # must not satisfy a real run at cap 30,000.
            with np.load(dest) as old:
                cached = int(old["test_cap"]) if "test_cap" in old else -1
            if cached == args.test_cap:
                print(f"  {n}: already encoded, skipping "
                      f"({dest.stat().st_size/1e6:.0f} MB)")
                continue
            print(f"  {n}: cached at test_cap {cached:,} but this run wants "
                  f"{args.test_cap:,} -- re-encoding")
        log[n] = encode_assay(enc, n, limits, args.test_cap, args.batch_size, args.seed)

    payload = json.loads(LOG.read_text()) if LOG.exists() else {}
    payload[args.instance] = {"groups": log, "total_seconds": round(time.time() - t0, 1),
                              "backbone": enc.name, "test_cap": args.test_cap}
    LOG.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n{time.time()-t0:.0f}s total. wrote {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
