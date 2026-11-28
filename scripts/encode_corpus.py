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
POOLED_DIR = Path("data/processed/pooled_cache")

H1_TRAIN_PER_ASSAY = 300

LANDSCAPES = {
    "TrpB4": SSMULA / "TrpB/fitness_landscape/TrpB4.csv",
    "TEV": SSMULA / "TEV/fitness_landscape/TEV.csv",
    "GB1": SSMULA / "GB1/fitness_landscape/GB1.csv",
    "ParD2": SSMULA / "ParD2/fitness_landscape/ParD2.csv",
    "ParD3": SSMULA / "ParD3/fitness_landscape/ParD3.csv",
}


def instance_sequences(instance: str) -> dict[str, list[str]]:
    """Every sequence one instance's H1 split needs: <=300 order<=2 plus all order>=3.

    `planning` excludes the 8 firewall-banned assays and is the only instance allowed
    to produce a Phase 6 number; `benchmark` may use them, for literature
    comparability. Deriving the lists from `biodreamer.reward.data` rather than here
    keeps the two-instance rule in one place.
    """
    from biodreamer.reward.data import load_instance  # noqa: PLC0415

    out = {}
    for name, (a, tr, te) in load_instance(instance).items():
        idx = np.concatenate([tr, te])
        out[name] = a.sequence[idx].tolist()
    return out


def transition_sequences(instance: str = "planning") -> dict[str, list[str]]:
    """EVERY variant of an instance's assays, for Phase 3's transition graph.

    `instance_sequences` returns H1's split: <=300 order<=2 variants plus all order>=3.
    That is the right set for the reward model and the wrong one for a world model,
    because a transition needs BOTH endpoints and the excluded low-order variants are
    precisely the parents. Measured on the cache as built, four of the seven planning
    assays yield fewer than 1,000 usable transitions out of 1,395,965 available.

    Encoding resumes per sequence on the sequence hash, so running this over an assay
    that is already partly cached pays only for what is missing -- about 59k sequences
    of the 354k total.
    """
    from biodreamer.reward.data import assays_for, detection_limits, load_assay  # noqa: PLC0415

    limits = detection_limits()
    return {n: load_assay(n, limits).sequence.tolist() for n in assays_for(instance)}


def literature_sequences() -> dict[str, list[str]]:
    """Only the variants the literature protocol actually draws, across its 3 seeds.

    The protocol samples n training sequences at random from the whole dataset plus 5,000
    for test, so the H1 split is the wrong subset -- CAPSD's H1 file holds 31,263 of its
    42,328 variants and the missing 11k are the low-order ones a random draw hits most
    often. But the whole dataset is the wrong subset too, and expensively so: 691,253
    sequences and 188M residues, against the 512 + 5,000 per seed the protocol can
    possibly touch. `literature_split` is shared with scripts/matched_literature.py so
    the encoder and the evaluator cannot disagree about which those are.
    """
    from biodreamer.reward.data import (  # noqa: PLC0415
        LITERATURE_MULTIMUTANT,
        LITERATURE_SEEDS,
        literature_split,
        load_assay,
    )

    out = {}
    for name in LITERATURE_MULTIMUTANT:
        a = load_assay(name)
        need: set[int] = set()
        for seed in LITERATURE_SEEDS:
            tr, te = literature_split(len(a.sequence), seed)
            need.update(tr.tolist())
            need.update(te.tolist())
        idx = np.array(sorted(need))
        out[name] = a.sequence[idx].tolist()
        print(f"  {name:34s} {len(idx):>6,} of {len(a.sequence):>7,} variants "
              f"({len(idx) / len(a.sequence):.1%}) across {len(LITERATURE_SEEDS)} seeds")
    return out


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
    """Full combinatorial landscapes, as whole proteins. Phase 6 scores every candidate.

    The CSVs store only the variable residues, so this must splice them into the
    parent -- `Landscape.sequences()` returns 4-character fragments for TrpB4, and a
    protein language model will happily embed those and return nothing useful. TEV
    ships a `Complete Sequence` column and is read from that directly; GB1 and TrpB4
    are spliced at their verified variable positions; ParD2 and ParD3 have no known
    positions and are skipped rather than silently encoded as fragments.
    """
    import pandas as pd  # noqa: PLC0415

    from biodreamer.eval.landscape import VARIABLE_POSITIONS, Landscape  # noqa: PLC0415

    out, skipped = {}, []
    for name, path in LANDSCAPES.items():
        if not path.exists():
            print(f"  {name}: MISSING {path}")
            continue
        if name == "TEV":
            df = pd.read_csv(path)
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            seqs = (
                df["Complete Sequence"].astype(str).str.strip().str.upper().unique().tolist()
            )
            out[name] = [s for s in seqs if s.isalpha()]
            continue
        if name not in VARIABLE_POSITIONS:
            skipped.append(name)
            continue
        out[name] = Landscape.from_csv(path, name).full_sequences()

    if skipped:
        print(f"  SKIPPED {skipped}: variable positions unknown, so the variants cannot")
        print("    be spliced into the parent. Encoding the fragments would produce")
        print("    embeddings of 3-residue 'proteins'. See VARIABLE_POSITIONS in")
        print("    biodreamer/eval/landscape.py -- needs the Ding 2023 site list.")
    return out


def probe_sequences(n: int = 200) -> dict[str, list[str]]:
    """A fixed small batch of REAL proteins, to benchmark the card.

    Taken from the D1 manifest, which holds whole wild-type and mutant sequences of
    36-390 residues. It must not come from `Landscape.sequences()`: those are the
    variable residues only -- three characters for ParD3 -- so a throughput number
    measured on them would be meaningless, and the encoder would be embedding
    3-residue "proteins" without complaining.
    """
    manifest = Path("data/processed/d1_fold_manifest.json")
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} missing -- run scripts/d1_build_manifest.py first"
        )
    out: list[str] = []
    for e in json.loads(manifest.read_text())["proteins"]:
        out.append(e["wt_sequence"])
        out.extend(m["sequence"] for m in e["mutants"])
        if len(out) >= n:
            break
    return {"probe": out[:n]}


WORKLOADS = {
    "h1": h1_sequences,
    "landscapes": landscape_sequences,
    "probe": probe_sequences,
    "planning": lambda: instance_sequences("planning"),
    "benchmark": lambda: instance_sequences("benchmark"),
    "literature": literature_sequences,
    "transitions": transition_sequences,
}


def pool_group(enc: ProteinEncoder, group: str, seqs: list[str], batch_size: int) -> Path:
    """Encode and store ONE mean-pooled vector per sequence.

    The reward head pools anyway, and training the adapter after pooling rather than
    before was measured to cost nothing: per-protein median Spearman 0.771 against
    0.753 on held-out Tsuboyama domains. The cache then drops from 615 GB to 2.4 GB
    for the benchmark instance -- small enough to bring home, which moves every later
    Phase 2 iteration off the rented box entirely.

    This serves the reward head only. The world model needs the per-residue tensor and
    must not read these files.
    """
    import numpy as np  # noqa: PLC0415

    from biodreamer.protein_dreamer.cache import sequence_hash  # noqa: PLC0415

    POOLED_DIR.mkdir(parents=True, exist_ok=True)
    dest = POOLED_DIR / f"{enc.name.replace('/', '_')}__{group}.npz"
    # One file per assay, so the instances overlap: `benchmark` contains all 7 of
    # `planning`'s assays plus 4 more. Re-encoding the shared ones would repeat 64M of
    # 240M residues -- about an hour of rented GPU for a byte-identical result.
    # Resume per SEQUENCE, not per file. Skipping a whole existing file was right while
    # every workload asked for the same H1 split, but the `literature` workload asks for
    # EVERY variant of assays whose file already holds only the split -- CAPSD's file has
    # 31,263 of its 42,328 -- and a whole-file skip would silently return the subset. A
    # hash-keyed merge keeps the original saving (all hashes present -> nothing encoded)
    # and additionally fills a partial file, which here is 47M residues instead of 188M.
    have: dict[str, np.ndarray] = {}
    if dest.exists():
        with np.load(dest) as old_npz:
            have = {k: v for k, v in zip(old_npz["keys"], old_npz["emb"])}
    wanted = {}
    for seq in seqs:
        wanted.setdefault(sequence_hash(seq), seq)
    todo = [seq for h, seq in wanted.items() if h not in have]
    if not todo:
        print(f"    {group}: all {len(wanted):,} sequences already encoded, skipping")
        return dest
    if have:
        print(f"    {group}: {len(have):,} cached, encoding {len(todo):,} more "
              f"({sum(len(s) for s in todo)/1e6:.1f}M res)", flush=True)
    vecs, keys = [], []
    for s in range(0, len(todo), batch_size):
        chunk = todo[s : s + batch_size]
        for seq, z in zip(chunk, enc.encode(chunk, batch_size=batch_size, use_cache=False)):
            vecs.append(z.mean(axis=0).astype(np.float16))
            keys.append(sequence_hash(seq))
        if (s // batch_size) % 200 == 0 and s:
            print(f"      {s:,}/{len(todo):,}", flush=True)
    for k, v in zip(keys, vecs):
        have[k] = v
    ordered = sorted(have)
    np.savez_compressed(
        dest,
        keys=np.array(ordered),
        emb=np.stack([have[k] for k in ordered]),
    )
    print(f"    -> {dest} ({dest.stat().st_size/1e6:.1f} MB, {len(ordered):,} vectors)")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit-assays", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="list the work and the storage cost, encode nothing")
    ap.add_argument("--pooled", action="store_true",
                    help="store ONE mean-pooled vector per sequence instead of the "
                         "per-residue matrix: 512x smaller, and measured to cost the "
                         "reward head nothing (see diagnostics/reward_head.md)")
    ap.add_argument("--backbone", default="saprot")
    ap.add_argument(
        "--assays", default="",
        help="comma-separated group names to keep from the workload. For re-encoding one "
             "assay without paying for the other ten.",
    )
    args = ap.parse_args()

    fn = WORKLOADS[args.workload]
    groups = fn(args.limit_assays) if args.workload == "h1" else fn()
    if args.assays:
        want = set(args.assays.split(","))
        missing = want - set(groups)
        if missing:
            raise KeyError(f"workload {args.workload!r} has no group(s) {sorted(missing)}")
        groups = {k: v for k, v in groups.items() if k in want}
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

    enc = ProteinEncoder(args.backbone)
    cache = enc.cache
    log, t0 = {}, time.time()
    for g, seqs in groups.items():
        t = time.time()
        if args.pooled:
            pool_group(enc, g, seqs, args.batch_size)
        else:
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
