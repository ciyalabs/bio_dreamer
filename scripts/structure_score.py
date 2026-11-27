"""Step 2.4, lever 4 — the structure-conditioned score Kermut's largest ablation points at.

Kermut's component ablation puts its **structure kernel** first: removing it costs −0.065
mean Spearman, against −0.045 for the sequence kernel and −0.026 for the zero-shot mean
function. That kernel is built from **ProteinMPNN site-conditioned amino-acid
distributions** — an inverse-folding model's opinion about which residue belongs at each
position, given the fold.

**This is not the 3Di channel, and the 3Di result does not rule it out.**
`PREREGISTRATION.md` records that feeding SaProt real predicted 3Di *hurt* here (|ddG|
correlation +0.459 → +0.304), because ESMFold reports a confident fold even for variants at
ddG ≤ −2, so the tokens assert "still folded" for destabilised proteins. That finding
stands. But it is a property of pushing a structure token through a sequence model. Reading
an inverse-folding model's per-site distribution is a different mechanism, and it is only
ever evaluated on the **wild-type** fold here, so the failure mode measured there — a
variant's structure being predicted too confidently — cannot arise.

Two stages, both cheap:

  1. Fold each wild type once with ESMFold (reusing `scripts/d1_fold.py`'s loader, so there
     is one ESMFold configuration in this project rather than two).
  2. Run ProteinMPNN with `--unconditional_probs_only`, which returns p(aa | structure) at
     every position without conditioning on the native sequence — the quantity Kermut uses.

The output has the same shape as `logprob_matrix_*.npz`: one (L, 20) array per assay in
`zeroshot_baseline.AA` order, so `score_from_matrix` scores any variant and the result
slots in beside the sequence zero-shot as a feature or a GP prior mean.

    python -I scripts/structure_score.py --dry-run
    python -I scripts/structure_score.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from biodreamer.reward.data import LITERATURE_MULTIMUTANT, assays_for, load_assay

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_fold import Folder  # noqa: E402
from encode_positional import resolve_wild_type  # noqa: E402
from zeroshot_baseline import AA  # noqa: E402

PDB_DIR = Path("data/processed/wt_structures")
MPNN_DIR = Path("tools/ProteinMPNN")
MPNN_REPO = "https://github.com/dauparas/ProteinMPNN.git"
OUT = Path("data/processed/mpnn_matrix.npz")


def ensure_mpnn() -> Path:
    if not (MPNN_DIR / "protein_mpnn_run.py").exists():
        MPNN_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", MPNN_REPO, str(MPNN_DIR)],
            check=True, capture_output=True, timeout=600,
        )
    return MPNN_DIR


def run_mpnn(pdb: Path, out_dir: Path, seed: int = 0) -> np.ndarray:
    """(L, 20) log-probabilities over amino acids, in `AA` order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(MPNN_DIR / "protein_mpnn_run.py"),
         "--pdb_path", str(pdb), "--out_folder", str(out_dir),
         "--unconditional_probs_only", "1", "--seed", str(seed), "--batch_size", "1"],
        check=True, capture_output=True, timeout=1800,
    )
    hits = list((out_dir / "unconditional_probs_only").glob("*.npz"))
    if not hits:
        raise FileNotFoundError(f"ProteinMPNN wrote no probabilities for {pdb}")
    with np.load(hits[0]) as z:
        # (B, L, 21) log-probs over MPNN's alphabet; batch of 1
        lp = z["log_p"][0] if "log_p" in z else z[z.files[0]][0]
    mpnn_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    idx = [mpnn_alphabet.index(a) for a in AA]
    sub = lp[:, idx]
    # renormalise over the 20 real amino acids: X is dropped, so the rows must sum to 1
    return sub - np.log(np.exp(sub).sum(axis=1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="benchmark")
    ap.add_argument("--include-literature", action="store_true",
                    help="also do the 5 datasets of the matched literature protocol")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = list(assays_for(args.instance))
    if args.include_literature:
        names = sorted(set(names) | set(LITERATURE_MULTIMUTANT))

    wts: dict[str, str] = {}
    for n in names:
        a = load_assay(n)
        wts[n] = resolve_wild_type(a, n)
        print(f"  {n:34s} L={len(wts[n]):4d}")
    print(f"{len(names)} wild types, {sum(len(v) for v in wts.values()):,} residues total")
    if args.dry_run:
        return 0

    PDB_DIR.mkdir(parents=True, exist_ok=True)
    folder = Folder()
    for n, wt in wts.items():
        pdb = PDB_DIR / f"{n}.pdb"
        if pdb.exists():
            print(f"  {n}: structure cached")
            continue
        print(f"  folding {n} ({len(wt)} residues)...", flush=True)
        pdb.write_text(folder.fold(wt))

    ensure_mpnn()
    have = dict(np.load(OUT)) if OUT.exists() else {}
    for n, wt in wts.items():
        if n in have:
            print(f"  {n}: MPNN cached {have[n].shape}")
            continue
        mat = run_mpnn(PDB_DIR / f"{n}.pdb", PDB_DIR / "mpnn_out" / n)
        if mat.shape[0] != len(wt):
            raise ValueError(
                f"{n}: ProteinMPNN returned {mat.shape[0]} positions for a "
                f"{len(wt)}-residue wild type"
            )
        have[n] = mat.astype(np.float32)
        np.savez(OUT, **have)  # per assay, so an interrupt costs one
        # sanity: the structure should prefer the native residue more often than chance
        native = float(np.mean([mat[i].argmax() == AA.index(wt[i])
                                for i in range(len(wt)) if wt[i] in AA]))
        print(f"  {n:34s} L={mat.shape[0]:4d}  native-argmax recovery {native:.1%}")
    meta = Path("data/processed/mpnn_meta.json")
    meta.write_text(json.dumps({"assays": sorted(have), "source": MPNN_REPO}, indent=2))
    print(f"\nwrote {OUT} and {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
