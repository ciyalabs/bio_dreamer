"""Step 1.2, GPU half — fold the manifest and measure how far the structure moves.

Consumes `data/processed/d1_fold_manifest.json`, which `scripts/d1_build_manifest.py`
produced on CPU. This script contains no selection logic on purpose: choosing which
proteins and mutants to fold is where the bugs live, and a selection bug found while
a rented GPU is billing costs money. Build the manifest, read it, then fold it.

Three measurements per mutant, against the wild type:

    ca_rmsd        C-alpha RMSD after optimal superposition
    tm_score       length-normalised, so it is comparable across proteins
    tdi_hamming    fraction of Foldseek 3Di tokens that changed

**The third is the one that decides the architecture.** SaProt's structure channel
*is* the 3Di token string. If a point mutation leaves it unchanged, a per-step
structure stream carries literally no new information, whatever the coordinates do.

Every wild type is folded **twice**. ESMFold is deterministic given identical input,
but not necessarily across batch geometry, and the floor for "the structure moved"
must be the predictor's own reproducibility rather than zero. Step 1.2's encoder
noise floor turned out to be fp16 rounding at 1.4e-3 relative; the folding floor is
a different and larger number, and comparing mutants to anything before measuring it
would make "the structure does not move" unfalsifiable.

A documented confound, to be stated in the write-up rather than discovered in review:
sequence-based structure predictors are known to respond weakly to point mutations.
A null result here is partly a statement about ESMFold. It is still the right
measurement for the *architectural* decision -- the stream we would build consumes
predicted structure, so if the prediction does not move the stream is empty -- but it
is not evidence about real structure.

Run from the repo root:

    python -I scripts/d1_fold.py --dry-run          # no model; exercises everything else
    python -I scripts/d1_fold.py                    # needs ~16 GB VRAM
    python -I scripts/d1_fold.py --limit 3          # first three proteins only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biodreamer import tracking
from biodreamer.protein_dreamer.encoder import foldseek_3di

MANIFEST = Path("data/processed/d1_fold_manifest.json")
OUT_JSON = Path("data/processed/d1_structure.json")
PDB_DIR = Path("data/processed/d1_folds")
MODEL = "facebook/esmfold_v1"


# ---- geometry ----


def ca_coords(pdb_text: str) -> np.ndarray:
    """C-alpha coordinates in residue order, straight from ATOM records."""
    out = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            out.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(out, dtype=float)


def superimpose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kabsch: rotate `b` onto `a`, both already centred on their centroids."""
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(bc.T @ ac)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return bc @ rot


def ca_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or len(a) == 0:
        return float("nan")
    diff = (a - a.mean(0)) - superimpose(a, b)
    return float(np.sqrt((diff**2).sum(axis=1).mean()))


def tm_score(a: np.ndarray, b: np.ndarray) -> float:
    """TM-score for two structures of equal length under the identity alignment.

    d0 = 1.24 * (L - 15)^(1/3) - 1.8, the standard normalisation, which is what
    makes the score comparable between a 36-residue domain and a 390-residue one.
    Undefined below L = 16, where d0 goes non-positive.
    """
    if a.shape != b.shape or len(a) < 16:
        return float("nan")
    L = len(a)
    d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8
    d = np.linalg.norm((a - a.mean(0)) - superimpose(a, b), axis=1)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def tdi_hamming(a: str, b: str) -> float:
    """Fraction of 3Di tokens that differ. The structure channel SaProt reads."""
    if not a or len(a) != len(b):
        return float("nan")
    return sum(x != y for x, y in zip(a, b)) / len(a)


# ---- folding ----


class Folder:
    """ESMFold, loaded once. `dry_run` substitutes a shipped structure instead."""

    def __init__(self, dry_run: bool = False, chunk_size: int | None = 64) -> None:
        self.dry_run = dry_run
        self._model = None
        self._tokenizer = None
        self.chunk_size = chunk_size

    def load(self) -> None:
        if self.dry_run or self._model is not None:
            return
        import torch  # noqa: PLC0415 - optional until a real fold is requested
        from transformers import AutoTokenizer, EsmForProteinFolding  # noqa: PLC0415

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = EsmForProteinFolding.from_pretrained(MODEL, low_cpu_mem_usage=True)
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            model.esm = model.esm.half()  # trunk in fp16; the folding head stays fp32
        if self.chunk_size:
            model.trunk.set_chunk_size(self.chunk_size)
        self._model = model

    def fold(self, sequence: str, fallback_pdb: Path | None = None) -> str:
        if self.dry_run:
            if fallback_pdb is None or not fallback_pdb.exists():
                raise FileNotFoundError(
                    "--dry-run needs a shipped PDB to stand in for a fold; "
                    f"{fallback_pdb} is missing"
                )
            return fallback_pdb.read_text()
        import torch  # noqa: PLC0415

        self.load()
        with torch.no_grad():
            return self._model.infer_pdb(sequence)


# ---- driver ----


def measure_protein(
    entry: dict, folder: Folder, pdb_dir: Path, fallback: Path | None
) -> dict:
    """Fold one wild type twice and each of its mutants once."""
    name = entry["wt_name"].replace("/", "_").replace("|", "_")
    wt_seq = entry["wt_sequence"]
    d = pdb_dir / name
    d.mkdir(parents=True, exist_ok=True)

    wt_a = folder.fold(wt_seq, fallback)
    wt_b = folder.fold(wt_seq, fallback)
    (d / "wt_a.pdb").write_text(wt_a)
    (d / "wt_b.pdb").write_text(wt_b)

    ca_a, ca_b = ca_coords(wt_a), ca_coords(wt_b)
    _, tdi_a = foldseek_3di(d / "wt_a.pdb")
    _, tdi_b = foldseek_3di(d / "wt_b.pdb")

    floor = {
        "floor_ca_rmsd": ca_rmsd(ca_a, ca_b),
        "floor_tm_score": tm_score(ca_a, ca_b),
        "floor_tdi_hamming": tdi_hamming(tdi_a, tdi_b),
    }

    rows = []
    for m in entry["mutants"]:
        pdb = folder.fold(m["sequence"], fallback)
        path = d / f"{m['mut_type']}.pdb"
        path.write_text(pdb)
        _, tdi_m = foldseek_3di(path)
        ca_m = ca_coords(pdb)
        rows.append(
            {
                "mut_type": m["mut_type"],
                "band": m["band"],
                "ddG": m["ddG"],
                "ca_rmsd": ca_rmsd(ca_a, ca_m),
                "tm_score": tm_score(ca_a, ca_m),
                "tdi_hamming": tdi_hamming(tdi_a, tdi_m),
            }
        )

    return {
        "wt_name": entry["wt_name"],
        "length": entry["length"],
        **floor,
        "mutants": rows,
    }


def aggregate(results: list[dict]) -> dict[str, float]:
    """Signal against the folding noise floor, overall and by band."""
    floor_rmsd = np.nanmedian([r["floor_ca_rmsd"] for r in results])
    floor_tdi = np.nanmedian([r["floor_tdi_hamming"] for r in results])
    rows = [m for r in results for m in r["mutants"]]

    def med(key: str, rs=rows) -> float:
        v = [m[key] for m in rs if m[key] is not None and np.isfinite(m[key])]
        return float(np.median(v)) if v else float("nan")

    out = {
        "n_proteins": float(len(results)),
        "n_mutants": float(len(rows)),
        "folding_floor_ca_rmsd": float(floor_rmsd),
        "folding_floor_tdi_hamming": float(floor_tdi),
        "folding_floor_tm_score": float(
            np.nanmedian([r["floor_tm_score"] for r in results])
        ),
        "median_ca_rmsd": med("ca_rmsd"),
        "median_tm_score": med("tm_score"),
        "median_tdi_hamming": med("tdi_hamming"),
    }
    # The architectural number: does the structure channel move at all?
    # A floor of exactly zero is the best case, not a missing measurement -- ESMFold
    # is deterministic, so any nonzero displacement is signal and the ratio is simply
    # undefined. Report the absolute value then, and say which case it was.
    out["folding_floor_is_zero"] = float(floor_tdi == 0.0 and floor_rmsd == 0.0)
    out["tdi_signal_over_floor"] = (
        out["median_tdi_hamming"] / floor_tdi if floor_tdi > 0 else float("nan")
    )
    out["rmsd_signal_over_floor"] = (
        out["median_ca_rmsd"] / floor_rmsd if floor_rmsd > 0 else float("nan")
    )
    for band in ("severe", "moderate", "mild", "stabilising", "unmeasured"):
        sub = [m for m in rows if m["band"] == band]
        if sub:
            out[f"median_ca_rmsd_{band}"] = med("ca_rmsd", sub)
            out[f"median_tdi_hamming_{band}"] = med("tdi_hamming", sub)
            out[f"n_{band}"] = float(len(sub))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="no model: substitute a shipped structure for every fold")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"missing {MANIFEST} -- run scripts/d1_build_manifest.py first")
        return 1
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["proteins"][: args.limit]

    fallback = None
    if args.dry_run:
        fallback = Path("data/raw/ssmula/extracted/data/ParD3/ParD3.pdb")
        print(f"DRY RUN: every fold returns {fallback}")
        print("  -- geometry, 3Di and aggregation are exercised; the numbers are not real\n")

    folder = Folder(dry_run=args.dry_run, chunk_size=args.chunk_size)
    folder.load()

    results, t0 = [], time.time()
    for i, entry in enumerate(entries, 1):
        t = time.time()
        rec = measure_protein(entry, folder, PDB_DIR, fallback)
        results.append(rec)
        print(
            f"  [{i}/{len(entries)}] {entry['wt_name'][:34]:<34} L={entry['length']:>3} "
            f"floor rmsd {rec['floor_ca_rmsd']:.3f} tdi {rec['floor_tdi_hamming']:.4f} | "
            f"{len(rec['mutants'])} mutants in {time.time()-t:.0f}s",
            flush=True,
        )

    agg = aggregate(results)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"aggregate": agg, "dry_run": args.dry_run, "proteins": results}, indent=2)
        + "\n"
    )

    print(f"\n{'':4}{'metric':<28}{'floor':>10}{'mutant':>10}{'ratio':>8}")
    print("-" * 60)
    print(f"{'':4}{'Ca RMSD (A)':<28}{agg['folding_floor_ca_rmsd']:>10.4f}"
          f"{agg['median_ca_rmsd']:>10.4f}{agg['rmsd_signal_over_floor']:>8.1f}")
    ratio = agg["tdi_signal_over_floor"]
    print(f"{'':4}{'3Di token Hamming':<28}{agg['folding_floor_tdi_hamming']:>10.4f}"
          f"{agg['median_tdi_hamming']:>10.4f}"
          f"{f'{ratio:>8.1f}' if np.isfinite(ratio) else '   floor=0'}")
    print(f"{'':4}{'TM-score':<28}{agg['folding_floor_tm_score']:>10.4f}"
          f"{agg['median_tm_score']:>10.4f}")
    if agg["folding_floor_is_zero"]:
        print("\n  Folding floor is exactly zero -- ESMFold reproduced both wild-type\n"
              "  passes bitwise, so every nonzero displacement above is real signal.")
    print(f"\n  by ddG band (median):")
    for band in ("severe", "moderate", "mild", "stabilising", "unmeasured"):
        if f"median_ca_rmsd_{band}" in agg:
            print(f"    {band:<13} n={agg[f'n_{band}']:>4.0f}  "
                  f"rmsd {agg[f'median_ca_rmsd_{band}']:>7.4f}  "
                  f"3Di {agg[f'median_tdi_hamming_{band}']:>7.4f}")
    print(f"\n  {time.time()-t0:.0f}s total; wrote {OUT_JSON}")

    if not args.no_track and not args.dry_run:
        with tracking.run(
            "1.2",
            "struct_displacement",
            variant={"model": MODEL, "n_proteins": len(results)},
            datasets=[str(MANIFEST)],
            notes=(
                "Wild types folded twice to establish the predictor's own "
                "reproducibility as the floor. 3Di Hamming is the decisive metric: it "
                "is literally SaProt's structure channel."
            ),
        ) as r:
            r.log_metrics({k: v for k, v in agg.items() if np.isfinite(v)})
            r.log_artifact(str(OUT_JSON), kind="table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
