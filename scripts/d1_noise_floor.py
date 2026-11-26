"""Step 1.2, local half — the encoder's numerical noise floor.

Step 1.2 asks whether a point mutation moves the structure enough to justify a
per-step structure stream in the world model. That comparison is meaningless
without knowing how much the encoder's output moves when *nothing* changes, and
the plan says so explicitly. This measures that floor.

**Measured result (2026-11-26): the floor is fp16 rounding, not randomness.** The backbone
runs in float16, whose epsilon is 9.77e-4, and the floor came out at 1.38-1.45e-3 on the three
short proteins -- 1.4x epsilon -- and at *exactly zero* on the two long ones, where cuBLAS
happened to select the same kernel on both sides of the comparison. A per-protein zero is
therefore not a measurement of perfect stability: **use the pooled floor, 1.45e-3, as the
threshold any structural displacement must clear.**

Three sources of variation, none of them biological:

    identical call     the same sequence, same batch, twice
    batch position     the same sequence alone, versus padded beside a longer one
    batch size         the same sequence at batch_size 1 versus 8

The third and especially the second are the ones that matter in practice: the
wild type and its mutants will not generally be encoded in identical batch
geometry, so any displacement smaller than this floor is an artifact.

It then measures the signal available **without folding anything**: how far the
embedding moves under a single substitution with the 3Di structure channel held
fixed. That isolates the sequence channel, which is exactly the quantity the
structure stream would have to beat to earn its place. The other half of Step 1.2
-- folding each mutant and measuring TM-score, RMSD and 3Di token change -- needs
a GPU that fits ESMFold and cannot run on this machine.

Run from the repo root:

    python -I scripts/d1_noise_floor.py
    python -I scripts/d1_noise_floor.py --proteins TEV --n-mutants 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from biodreamer import tracking
from biodreamer.protein_dreamer.encoder import ProteinEncoder, foldseek_3di

SSMULA = Path("data/raw/ssmula/extracted/data")
OUT_JSON = Path("data/processed/d1_noise_floor.json")

# wild-type structures shipped with the landscapes, so 3Di is real rather than unknown
PROTEINS = {
    "TEV": SSMULA / "TEV/TEV.pdb",
    "TrpB": SSMULA / "TrpB/TrpB.pdb",
    "ParD2": SSMULA / "ParD2/ParD2.pdb",
    "ParD3": SSMULA / "ParD3/ParD3.pdb",
    "GB1": SSMULA / "GB1/GB1.pdb",
}
AMINO = "ACDEFGHIKLMNPQRSTVWY"
PAD_PARTNER = "M" * 400  # a longer sequence, to force the target to be padded


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """‖a − b‖_F ÷ ‖a‖_F — scale-free, so it is comparable across proteins."""
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError(f"expected matching (L, d) arrays, got {a.shape} and {b.shape}")
    denom = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else float("nan")


def per_residue_shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1) / (np.linalg.norm(a, axis=1) + 1e-12)


def noise_floor(enc: ProteinEncoder, seq: str, tdi: str) -> dict[str, float]:
    """How much the output moves when the input does not."""
    # the cache would hand back the identical array and hide everything
    one = enc.encode([seq], structure=tdi, batch_size=1, use_cache=False)[0]
    again = enc.encode([seq], structure=tdi, batch_size=1, use_cache=False)[0]

    # Same sequence, padded beside a longer one. `encode` applies one structure
    # string to the whole batch, so a mixed-length batch cannot carry 3Di --
    # this comparison is therefore structure-free on both sides, which keeps it
    # a clean padding test rather than a padding-plus-structure test.
    padded = enc.encode(
        [seq, PAD_PARTNER[: max(len(seq) + 50, 400)]],
        structure=None,
        batch_size=2,
        use_cache=False,
    )[0]
    unpadded = enc.encode([seq], structure=None, batch_size=1, use_cache=False)[0]

    batched = enc.encode([seq] * 8, structure=tdi, batch_size=8, use_cache=False)[0]

    out = {
        "noise_identical_call": relative_l2(one, again),
        "noise_batch_position_nostruct": relative_l2(unpadded, padded),
        "noise_batch_size": relative_l2(one, batched),
    }
    out["noise_floor"] = max(v for v in out.values() if np.isfinite(v))
    return out


def mutation_signal(
    enc: ProteinEncoder, seq: str, tdi: str, n: int, seed: int
) -> dict[str, float]:
    """Displacement under one substitution, with the structure channel held fixed."""
    rng = np.random.default_rng(seed)
    wt = enc.encode([seq], structure=tdi, batch_size=1, use_cache=False)[0]

    shifts, local = [], []
    for _ in range(n):
        pos = int(rng.integers(len(seq)))
        alt = rng.choice([a for a in AMINO if a != seq[pos]])
        mut = seq[:pos] + alt + seq[pos + 1 :]
        emb = enc.encode([mut], structure=tdi, batch_size=1, use_cache=False)[0]
        shifts.append(relative_l2(wt, emb))

        # how much of the change stays within +/-5 residues of the substitution
        pr = per_residue_shift(wt, emb)
        lo, hi = max(0, pos - 5), min(len(seq), pos + 6)
        total = float(pr.sum())
        local.append(float(pr[lo:hi].sum()) / total if total > 0 else float("nan"))

    return {
        "n_mutants": float(n),
        "mutation_displacement_median": float(np.median(shifts)),
        "mutation_displacement_min": float(np.min(shifts)),
        "mutation_displacement_max": float(np.max(shifts)),
        "frac_displacement_within_5_residues": float(np.nanmedian(local)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteins", nargs="*", default=None)
    ap.add_argument("--n-mutants", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    want = PROTEINS if not args.proteins else {k: PROTEINS[k] for k in args.proteins}
    enc = ProteinEncoder("saprot")
    results = {}

    for name, pdb in want.items():
        if not pdb.exists():
            print(f"  {name:<6} MISSING {pdb}")
            continue
        seq, tdi = foldseek_3di(pdb)
        rec = noise_floor(enc, seq, tdi) | mutation_signal(
            enc, seq, tdi, args.n_mutants, args.seed
        )
        rec["length"] = float(len(seq))
        # A floor of exactly zero means the forward pass was bitwise identical under
        # all three perturbations -- the best case, not a missing measurement. The
        # ratio is then undefined, so the absolute displacement is what to read.
        floor = rec["noise_floor"]
        rec["noise_floor_below_float_resolution"] = float(floor == 0.0)
        rec["signal_to_noise"] = (
            rec["mutation_displacement_median"] / floor if floor > 0 else float("nan")
        )
        results[name] = rec
        snr = rec["signal_to_noise"]
        print(
            f"  {name:<6} L={len(seq):>4}  floor {rec['noise_floor']:.2e}  "
            f"(identical {rec['noise_identical_call']:.1e}, pad {rec['noise_batch_position_nostruct']:.1e}, "
            f"bs {rec['noise_batch_size']:.1e})  mutation {rec['mutation_displacement_median']:.4f}  "
            f"SNR {f'{snr:.0f}x' if np.isfinite(snr) else 'floor=0':>8}  "
            f"local {rec['frac_displacement_within_5_residues']:.1%}",
            flush=True,
        )

        if not args.no_track:
            with tracking.run(
                "1.2",
                "noise_floor",
                variant={"protein": name, "backbone": enc.name},
                datasets=[str(pdb)],
                seed=args.seed,
                notes=(
                    "SaProt with real 3Di from the shipped wild-type structure. Mutation "
                    "displacement holds the structure channel fixed, isolating the sequence "
                    "channel -- the quantity a per-step structure stream would have to beat."
                ),
            ) as r:
                r.log_metrics({k: v for k, v in rec.items() if np.isfinite(v)})

    if results:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
