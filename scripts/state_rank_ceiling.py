"""Step 3.2 — the effective-rank CEILING the frozen encoder imposes, pooled vs per-residue.

Step 3.2's registered gate is `latent_effective_rank_ratio >= 0.5` against `d_model`. The
pooled run fails it at 0.022, and the reason is not the adapter: the frozen encoder's own
POOLED states have effective rank 6.6-21.8 of 1280, because every variant of one protein
is a near-identical sequence.

The obvious objection is that pooling is what destroyed the rank -- mean-pooling averages
away the positional variation between residue representations -- and that the per-residue
state Step 3.1 registers would not have that problem. This measures whether that is true.

It needs no new encoding: `positional_cache` already holds each assay's full wild-type
`L x d` latent plus every cached variant's rows at its own mutated positions.

Columns:
    pooled      effective rank of the mean-pooled variant vectors (what 3.2 was read on)
    wt rows     effective rank of the wild type's own L residue rows
    mut rows    effective rank of the variants' mutated-position rows
    all rows    the two row sets together -- the per-residue ceiling
    row disp    mutated-position row MINUS the wild-type row at the same position: the
                part that actually distinguishes a variant, with "which position is this"
                removed

    python -I scripts/state_rank_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CACHE = Path("data/processed/positional_cache")
OUT = Path("data/processed/state_rank_ceiling.json")


def effective_rank(x: np.ndarray) -> float:
    """exp(entropy of the normalised eigenspectrum). float64: the fp16 cache has a wide
    spectrum and the small eigenvalues carry the entropy."""
    x = x.astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    ev = s**2
    tot = ev.sum()
    if tot <= 0:
        return 0.0
    p = ev / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4096, help="rows sampled per quantity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    files = sorted(CACHE.glob("*.npz"))
    if not files:
        raise SystemExit(f"{CACHE} is empty -- run scripts/encode_positional.py")
    rng = np.random.default_rng(args.seed)
    res = {}
    print(f"{'assay':34s} {'L':>5} {'pooled':>8} {'wt rows':>9} {'mut rows':>9} "
          f"{'all rows':>9} {'row disp':>9}")
    for f in files:
        name = f.name.split("__")[1][: -len(".npz")]
        with np.load(f) as z:
            pooled = z["pooled"][: args.limit]
            wt = z["wt_latent"]
            pl = z["pos_latents"]
            pos = z["positions"]
        sel = rng.choice(len(pl), size=min(args.limit, len(pl)), replace=False)
        mut = pl[sel]
        disp = mut.astype(np.float32) - wt[pos[sel] - 1].astype(np.float32)
        row = {
            "length": int(wt.shape[0]),
            "d": int(wt.shape[1]),
            "pooled": effective_rank(pooled),
            "wt_rows": effective_rank(wt),
            "mut_rows": effective_rank(mut),
            "all_rows": effective_rank(np.concatenate([wt, mut])),
            "row_displacement": effective_rank(disp),
            "n_pooled": int(len(pooled)),
            "n_rows": int(len(mut)),
        }
        row["per_residue_gain"] = row["all_rows"] / max(row["pooled"], 1e-9)
        res[name] = row
        print(f"{name:34s} {row['length']:>5} {row['pooled']:>8.1f} "
              f"{row['wt_rows']:>9.1f} {row['mut_rows']:>9.1f} "
              f"{row['all_rows']:>9.1f} {row['row_displacement']:>9.1f}")

    d = next(iter(res.values()))["d"]
    gains = [r["per_residue_gain"] for r in res.values()]
    ceil = [r["all_rows"] for r in res.values()]
    print(f"\nall against d = {d}.  per-residue raises the ceiling "
          f"{min(gains):.1f}x-{max(gains):.1f}x, to {min(ceil):.0f}-{max(ceil):.0f}.")
    print(f"0.5 * d = {0.5 * d:.0f}, so the registered floor stays unreachable on the "
          f"per-residue state too:\nthe ceiling moves by a factor of ~3, and clearing "
          f"the floor would need a factor of ~{0.5 * d / max(ceil):.0f} more.")
    Path(args.out).write_text(
        json.dumps({"d_model_of_encoder": d, "registered_floor_rank": 0.5 * d,
                    "per_assay": res}, indent=2) + "\n"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
