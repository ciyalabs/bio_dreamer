"""How far does one substitution move the encoder's OTHER residue rows?

Step 3.1 registers a per-residue state `Z in R^(L x d)`. Storing it for every
transition endpoint costs 50.6 GB on `PHOT_CHLRE` and 216.7 GB over the seven
`planning` assays (`results/world_model_B.md`). Before paying that, the question
worth asking is how many rows actually carry the mutation's effect.

`diagnostics/` already has the r = 0 answer and it is bad: rebuilding a variant as
"wild-type rows with the mutated rows substituted in" keeps **5.2%** of the pooled
displacement on CAPSD. That is the reason a per-residue cache is needed at all. But
5.2% at r = 0 does not say whether the remaining 95% is spread over the whole chain
or concentrated in a neighbourhood -- and those two answers have very different
storage bills.

This measures it directly, against the true per-residue encoding:

  rowwise[r]   fraction of total squared row displacement ||z_var[i] - z_wt[i]||^2
               lying within r positions of the nearest mutated site.
  pooled[r]    fraction of the POOLED displacement recovered by keeping only those
               rows and taking wild-type rows everywhere else -- the same quantity
               the 5.2% measures, so r = 0 here must reproduce it.
  cos[r]       direction agreement of that recovered pooled displacement.

`pooled[r]` is the one that decides the cache format, because the pooled vector is
what every downstream head reads. Run:

    python -I scripts/mutation_propagation.py --assay CAPSD_AAV2S_Sinai_2021 -n 128
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from biodreamer.protein_dreamer.encoder import ProteinEncoder
from biodreamer.reward.data import detection_limits, load_assay

OUT = Path("data/processed/mutation_propagation.json")
POSITIONAL = Path("data/processed/positional_cache")
RADII = [0, 1, 2, 4, 8, 16, 32, 64, 128]
SUB = re.compile(r"([A-Z])(\d+)([A-Z])")


def mutated_sites(mut: str) -> np.ndarray:
    """0-based positions this variant substitutes."""
    sites = [int(m.group(2)) - 1 for m in SUB.finditer(mut)]
    if not sites:
        raise ValueError(f"no substitutions parsed from {mut!r}")
    return np.array(sites)


def measure(assay: str, n: int, backbone: str, batch: int, seed: int) -> dict:
    a = load_assay(assay, detection_limits())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(a.sequence), size=min(n, len(a.sequence)), replace=False)

    enc = ProteinEncoder(backbone)
    # The wild-type per-residue latent is read from the positional cache rather than
    # re-encoded, so this compares against exactly the reference every other positional
    # diagnostic in the project uses -- including the r = 0 figure this must reproduce.
    src = POSITIONAL / f"{enc.name.replace('/', '_')}__{assay}.npz"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing -- run scripts/encode_positional.py")
    with np.load(src) as f:
        z_wt = f["wt_latent"].astype(np.float32)
    L = z_wt.shape[0]
    if len(a.sequence[int(idx[0])]) != L:
        raise ValueError(f"{assay}: cached wild type has {L} rows but a variant is "
                         f"{len(a.sequence[int(idx[0])])} residues; the row/position "
                         "mapping would be wrong")
    pooled_wt = z_wt.mean(0)

    rows = {r: [] for r in RADII}
    pooled = {r: [] for r in RADII}
    cos = {r: [] for r in RADII}
    # the UNION of one window per mutated site, which is what storage actually costs.
    # min(2r+1, L)/L would be right only for single mutants; the mean variant here has
    # several substitutions and their windows overlap by an amount that depends on how
    # clustered the sites are, so it has to be counted per variant rather than assumed.
    kept = {r: [] for r in RADII}
    full_disp = []

    seqs = [a.sequence[i] for i in idx]
    muts = [a.mutant[i] for i in idx]
    for s0 in range(0, len(seqs), batch):
        chunk = seqs[s0 : s0 + batch]
        embs = enc.encode(chunk, batch_size=batch, use_cache=False)
        for seq, mut, z in zip(chunk, muts[s0 : s0 + batch], embs, strict=True):
            z = z.astype(np.float32)
            d = z - z_wt                           # (L, d) row displacements
            e = (d * d).sum(1)                     # (L,) squared row displacement
            tot = e.sum()
            true_pooled = z.mean(0) - pooled_wt    # the displacement that matters
            nt = np.linalg.norm(true_pooled)
            full_disp.append(nt)
            sites = mutated_sites(mut)
            # distance from every row to its NEAREST mutated site
            dist = np.abs(np.arange(L)[:, None] - sites[None, :]).min(1)
            for r in RADII:
                keep = dist <= r
                rows[r].append(float(e[keep].sum() / max(tot, 1e-12)))
                kept[r].append(float(keep.sum() / L))
                # rebuild with wild-type rows outside the window; the pooled
                # displacement is then just the kept rows' displacement over L
                rec = d[keep].sum(0) / L
                pooled[r].append(float(np.linalg.norm(rec) / max(nt, 1e-12)))
                cos[r].append(float(rec @ true_pooled / max(np.linalg.norm(rec) * nt, 1e-12)))
        print(f"  {assay}: {min(s0 + batch, len(seqs))}/{len(seqs)}", flush=True)

    return {
        "assay": assay, "length": L, "n_variants": len(seqs), "backbone": enc.name,
        "wild_type_latent_from": str(src),
        "mean_pooled_displacement": float(np.mean(full_disp)),
        "rowwise_fraction": {str(r): float(np.mean(rows[r])) for r in RADII},
        "pooled_norm_recovered": {str(r): float(np.mean(pooled[r])) for r in RADII},
        "pooled_direction_cosine": {str(r): float(np.mean(cos[r])) for r in RADII},
        "rows_stored_fraction": {
            str(r): float(np.mean(kept[r])) for r in RADII
        },
        "mean_sites_per_variant": float(
            np.mean([len(mutated_sites(m)) for m in muts])
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", action="append", required=True)
    ap.add_argument("-n", "--n-variants", type=int, default=128)
    ap.add_argument("--backbone", default="saprot")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    res = {}
    for name in args.assay:
        res[name] = measure(name, args.n_variants, args.backbone, args.batch, args.seed)
        r = res[name]
        print(f"\n=== {name}  L={r['length']}  n={r['n_variants']} ===")
        print(f"{'radius':>7} {'rows kept':>10} {'rowwise':>9} {'pooled |.|':>11} {'cos':>7}")
        for k in [str(x) for x in RADII]:
            print(f"{k:>7} {r['rows_stored_fraction'][k]:>10.1%} "
                  f"{r['rowwise_fraction'][k]:>9.1%} "
                  f"{r['pooled_norm_recovered'][k]:>11.1%} "
                  f"{r['pooled_direction_cosine'][k]:>7.3f}")

    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
