"""Step 1.3 — D2: does the latent track stability, and at what dimension?

The revision's central claim about the latent is that **per-residue** latents carry a
single substitution's effect while **mean-pooled** ones attenuate it by O(1/L). That
claim decided the architecture. This measures it instead of arguing it, on Tsuboyama
mutants whose ddG is measured in a single consistent assay.

Three questions, matching the declared experiments:

    displacement_vs_ddg  Spearman of latent displacement against |ddG|,
                         per-residue versus mean-pooled
    dim_sweep            how far d can be cut before the signal degrades
    backbone_sweep       SaProt versus ESM-2 650M

If neither representation tracks |ddG|, the frozen-backbone strategy itself needs
revisiting before Phase 2 -- that is the go/no-go, and it is a real one.

Three displacement scalars are reported, because they are not interchangeable:

    at_pos     ||Z_mut[i*] - Z_wt[i*]||, the substituted residue only
    frobenius  ||Z_mut - Z_wt||_F, the whole per-residue matrix
    pooled     ||mean(Z_mut) - mean(Z_wt)||, what a pooled model would see

The d sweep truncates a PCA of the at-position difference vectors, fitted out of fold
so the retained dimensions are not chosen using the mutants they are scored on.

**ProSST is deliberately absent** from the backbone sweep: it publishes no licence
(DATA_REGISTER.md), which is why SaProt is the default. ESM-C needs a download and is
not run here; the two local backbones answer the pooling question, which is the one
the architecture turns on.

Runs on CPU or a 4 GB GPU. Reuses the embedding cache, so after Step 1.4 the SaProt
pass is nearly free.

    python -I scripts/d2_latent_capacity.py --limit 10
    python -I scripts/d2_latent_capacity.py --backbones saprot esm2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from biodreamer import tracking
from biodreamer.protein_dreamer.encoder import ProteinEncoder

MANIFEST = Path("data/processed/d1_fold_manifest.json")
OUT = Path("data/processed/d2_latent.json")
# see the note in d1_fold.py: a limited run writes elsewhere
PARTIAL = Path("data/processed/d2_latent.partial.json")
DIMS = (64, 128, 256, 512)


def collect(enc: ProteinEncoder, entries: list[dict]) -> list[dict]:
    """Displacement scalars and |ddG| for every mutant with a measured ddG."""
    rows = []
    for i, e in enumerate(entries, 1):
        wt_seq = e["wt_sequence"]
        z_wt = enc.encode([wt_seq], batch_size=1, use_cache=True)[0]
        mu_wt = z_wt.mean(axis=0)
        n = 0
        for m in e["mutants"]:
            if m["ddG"] is None:  # the length-probe group has no measured ddG
                continue
            seq = m["sequence"]
            if len(seq) != len(wt_seq):
                continue
            pos = next((k for k, (a, b) in enumerate(zip(wt_seq, seq)) if a != b), None)
            if pos is None:
                continue
            z = enc.encode([seq], batch_size=1, use_cache=True)[0]
            if z.shape != z_wt.shape:
                continue
            diff = z - z_wt
            rows.append(
                {
                    "wt_name": e["wt_name"],
                    "length": len(wt_seq),
                    "abs_ddG": abs(float(m["ddG"])),
                    "ddG": float(m["ddG"]),
                    "band": m["band"],
                    "at_pos": float(np.linalg.norm(diff[pos])),
                    "frobenius": float(np.linalg.norm(diff)),
                    "pooled": float(np.linalg.norm(z.mean(axis=0) - mu_wt)),
                    "diff_vec": diff[pos].astype(np.float32),
                    # the full L x d matrix, kept for a bounded subsample so the
                    # dimension sweep can project the per-residue representation
                    # rather than a single residue's vector
                    "diff_full": diff.astype(np.float32) if len(rows) < 400 else None,
                }
            )
            n += 1
        print(f"  [{i}/{len(entries)}] {e['wt_name'][:30]:<30} L={e['length']:>3} {n:>2} pairs",
              flush=True)
    return rows


def rho(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def dim_sweep(
    fulls: list[np.ndarray], target: np.ndarray, folds: int = 5, seed: int = 0
) -> dict[int, float]:
    """How much of the |ddG| signal survives projecting the latent to d dimensions.

    This is the question the adapter poses: it maps 1280 -> d, and Phase 2 onwards
    sees only the projection. So the sweep projects the **whole per-residue difference
    matrix** and takes its Frobenius norm -- the scalar that was measured to track
    |ddG| at rho ~0.45 -- rather than a single residue's vector, whose raw correlation
    is only 0.13-0.27 and which therefore cannot show a projection degrading anything.

    The basis is the top-d right singular vectors of the training folds' residue
    differences, so no scoring fold contributes to the directions that score it.
    """
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(target)), folds)
    dims = [d for d in DIMS if d < fulls[0].shape[1]]
    if not dims:
        return {}

    # One SVD per fold, sliced for every d. Recomputing it per (fold, d) was 4x the
    # work for the same basis -- the top-64 directions ARE the first 64 columns of
    # the top-512 basis.
    pred = {d: np.full(len(target), np.nan) for d in dims}
    for i in range(folds):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        X = np.concatenate([fulls[j] for j in tr], axis=0)
        mu = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
        for j in te:
            proj = fulls[j] @ vt[: max(dims)].T
            for d in dims:
                pred[d][j] = float(np.linalg.norm(proj[:, :d]))
    return {d: rho(pred[d], target) for d in dims}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=["saprot", "esm2"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"missing {MANIFEST} -- run scripts/d1_build_manifest.py first")
        return 1
    entries = json.loads(MANIFEST.read_text())["proteins"][: args.limit]

    results = {}
    for backbone in args.backbones:
        print(f"\n=== {backbone} ===")
        t0 = time.time()
        enc = ProteinEncoder(backbone)
        rows = collect(enc, entries)
        if len(rows) < 20:
            print(f"  only {len(rows)} pairs; skipping")
            continue

        ad = np.array([r["abs_ddG"] for r in rows])
        lengths = np.array([r["length"] for r in rows])
        m = {
            "backbone": enc.name,
            "n_pairs": float(len(rows)),
            "rho_at_pos_vs_absddG": rho(np.array([r["at_pos"] for r in rows]), ad),
            "rho_frobenius_vs_absddG": rho(np.array([r["frobenius"] for r in rows]), ad),
            "rho_pooled_vs_absddG": rho(np.array([r["pooled"] for r in rows]), ad),
            "mean_length": float(lengths.mean()),
        }
        # The honest comparison is the WHOLE per-residue representation against
        # pooled. Comparing the single substituted residue against pooled made
        # pooling look better, which says more about that scalar than about pooling.
        m["per_residue_advantage_rho"] = (
            m["rho_frobenius_vs_absddG"] - m["rho_pooled_vs_absddG"]
        )

        # the O(1/L) claim: pooled displacement should shrink with length
        m["rho_pooled_vs_length"] = rho(np.array([r["pooled"] for r in rows]), lengths.astype(float))
        m["rho_at_pos_vs_length"] = rho(np.array([r["at_pos"] for r in rows]), lengths.astype(float))

        sub = [(k, r) for k, r in enumerate(rows) if r["diff_full"] is not None]
        if len(sub) >= 50:
            sweep = dim_sweep(
                [r["diff_full"] for _, r in sub],
                ad[[k for k, _ in sub]],
                seed=args.seed,
            )
        else:
            sweep = {}
        for d, v in sweep.items():
            m[f"rho_d{d}"] = v

        print(f"\n  |ddG| versus displacement (Spearman, n={len(rows)}):")
        print(f"    at the substituted residue  {m['rho_at_pos_vs_absddG']:+.3f}")
        print(f"    whole per-residue matrix    {m['rho_frobenius_vs_absddG']:+.3f}")
        print(f"    mean-pooled                 {m['rho_pooled_vs_absddG']:+.3f}")
        print(f"    per-residue advantage       {m['per_residue_advantage_rho']:+.3f}"
          f"   (whole matrix minus pooled)")
        print(f"  displacement versus length: pooled {m['rho_pooled_vs_length']:+.3f}, "
              f"at-position {m['rho_at_pos_vs_length']:+.3f}")
        if sweep:
            print(f"  dimension sweep ({len(sub)} mutants, projected per-residue "
                  f"Frobenius norm, out-of-fold basis):")
            for d, v in sweep.items():
                print(f"    d={d:<5} rho {v:+.3f}")
        results[backbone] = {"metrics": m, "dim_sweep": sweep}
        print(f"  {time.time()-t0:.0f}s")

        if not args.no_track:
            with tracking.run(
                "1.3", "displacement_vs_ddg",
                variant={"backbone": enc.name},
                datasets=[str(MANIFEST)], seed=args.seed,
                notes="Per-residue versus mean-pooled displacement against measured |ddG|.",
            ) as r:
                r.log_metrics({k: v for k, v in m.items()
                               if isinstance(v, float) and np.isfinite(v)})
            if sweep:
                with tracking.run(
                    "1.3", "dim_sweep", variant={"backbone": enc.name},
                    datasets=[str(MANIFEST)], seed=args.seed,
                    notes="Out-of-fold PCA truncation; the basis never sees its scoring fold.",
                ) as r:
                    r.log_metrics({f"rho_d{d}": v for d, v in sweep.items() if np.isfinite(v)})

    if len(results) > 1 and not args.no_track:
        best = max(results, key=lambda b: results[b]["metrics"]["rho_at_pos_vs_absddG"])
        with tracking.run(
            "1.3", "backbone_sweep",
            variant={"backbones": ",".join(results)},
            datasets=[str(MANIFEST)], seed=args.seed,
            notes=f"ProSST excluded: no licence. Best on at-position rho: {best}.",
        ) as r:
            r.log_metrics({
                f"rho_at_pos_{b}": results[b]["metrics"]["rho_at_pos_vs_absddG"]
                for b in results
            })

    if results:
        out_path = PARTIAL if args.limit else OUT
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {out_path}")
        if out_path is PARTIAL:
            print(f"  (--limit given -- {OUT} left untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
