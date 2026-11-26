"""Step 1.4 — D4: is the applied mutation recoverable from frozen embeddings?

The inverse-dynamics anchor is the most load-bearing new component in the revised
architecture. It only works if the pair (Z_wildtype, Z_mutant) actually identifies
which substitution was applied. If the latent carries no action signal there is
nothing for the anchor to learn, and that is worth knowing before it is wired into a
training loop rather than after.

Three questions, matching the declared experiments:

    mutation_recovery          can the position and the new residue be recovered?
    recovery_pooled_vs_residue does mean-pooling destroy it, as the proposal claims?
    recovery_vs_length         does recovery decay with protein length?

Position recovery is **parameter-free**: rank residues by how far their embedding
moved and ask whether the substituted one is at the top. Nothing is fitted, so the
number cannot be inflated by a model. Chance is 1/L.

Residue-identity recovery does need a classifier, so it is a multinomial logistic
regression on the difference vector, scored out of fold. Fitting it on the same data
it is scored on would report the memorisation capacity of a 1280-dimensional feature,
not a property of the embedding.

The pooled comparison is the one the architecture turns on. Section 3 of the proposal
argues that mean-pooling attenuates a single-residue change by O(1/L); this measures
it on the same mutants with the same classifier, changing only the feature.

Runs on CPU or a 4 GB GPU -- the encoder is 1.3 GiB and the heads are tiny.

    python -I scripts/d4_inverse_dynamics.py --limit 10     # quick
    python -I scripts/d4_inverse_dynamics.py                # the whole manifest
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biodreamer import tracking
from biodreamer.protein_dreamer.encoder import ProteinEncoder

MANIFEST = Path("data/processed/d1_fold_manifest.json")
OUT = Path("data/processed/d4_inverse.json")
AMINO = "ACDEFGHIKLMNPQRSTVWY"


def features(enc: ProteinEncoder, wt_seq: str, mutants: list[dict]) -> list[dict]:
    """Per-mutant: where the embedding moved, and what the truth was."""
    z_wt = enc.encode([wt_seq], batch_size=1, use_cache=True)[0]
    rows = []
    for m in mutants:
        seq = m["sequence"]
        if len(seq) != len(wt_seq):
            continue
        pos = next((i for i, (a, b) in enumerate(zip(wt_seq, seq)) if a != b), None)
        if pos is None:
            continue
        z = enc.encode([seq], batch_size=1, use_cache=True)[0]
        if z.shape != z_wt.shape:
            continue
        diff = z - z_wt
        shift = np.linalg.norm(diff, axis=1)
        order = np.argsort(shift)[::-1]
        rows.append(
            {
                "length": len(wt_seq),
                "true_pos": pos,
                "rank_of_true_pos": int(np.where(order == pos)[0][0]),
                "new_aa": seq[pos],
                # the two competing features for residue identity
                "diff_at_pos": diff[pos].astype(np.float32),
                "diff_pooled": diff.mean(axis=0).astype(np.float32),
                # how concentrated the change is, as a sanity signal
                "shift_share_at_pos": float(shift[pos] / shift.sum()) if shift.sum() else np.nan,
            }
        )
    return rows


def cv_accuracy(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> dict[str, float]:
    """Out-of-fold top-1 and top-5 accuracy of a multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    classes = np.unique(y)
    if len(classes) < 2 or len(y) < folds * 2:
        return {"top1": float("nan"), "top5": float("nan"), "n": float(len(y))}

    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(y)), folds)
    top1 = top5 = total = 0
    for i in range(folds):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit(sc.transform(X[tr]), y[tr])
        proba = clf.predict_proba(sc.transform(X[te]))
        ranked = clf.classes_[np.argsort(proba, axis=1)[:, ::-1]]
        for k, truth in enumerate(y[te]):
            top1 += int(ranked[k, 0] == truth)
            top5 += int(truth in ranked[k, :5])
            total += 1
    if not total:
        return {"top1": float("nan"), "top5": float("nan"), "n": float(len(y))}
    return {"top1": top1 / total, "top5": top5 / total, "n": float(total)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N proteins")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"missing {MANIFEST} -- run scripts/d1_build_manifest.py first")
        return 1
    entries = json.loads(MANIFEST.read_text())["proteins"][: args.limit]

    enc = ProteinEncoder("saprot")
    rows: list[dict] = []
    t0 = time.time()
    for i, e in enumerate(entries, 1):
        got = features(enc, e["wt_sequence"], e["mutants"])
        rows.extend(got)
        print(f"  [{i}/{len(entries)}] {e['wt_name'][:32]:<32} L={e['length']:>3} "
              f"{len(got):>2} pairs  ({time.time()-t0:.0f}s)", flush=True)
    if not rows:
        print("no usable pairs")
        return 1

    ranks = np.array([r["rank_of_true_pos"] for r in rows])
    lengths = np.array([r["length"] for r in rows])
    shares = np.array([r["shift_share_at_pos"] for r in rows])

    # --- position: parameter-free, nothing fitted ---
    pos_top1 = float((ranks == 0).mean())
    pos_top5 = float((ranks < 5).mean())
    chance1 = float(np.mean(1.0 / lengths))
    chance5 = float(np.mean(np.minimum(5.0, lengths) / lengths))

    # --- residue identity: per-residue feature versus mean-pooled ---
    y = np.array([r["new_aa"] for r in rows])
    at_pos = cv_accuracy(np.stack([r["diff_at_pos"] for r in rows]), y, seed=args.seed)
    pooled = cv_accuracy(np.stack([r["diff_pooled"] for r in rows]), y, seed=args.seed)

    metrics = {
        "n_pairs": float(len(rows)),
        "n_proteins": float(len(entries)),
        "position_top1": pos_top1,
        "position_top5": pos_top5,
        "position_top1_chance": chance1,
        "position_top5_chance": chance5,
        "position_top1_over_chance": pos_top1 / chance1 if chance1 else float("nan"),
        "median_rank_of_true_position": float(np.median(ranks)),
        "median_shift_share_at_position": float(np.nanmedian(shares)),
        "aa_top1_per_residue": at_pos["top1"],
        "aa_top5_per_residue": at_pos["top5"],
        "aa_top1_pooled": pooled["top1"],
        "aa_top5_pooled": pooled["top5"],
        "aa_top1_chance": 1.0 / len(np.unique(y)),
        "pooled_penalty_top1": at_pos["top1"] - pooled["top1"],
    }

    print(f"\n  POSITION (parameter-free: rank residues by how far they moved)")
    print(f"    top-1 {pos_top1:.3f}  vs chance {chance1:.3f}   "
          f"({metrics['position_top1_over_chance']:.0f}x)")
    print(f"    top-5 {pos_top5:.3f}  vs chance {chance5:.3f}")
    print(f"    median rank of the true position: {metrics['median_rank_of_true_position']:.0f}")
    print(f"    median share of total movement at the substituted residue: "
          f"{metrics['median_shift_share_at_position']:.1%}")
    print(f"\n  NEW RESIDUE (out-of-fold logistic regression, 20 classes, chance 0.050)")
    print(f"    per-residue feature  top-1 {at_pos['top1']:.3f}  top-5 {at_pos['top5']:.3f}")
    print(f"    mean-pooled feature  top-1 {pooled['top1']:.3f}  top-5 {pooled['top5']:.3f}")
    print(f"    pooling costs {metrics['pooled_penalty_top1']:+.3f} top-1")

    # --- length dependence ---
    print(f"\n  BY LENGTH")
    by_len = {}
    edges = [0, 50, 60, 70, 100, 10**6]
    for lo, hi in zip(edges, edges[1:]):
        sel = (lengths >= lo) & (lengths < hi)
        if sel.sum() < 10:
            continue
        band = f"{lo}-{hi if hi < 10**6 else 'inf'}"
        by_len[band] = {
            "n": float(sel.sum()),
            "position_top1": float((ranks[sel] == 0).mean()),
            "chance": float(np.mean(1.0 / lengths[sel])),
            "shift_share": float(np.nanmedian(shares[sel])),
        }
        metrics[f"position_top1_L_{band}"] = by_len[band]["position_top1"]
        print(f"    L {band:<9} n={sel.sum():>4}  top-1 {by_len[band]['position_top1']:.3f}  "
              f"chance {by_len[band]['chance']:.3f}  share {by_len[band]['shift_share']:.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"metrics": metrics, "by_length": by_len}, indent=2) + "\n")
    print(f"\n  {time.time()-t0:.0f}s; wrote {OUT}")

    if not args.no_track:
        for exp, keys in (
            ("mutation_recovery", ("position_top1", "position_top5", "position_top1_chance",
                                   "position_top5_chance", "position_top1_over_chance",
                                   "median_rank_of_true_position", "aa_top1_per_residue",
                                   "aa_top5_per_residue", "aa_top1_chance", "n_pairs")),
            ("recovery_pooled_vs_residue", ("aa_top1_per_residue", "aa_top1_pooled",
                                            "aa_top5_pooled", "pooled_penalty_top1",
                                            "median_shift_share_at_position")),
            ("recovery_vs_length", tuple(k for k in metrics if k.startswith("position_top1_L_"))),
        ):
            vals = {k: metrics[k] for k in keys if k in metrics and np.isfinite(metrics[k])}
            if not vals:
                continue
            with tracking.run(
                "1.4", exp,
                variant={"backbone": enc.name, "n_proteins": len(entries)},
                datasets=[str(MANIFEST)], seed=args.seed,
                notes="Position recovery is parameter-free; residue identity is scored out of fold.",
            ) as r:
                r.log_metrics(vals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
