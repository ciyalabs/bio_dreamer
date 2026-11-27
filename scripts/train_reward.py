"""Step 2.1 — train the reward head's stability target on cached latents.

This trains the half that can be trained today. Steps 1.3 and 1.4 already encoded all
1,080 Tsuboyama mutants with ESM-2, so every latent this needs is a cache hit and the
frozen 650M backbone never runs. The fitness half needs ProteinGym latents, which are
287,000 sequences of GPU time and belong in the next rental.

**Splits are by protein, never by mutant.** Twenty mutants of one domain share 95% of
their residues; a random split over mutants would put near-duplicates on both sides and
report a number that is mostly memorised protein identity. Held-out proteins are the
only split that measures what the head will actually be asked to do.

**Per-protein performance is reported as a distribution, not a mean.** The plan asks
for this explicitly, and for good reason: a mean over 54 domains can be carried by a
handful of easy ones while the head is useless on the rest.

What this run establishes is the **frozen-encoder regression baseline** that H1's
`spearman_mean_vs_frozen_regression > 0` is measured against — there is no world model
yet, so this *is* that baseline. Beating it later is what shows the world-model
training added something.

    python -I scripts/train_reward.py
    python -I scripts/train_reward.py --backbone saprot --epochs 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from biodreamer import tracking
from biodreamer.protein_dreamer.encoder import ProteinEncoder
from biodreamer.reward import RewardConfig, RewardHead, weighted_smooth_l1

MANIFEST = Path("data/processed/d1_fold_manifest.json")
OUT = Path("data/processed/reward_training.json")


def load_latents(backbone: str, limit: int | None) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """Cached per-residue latents, ddG targets and a protein index per mutant."""
    entries = json.loads(MANIFEST.read_text())["proteins"][:limit]
    enc = ProteinEncoder(backbone)
    lat, ddg, pid, names = [], [], [], []
    for k, e in enumerate(entries):
        muts = [m for m in e["mutants"] if m["ddG"] is not None]
        if not muts:
            continue
        names.append(e["wt_name"])
        seqs = [m["sequence"] for m in muts]
        zs = enc.encode(seqs, batch_size=4, use_cache=True)
        for z, m in zip(zs, muts):
            lat.append(z.astype(np.float32))
            ddg.append(float(m["ddG"]))
            pid.append(len(names) - 1)
        print(f"  [{k+1}/{len(entries)}] {e['wt_name'][:32]:<32} {len(muts):>2} mutants", flush=True)
    return lat, np.array(ddg, np.float32), np.array(pid), names


def pad_batch(lat: list[np.ndarray], idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(lat[i].shape[0] for i in idx)
    d = lat[idx[0]].shape[1]
    z = torch.zeros(len(idx), L, d)
    mask = torch.zeros(len(idx), L, dtype=torch.bool)
    for r, i in enumerate(idx):
        n = lat[i].shape[0]
        z[r, :n] = torch.from_numpy(lat[i])
        mask[r, :n] = True
    return z, mask


def per_protein_spearman(pred: np.ndarray, truth: np.ndarray, pid: np.ndarray,
                         names: list[str]) -> dict[str, float]:
    out = {}
    for p in np.unique(pid):
        m = pid == p
        if m.sum() >= 5 and np.ptp(truth[m]) > 0:
            out[names[p]] = float(spearmanr(pred[m], truth[m]).statistic)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="esm2", help="esm2 is the default after step 1.3")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"missing {MANIFEST}")
        return 1
    torch.manual_seed(args.seed)

    print(f"loading cached {args.backbone} latents ...")
    lat, ddg, pid, names = load_latents(args.backbone, args.limit)
    print(f"\n{len(lat):,} mutants over {len(names)} proteins, "
          f"ddG {ddg.min():.2f} to {ddg.max():.2f}")

    # cross-validate over PROTEINS, so no domain appears in both halves
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(names))
    folds = np.array_split(order, args.folds)

    oof = np.full(len(ddg), np.nan, dtype=np.float32)
    t0 = time.time()
    for f, held in enumerate(folds):
        tr = np.where(~np.isin(pid, held))[0]
        te = np.where(np.isin(pid, held))[0]
        head = RewardHead(RewardConfig(backbone_dim=lat[0].shape[1]))
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        y = torch.from_numpy(ddg)

        head.train()
        for _ in range(args.epochs):
            perm = rng.permutation(len(tr))
            for s in range(0, len(tr), args.batch_size):
                b = tr[perm[s : s + args.batch_size]]
                z, mask = pad_batch(lat, b)
                loss = weighted_smooth_l1(head(z, mask)["stability"], y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()

        head.eval()
        with torch.no_grad():
            for s in range(0, len(te), 64):
                b = te[s : s + 64]
                z, mask = pad_batch(lat, b)
                oof[b] = head(z, mask)["stability"].numpy()
        rho = spearmanr(oof[te], ddg[te]).statistic
        print(f"  fold {f+1}/{args.folds}: {len(held)} held-out proteins, "
              f"{len(te)} mutants, rho {rho:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

    overall = float(spearmanr(oof, ddg).statistic)
    per_prot = per_protein_spearman(oof, ddg, pid, names)
    vals = np.array(list(per_prot.values()))
    metrics = {
        "n_mutants": float(len(ddg)),
        "n_proteins": float(len(names)),
        "spearman_pooled": overall,
        "spearman_per_protein_mean": float(vals.mean()),
        "spearman_per_protein_median": float(np.median(vals)),
        "spearman_per_protein_p10": float(np.percentile(vals, 10)),
        "spearman_per_protein_p90": float(np.percentile(vals, 90)),
        "frac_proteins_positive": float((vals > 0).mean()),
        "frac_proteins_above_0.3": float((vals > 0.3).mean()),
        "n_trainable_params": float(RewardHead(RewardConfig(backbone_dim=lat[0].shape[1])).n_trainable()),
        "epochs": float(args.epochs),
        "folds": float(args.folds),
    }

    print(f"\n  pooled Spearman (out of fold)   {overall:+.3f}")
    print(f"  per protein: median {metrics['spearman_per_protein_median']:+.3f}  "
          f"mean {metrics['spearman_per_protein_mean']:+.3f}  "
          f"p10 {metrics['spearman_per_protein_p10']:+.3f}  "
          f"p90 {metrics['spearman_per_protein_p90']:+.3f}")
    print(f"  {metrics['frac_proteins_positive']:.0%} of proteins positive, "
          f"{metrics['frac_proteins_above_0.3']:.0%} above 0.3")
    worst = sorted(per_prot.items(), key=lambda kv: kv[1])[:3]
    print(f"  worst: " + ", ".join(f"{n[:22]} {v:+.2f}" for n, v in worst))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(OUT.read_text()) if OUT.exists() else {}
    payload[args.backbone] = {"metrics": metrics, "per_protein": per_prot}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  {time.time()-t0:.0f}s; wrote {OUT}")

    if not args.no_track:
        with tracking.run(
            "2.1", "reward_training",
            variant={"backbone": args.backbone, "target": "stability"},
            config=vars(args), seed=args.seed, datasets=[str(MANIFEST)],
            notes=("Cross-validated over PROTEINS, not mutants: 20 mutants of one domain "
                   "share 95% of their residues. No world model exists yet, so this is the "
                   "frozen-encoder regression baseline H1 compares against."),
        ) as r:
            r.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
            r.log_artifact(str(OUT), kind="table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
