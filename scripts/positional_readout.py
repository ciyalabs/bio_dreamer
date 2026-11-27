"""Step 2.2 follow-up — does reading the mutated positions beat mean-pooling?

H1 failed with the trained head at +0.409 against +0.390 for
`RidgeCV(pooled_embedding ++ mutation_count)`. The diagnosis was the representation, so
this compares readouts of the SAME frozen SaProt latents under the SAME head, recipe and
test subsample. Only the input vector changes.

| readout | what it can express |
|---|---|
| `pooled` | the baseline; a mean over all residues, blind to which position changed |
| `mut_mean` | mean latent at the variant's own mutated positions |
| `displacement` | mean of (variant − wild type) at those positions |
| `pooled+displacement` | both, concatenated |

`displacement` is the one the diagnosis predicts should win: it is the only readout that
sees the change rather than the state, and Step 1.3 found displacement norm correlating
with |ddG| at +0.459 unsupervised.

Everything is paired -- same assays, same capped test variants, same 300-epoch recipe,
same seed -- so differences are attributable to the readout and nothing else.

    python -I scripts/positional_readout.py
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
from biodreamer.reward import RewardConfig, RewardHead
from biodreamer.reward.data import assays_for, banned_assays
from biodreamer.reward.losses import pairwise_ranking_loss, standardise_per_assay

CACHE = Path("data/processed/positional_cache")
BACKBONE = "westlake-repl_SaProt_650M_AF2"
OUT = Path("data/processed/positional_readout.json")
READOUTS = ("pooled", "mut_mean", "displacement", "pooled+displacement")


def features(z, readout: str) -> np.ndarray:
    """One vector per variant, from the cached pooled / per-position latents."""
    pooled = z["pooled"].astype(np.float32)
    if readout == "pooled":
        return pooled
    off, lat = z["offsets"], z["pos_latents"].astype(np.float32)
    mut_mean = np.stack([lat[off[i] : off[i + 1]].mean(axis=0)
                         for i in range(len(off) - 1)])
    if readout == "mut_mean":
        return mut_mean
    wt, pos = z["wt_latent"].astype(np.float32), z["positions"]
    disp = np.stack([(lat[off[i] : off[i + 1]] - wt[pos[off[i] : off[i + 1]] - 1]).mean(axis=0)
                     for i in range(len(off) - 1)])
    if readout == "displacement":
        return disp
    if readout == "pooled+displacement":
        return np.hstack([pooled, disp])
    raise ValueError(f"unknown readout {readout!r}")


def run(readout: str, names: list[str], data: dict, epochs: int, lr: float,
        seed: int) -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Xtr, ytr, atr, ctr, otr = [], [], [], [], []
    for k, n in enumerate(names):
        d = data[n]
        m = d["split"] == 0
        Xtr.append(d[readout][m]); ytr.append(d["score"][m])
        atr.append(np.full(int(m.sum()), k)); ctr.append(d["censored"][m])
        otr.append(d["order"][m])
    X = torch.from_numpy(np.concatenate(Xtr))
    assay = torch.from_numpy(np.concatenate(atr))
    y = standardise_per_assay(
        torch.from_numpy(np.concatenate(ytr).astype(np.float32)), assay)
    cens = torch.from_numpy(np.concatenate(ctr))
    order_t = torch.from_numpy(np.concatenate(otr).astype(np.int64))

    head = RewardHead(RewardConfig(backbone_dim=X.shape[1], pool_first=True,
                                   depth_aware=True, depth_transform="linear"))
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    for _ in range(epochs):
        loss = pairwise_ranking_loss(
            head(X, order=order_t)["fitness"], y, assay, censored=cens,
            max_pairs=4096, generator=gen)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()

    per = {}
    for n in names:
        d = data[n]
        m = d["split"] == 1
        xb = torch.from_numpy(d[readout][m])
        ob = torch.from_numpy(d["order"][m].astype(np.int64))
        with torch.no_grad():
            p = head(xb, order=ob)["fitness"].numpy()
        per[n] = float(spearmanr(p, d["score"][m]).statistic)
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="benchmark", choices=["planning", "benchmark"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    names = [n for n in assays_for(args.instance)
             if (CACHE / f"{BACKBONE}__{n}.npz").exists()]
    missing = [n for n in assays_for(args.instance) if n not in names]
    if missing:
        print(f"NOT ENCODED, excluded from this run: {missing}\n")
    if not names:
        raise SystemExit(f"no positional cache under {CACHE}")

    print(f"building features for {len(names)} assays ...")
    data = {}
    for n in names:
        z = np.load(CACHE / f"{BACKBONE}__{n}.npz")
        d = {"score": z["score"], "order": z["order"],
             "censored": z["censored"], "split": z["split"]}
        for r in READOUTS:
            d[r] = features(z, r)
        z.close()
        data[n] = d
        print(f"  {n:32s} train {int((d['split']==0).sum()):>5,} "
              f"test {int((d['split']==1).sum()):>7,}  dims "
              + " ".join(f"{r}={d[r].shape[1]}" for r in READOUTS), flush=True)

    banned = banned_assays()
    results, t0 = {}, time.time()
    for r in READOUTS:
        results[r] = run(r, names, data, args.epochs, args.lr, args.seed)
        print(f"\n{r}: mean {np.mean(list(results[r].values())):+.3f}", flush=True)

    print(f"\n{'assay':32s} " + " ".join(f"{r:>20s}" for r in READOUTS))
    for n in names:
        print(f"{n:32s} " + " ".join(f"{results[r][n]:>+20.3f}" for r in READOUTS)
              + ("  [BANNED]" if n in banned else ""))
    means = {r: float(np.mean([results[r][n] for n in names])) for r in READOUTS}
    print(f"\n{'MEAN':32s} " + " ".join(f"{means[r]:>+20.3f}" for r in READOUTS))
    best = max(means, key=means.get)
    print(f"\nbest readout: {best} at {means[best]:+.3f}  "
          f"(pooled baseline {means['pooled']:+.3f}, "
          f"delta {means[best]-means['pooled']:+.3f})")
    print(f"H1 needs spearman_mean >= 0.55 -- {'CLEARED' if means[best] >= 0.55 else 'still short'}")
    print(f"{time.time()-t0:.0f}s")

    OUT.write_text(json.dumps(
        {"instance": args.instance, "assays": names, "means": means,
         "per_assay": results, "epochs": args.epochs,
         "note": "paired comparison: same head, recipe, seed and capped test subsample; "
                 "only the input vector differs"}, indent=2) + "\n")
    print(f"wrote {OUT}")

    if not args.no_track:
        with tracking.run(
            "2.2", "positional_readout",
            variant={"instance": args.instance, "best_readout": best},
            config=vars(args), seed=args.seed, datasets=[str(CACHE)],
            notes=("Tests whether mean-pooling is H1's binding constraint. Paired over "
                   "readouts of the same frozen latents; `displacement` is the only one "
                   "that sees the change rather than the state."),
        ) as t:
            t.log_metrics({f"mean_{r}": v for r, v in means.items()})
            t.log_artifact(str(OUT), kind="table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
