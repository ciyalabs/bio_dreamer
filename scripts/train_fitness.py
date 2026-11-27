"""Step 2.1 — train the reward head's fitness target on the pooled ProteinGym cache.

The stability half was trained on 2026-11-27 from the Tsuboyama latents Steps 1.3/1.4
had already cached. This is the other half, and it needed a GPU rental first: 946,365
pooled SaProt vectors over 11 assays, encoded by `scripts/encode_corpus.py`.

**Pooled input is legitimate here because it was measured, not assumed.** Step 2.1's
first run compared `pool_first=False` against `pool_first=True` and got +0.337 against
+0.329 pooled, per-protein median +0.753 against +0.771 — a wash. That is what licenses
a cache 512x smaller than per-residue latents, and the whole of Phase 2 rests on it.

**Per-assay standardisation and a ranking loss, never raw regression across assays.**
The plan is explicit and the reason is arithmetic: these assays report growth rates,
fluorescence and enrichment in incomparable units, so a shared regression head would
spend its capacity learning which assay a variant came from.

**Censoring: rank, and drop pairs where both members are censored.** Decided by
measurement in `diagnostics/reward_head.md`; the mask comes from
`data/processed/censoring_audit.json` so the loss and the audit cannot drift apart.

**The recipe below is fixed and was not tuned on the order>=3 test set.** That set is
H1's test set (Step 2.2) and the threshold against it is pre-registered, so tuning
against it here would quietly spend the pre-registration. Epochs, learning rate and
capacity are the stability run's values, carried over unchanged.

    python -I scripts/train_fitness.py --instance planning
    python -I scripts/train_fitness.py --instance benchmark
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from biodreamer import tracking
from biodreamer.protein_dreamer.cache import sequence_hash
from biodreamer.reward import RewardConfig, RewardHead
from biodreamer.reward.data import (
    assays_for,
    banned_assays,
    detection_limits,
    h1_split,
    load_assay,
)
from biodreamer.reward.losses import (
    pairwise_ranking_loss,
    ranking_pairs,
    standardise_per_assay,
)

CACHE = Path("data/processed/pooled_cache")
BACKBONE = "westlake-repl/SaProt_650M_AF2"
OUT = Path("data/processed/fitness_training.json")


def cache_path(assay: str) -> Path:
    return CACHE / f"{BACKBONE.replace('/', '_')}__{assay}.npz"


def assay_rows(assay: str, limits: dict[str, float], seed: int):
    """Cache row indices for one assay's H1 train and test variants.

    Returns the open npz too, so the caller decides when to read the big `emb` array.
    A miss is raised rather than skipped: the cache is keyed by sequence hash, so a
    mismatch between encoder and loader is silent and would surface later as an
    inexplicably weak H1 number.
    """
    a = load_assay(assay, limits)
    tr, te = h1_split(a.order, seed=seed)
    z = np.load(cache_path(assay))
    index = {k: i for i, k in enumerate(z["keys"].tolist())}

    def rows(idx):
        out = np.empty(len(idx), np.int64)
        for n, i in enumerate(idx):
            h = sequence_hash(a.sequence[i])
            if h not in index:
                raise KeyError(
                    f"{assay}: variant {i} not in the pooled cache. The cache holds the "
                    f"seed=0 H1 split; re-encoding is the only fix and it costs a GPU."
                )
            out[n] = index[h]
        return out

    return a, tr, te, rows(tr), rows(te), z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="planning", choices=["planning", "benchmark"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-pairs", type=int, default=4096)
    ap.add_argument("--depth-aware", action="store_true",
                    help="subtract a learned non-negative multiple of mutation count; "
                         "the weight is fitted on the TRAINING split like everything else")
    ap.add_argument("--depth-transform", default="linear", choices=["linear", "log"])
    ap.add_argument("--depth-beta-init", type=float, default=None,
                    help="initial VALUE of softplus(depth_beta); the default init is "
                         "beta=0, i.e. 0.693. Exposed because the fitted beta moves "
                         "only ~9%% from its init, so the init is doing most of the work "
                         "and its influence has to be measured rather than assumed.")
    ap.add_argument("--equal-order-frac", type=float, default=0.0,
                    help="share of each step's pairs reserved for EQUAL mutation order; "
                         "0.0 is the uniform sampler that tied with counting mutations")
    ap.add_argument("--seed", type=int, default=0,
                    help="split seed; LOCKED to 0 -- the pooled cache holds that split")
    ap.add_argument("--init-seed", type=int, default=None,
                    help="head init only, to measure run-to-run spread without "
                         "touching the split the cache was encoded for")
    ap.add_argument("--eval-batch", type=int, default=8192)
    ap.add_argument("--control-seeds", type=int, default=10)
    ap.add_argument("--control-cap", type=int, default=30000)
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="short smoke run; writes *.dryrun.json and never the real file")
    args = ap.parse_args()
    out = OUT.with_suffix(".dryrun.json") if args.dry_run else OUT
    if args.dry_run:
        args.epochs, args.no_track = min(args.epochs, 3), True
        print(f"DRY RUN -- {args.epochs} epochs, results go to {out}\n")

    names = assays_for(args.instance)
    banned = banned_assays()
    if args.instance == "planning" and set(names) & banned:
        raise RuntimeError("planning instance would train on held-out assays")
    limits = detection_limits()
    init_seed = args.seed if args.init_seed is None else args.init_seed
    torch.manual_seed(init_seed)
    gen = torch.Generator().manual_seed(init_seed)

    print(f"instance {args.instance}: {len(names)} assays "
          f"({len(set(names) & banned)} of them on the banned list)")

    # ---- pass 1: the training rows only. Small enough to hold in full. ----
    Xtr, ytr, atr, ctr, otr, plan = [], [], [], [], [], []
    for k, n in enumerate(names):
        a, tr, te, rtr, rte, z = assay_rows(n, limits, args.seed)
        Xtr.append(z["emb"][rtr].astype(np.float32))
        ytr.append(a.score[tr])
        atr.append(np.full(len(tr), k))
        ctr.append(a.censored[tr])
        otr.append(a.order[tr])
        plan.append((n, k, rte, te))
        print(f"  {n:34s} train {len(tr):>5,}  test {len(te):>7,}  "
              f"censored(train) {int(a.censored[tr].sum()):>4,}"
              f"{'   [BANNED]' if n in banned else ''}", flush=True)
        z.close()

    X = torch.from_numpy(np.concatenate(Xtr))
    y = standardise_per_assay(
        torch.from_numpy(np.concatenate(ytr).astype(np.float32)),
        torch.from_numpy(np.concatenate(atr)),
    )
    assay = torch.from_numpy(np.concatenate(atr))
    cens = torch.from_numpy(np.concatenate(ctr))
    order_t = torch.from_numpy(np.concatenate(otr).astype(np.int64))
    print(f"\ntraining rows {len(X):,}  censored {int(cens.sum()):,}  dim {X.shape[1]}")
    oc = np.bincount(order_t.numpy(), minlength=3)[1:3]
    print(f"  order composition: {oc[0]:,} singles, {oc[1]:,} doubles")

    # How many pairs each sampler actually offers. With equal_order_frac=1.0 an empty
    # set means no gradient and silent non-training, so it is checked, not assumed.
    n_any = ranking_pairs(y, assay, cens, None).__getitem__(0).numel()
    n_eq = ranking_pairs(y, assay, cens, None, order=order_t,
                         equal_order_frac=1.0)[0].numel()
    print(f"  comparable pairs: {n_any:,} total, {n_eq:,} equal-order "
          f"({n_eq / max(n_any, 1):.1%})")
    if args.equal_order_frac > 0 and n_eq == 0:
        raise RuntimeError("stratified sampling requested but no equal-order pair exists")
    print(f"  equal_order_frac = {args.equal_order_frac}")

    cfg = RewardConfig(backbone_dim=X.shape[1], pool_first=True,
                       depth_aware=args.depth_aware,
                       depth_transform=args.depth_transform)
    head = RewardHead(cfg)
    if args.depth_beta_init is not None:
        if not args.depth_aware:
            raise RuntimeError("--depth-beta-init requires --depth-aware")
        with torch.no_grad():  # inverse softplus
            v = torch.tensor(float(args.depth_beta_init))
            head.depth_beta.fill_(float(torch.log(torch.expm1(v))))
    # The untrained control needs MANY seeds. One random head is not a control: on
    # PHOT_CHLRE, 20 random inits gave mean -0.031 with sd 0.248 and a range of -0.438
    # to +0.472, so a single draw can look like a strong result or a strong anti-result
    # purely by chance. Reported as mean |rho| with its spread.
    controls = []
    for cs in range(args.control_seeds):
        torch.manual_seed(10_000 + cs)
        m = RewardHead(cfg)
        m.eval()
        controls.append(m)
    torch.manual_seed(init_seed)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    t0, curve = time.time(), []
    head.train()
    for e in range(args.epochs):
        loss = pairwise_ranking_loss(
            head(X, order=order_t)["fitness"], y, assay, censored=cens,
            max_pairs=args.max_pairs, generator=gen,
            order=order_t, equal_order_frac=args.equal_order_frac,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 25 == 0 or e == args.epochs - 1:
            beta = float(F.softplus(head.depth_beta).detach()) if args.depth_aware else 0.0
            curve.append({"epoch": e, "loss": float(loss.detach()), "depth_beta": beta})
            print(f"  epoch {e:>4}  ranking loss {float(loss.detach()):.4f}"
                  + (f"  depth_beta {beta:.4f}" if args.depth_aware else ""), flush=True)
    print(f"  trained in {time.time()-t0:.0f}s")

    # ---- pass 2: per-assay Spearman on the unseen order>=3 variants ----
    # Both models and the confound check share one pass, so each 1.18 GB file is read
    # once. The confound check is mutation ORDER against fitness: a random readout of a
    # pooled embedding scored +0.517 on PHOT_CHLRE in the smoke run, which is what a
    # latent whose magnitude tracks "how many mutations" will do. It is NOT H1's
    # additive null (that fits per-site effects and belongs to Step 2.2); it is the
    # cheapest guard against reading mutation count as learned biology.
    def score_all() -> tuple[dict, dict, dict, dict, dict, dict]:
        head.eval()
        tr_, ct_, cs_, nm_, sq_, wo_ = {}, {}, {}, {}, {}, {}
        for n, k, rte, te in plan:
            a = load_assay(n, limits)
            z = np.load(cache_path(n))
            emb = z["emb"]
            pt = np.empty(len(rte), np.float32)
            od_t = torch.from_numpy(a.order[te].astype(np.int64))
            ps = np.empty(len(rte), np.float32)  # before the depth penalty
            with torch.no_grad():
                for s in range(0, len(rte), args.eval_batch):
                    b = rte[s : s + args.eval_batch]
                    xb = torch.from_numpy(emb[b].astype(np.float32))
                    ob = od_t[s : s + len(b)]
                    o = head(xb, order=ob)
                    pt[s : s + len(b)] = o["fitness"].numpy()
                    ps[s : s + len(b)] = o["fitness_sequence_only"].numpy()
            truth = a.score[te]
            tr_[n] = float(spearmanr(pt, truth).statistic)
            nm_[n] = float(spearmanr(a.order[te], truth).statistic)
            sq_[n] = float(spearmanr(ps, truth).statistic)
            # rho at FIXED depth: the measurement that showed the head and mutation
            # count are complementary rather than competing. Depths with <200 variants
            # are skipped because their Spearman is mostly sampling noise.
            od = a.order[te]
            w = [float(spearmanr(ps[od == o], truth[od == o]).statistic)
                 for o in np.unique(od)
                 if (od == o).sum() >= 200 and np.ptp(truth[od == o]) > 0]
            wo_[n] = float(np.mean(w)) if w else float("nan")

            # control on a fixed subsample: 10 heads x the full set would triple the
            # run for a number that is a chance band, not a result.
            sub = np.sort(np.random.default_rng(args.seed).choice(
                len(rte), size=min(args.control_cap, len(rte)), replace=False))
            xc = torch.from_numpy(emb[rte[sub]].astype(np.float32))
            z.close()
            rs = []
            with torch.no_grad():
                oc = od_t[sub]
                for m in controls:
                    rs.append(float(spearmanr(m(xc, order=oc)["fitness"].numpy(),
                                              truth[sub]).statistic))
            rs = np.array(rs)
            ct_[n], cs_[n] = float(np.abs(rs).mean()), float(rs.std())
            print(f"  {n:34s} {tr_[n]:+.3f}  seq-only {sq_[n]:+.3f}  "
                  f"within-order {wo_[n]:+.3f}  chance {ct_[n]:.3f}+-{cs_[n]:.3f}  "
                  f"order-only {nm_[n]:+.3f}"
                  f"{'  [BANNED]' if n in banned else ''}", flush=True)
        return tr_, ct_, cs_, nm_, sq_, wo_

    print("\nper-assay Spearman on unseen order>=3 variants:")
    trained, control, control_sd, order_only, seq_only, within_order = score_all()
    vals = np.array([trained[n] for n in names])

    cvals = np.array([control[n] for n in names])
    ovals = np.array([order_only[n] for n in names])
    metrics = {
        "n_assays": float(len(names)),
        "n_train_rows": float(len(X)),
        "n_test_variants": float(sum(len(t) for _, _, _, t in plan)),
        "spearman_mean": float(vals.mean()),
        "spearman_median": float(np.median(vals)),
        "spearman_p10": float(np.percentile(vals, 10)),
        "spearman_p90": float(np.percentile(vals, 90)),
        "spearman_min": float(vals.min()),
        "frac_assays_positive": float((vals > 0).mean()),
        "chance_spearman_mean_abs": float(cvals.mean()),
        "chance_spearman_max_abs": float(cvals.max()),
        "n_assays_above_chance": float((np.abs(vals) > cvals).sum()),
        "order_only_spearman_mean_abs": float(np.abs(ovals).mean()),
        "order_only_spearman_max_abs": float(np.abs(ovals).max()),
        "n_assays_beating_order_only": float((np.abs(vals) > np.abs(ovals)).sum()),
        "epochs": float(args.epochs),
        "depth_aware": float(args.depth_aware),
        "depth_beta": (float(F.softplus(head.depth_beta).detach())
                       if args.depth_aware else 0.0),
        "depth_beta_init": (0.6931 if args.depth_beta_init is None
                            else float(args.depth_beta_init)),
        "spearman_seq_only_mean": float(np.mean([seq_only[n] for n in names])),
        "within_order_mean": float(np.nanmean([within_order[n] for n in names])),
        "init_seed": float(init_seed),
        "equal_order_frac": float(args.equal_order_frac),
        "n_pairs_total": float(n_any),
        "n_pairs_equal_order": float(n_eq),
        "final_loss": curve[-1]["loss"],
    }
    print(f"\n  mean {metrics['spearman_mean']:+.3f}   median "
          f"{metrics['spearman_median']:+.3f}   min {metrics['spearman_min']:+.3f}   "
          f"{metrics['frac_assays_positive']:.0%} positive")
    print(f"  chance band ({args.control_seeds} random heads)  mean |rho| "
          f"{metrics['chance_spearman_mean_abs']:.3f}  max {metrics['chance_spearman_max_abs']:.3f}")
    print(f"  above chance on {int(metrics['n_assays_above_chance'])}/{len(names)} assays")
    print(f"  order-only control mean |rho| {metrics['order_only_spearman_mean_abs']:.3f}"
          f"  max |rho| {metrics['order_only_spearman_max_abs']:.3f}")
    print(f"  beats order-only on {int(metrics['n_assays_beating_order_only'])}"
          f"/{len(names)} assays")
    print(f"  sequence-only mean {metrics['spearman_seq_only_mean']:+.3f}   "
          f"within-order mean {metrics['within_order_mean']:+.3f}")
    if args.depth_aware:
        print(f"  learned depth_beta {metrics['depth_beta']:.4f} "
              f"({args.depth_transform}, fitted on orders 1-2 only)")

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(out.read_text()) if out.exists() else {}
    tag = args.instance
    if args.depth_aware:
        tag += f"_depth{args.depth_transform}"
        if args.depth_beta_init is not None:
            tag += f"_b{args.depth_beta_init:g}"
    if args.equal_order_frac > 0:
        tag += f"_eq{args.equal_order_frac:g}"
    if args.init_seed is not None:
        tag += f"_init{init_seed}"
    payload[tag] = {
        "metrics": metrics, "per_assay": trained,
        "per_assay_chance_mean_abs": control, "per_assay_chance_sd": control_sd,
        "per_assay_order_only": order_only,
        "per_assay_seq_only": seq_only, "per_assay_within_order": within_order,
        "curve": curve,
        "backbone": BACKBONE, "assays": names,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {out}")

    if not args.no_track:
        with tracking.run(
            "2.1", "fitness_training",
            variant={"instance": args.instance, "target": "fitness",
                     "backbone": BACKBONE,
                     "equal_order_frac": args.equal_order_frac,
                     "depth_aware": args.depth_aware,
                     "depth_transform": args.depth_transform},
            config=vars(args), seed=args.seed, datasets=[str(CACHE)],
            notes=("Per-assay standardisation + RankNet, censored pairs dropped where "
                   "BOTH members sit at the detection limit. Trained on <=300 order<=2 "
                   "per assay, scored on unseen order>=3 -- the H1 protocol, but this is "
                   "the 2.1 training deliverable, not the H1 verdict: no additive null "
                   "and no literature baseline is computed here. Recipe fixed in advance "
                   "and not tuned on the test set."),
        ) as r:
            r.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
            r.log_artifact(str(out), kind="table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
