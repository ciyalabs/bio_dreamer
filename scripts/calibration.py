"""Step 2.3 — uncertainty calibration, and the shared-trunk question the plan singles out.

The policy consumes uncertainty directly, so an uncalibrated epistemic estimate is worse
than none: it tells the planner to explore where the model happens to be noisy rather than
where it is ignorant. This step measures three things the plan asks for.

**1. Does predicted uncertainty rank error?** That is the only property the policy needs.
A model whose uncertainty is well-scaled but uninformative about *which* predictions are
wrong is useless for exploration; one whose scale is off by a constant but which orders
errors correctly is still usable. So the headline number here is the rank correlation
between predicted sd and realised error, not a calibration curve's prettiness.

**2. Risk-coverage.** Keep the most confident x% of predictions and measure accuracy on
them. A useful uncertainty makes accuracy rise as coverage falls. This is also the
operationally honest framing, because a planner *does* get to decline to act.

**3. Shared-trunk versus independent-trunk ensembles.** The plan commits to quantifying how
badly the cheap variant under-estimates epistemic uncertainty, because that is the variant
one is tempted to ship. An independent ensemble trains N complete heads; the shared variant
trains one trunk with N output heads, so the members cannot disagree about anything the
trunk has already decided. The prediction is that it under-disperses, and the question is by
how much.

**The GP gets this for free, which is part of why Step 2.4 moved to one.** Its predictive
variance is closed-form and drops out of the same Cholesky factor as the mean, where a deep
ensemble has to approximate the same quantity with N training runs.

    python -I scripts/calibration.py --dry-run
    python -I scripts/calibration.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from biodreamer import tracking
from biodreamer.protein_dreamer.cache import sequence_hash
from biodreamer.reward import RewardConfig, RewardHead
from biodreamer.reward.data import assays_for, detection_limits, h1_split, load_assay
from biodreamer.reward.losses import pairwise_ranking_loss, standardise_per_assay

sys.path.insert(0, str(Path(__file__).resolve().parent))

GP_PRED = Path("data/processed/gp_predictions.npz")
CACHE = Path("data/processed/pooled_cache")
BACKBONE = "westlake-repl_SaProt_650M_AF2"
OUT = Path("data/processed/calibration.json")
REPORT = Path("results/calibration.md")
COVERAGES = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]


def rho(a: np.ndarray, b: np.ndarray) -> float:
    r = spearmanr(a, b).statistic
    return 0.0 if np.isnan(r) else float(r)


def rank_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """|rank(pred) - rank(truth)| / n, per variant.

    **The scale-free error, and the only honest one here.** A pairwise ranking loss fixes
    the ORDER of the outputs and leaves their scale and offset completely free, so the
    independent ensemble's members come out with predictions whose sd is ~2.5 while the
    shared-trunk one sits at ~0.06. An absolute residual against a z-scored truth then
    measures the arbitrary scale of each member far more than it measures error, and the
    first version of this script reported exactly that artifact: it had the independent
    ensemble's uncertainty ANTI-correlated with error at -0.204, which was a statement
    about output scale, not about calibration.
    """
    from scipy.stats import rankdata

    n = len(pred)
    return np.abs(rankdata(pred) - rankdata(truth)) / n


def error_vs_uncertainty(pred, truth, sd, bins: int = 10) -> list[dict]:
    """Bin by predicted sd and report the realised error in each bin.

    Rank error, so bins are comparable across assays with different dynamic ranges and
    across models whose output scale is unconstrained.
    """
    out = []
    edges = np.quantile(sd, np.linspace(0, 1, bins + 1))
    err = rank_error(pred, truth)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (sd >= lo) & (sd <= hi) if i == bins - 1 else (sd >= lo) & (sd < hi)
        if m.sum() < 10:
            continue
        out.append({
            "bin": i,
            "sd_mean": float(sd[m].mean()),
            "abs_error_mean": float(err[m].mean()),
            "n": int(m.sum()),
        })
    return out


def risk_coverage(pred, truth, sd) -> dict[str, float]:
    """Spearman on the most-confident fraction of the test set.

    Reported as rho rather than MSE because rho is what H1 is scored on and what a planner
    actually uses -- it ranks candidates, it does not need calibrated absolute values.
    """
    order = np.argsort(sd)  # ascending: most confident first
    out = {}
    for c in COVERAGES:
        k = max(int(len(order) * c), 30)
        idx = order[:k]
        out[f"{c:.2f}"] = rho(pred[idx], truth[idx])
    return out


def interval_coverage(pred, truth_std, sd, z: float = 1.645) -> float:
    """Empirical coverage of the nominal 90% predictive interval."""
    return float((np.abs(pred - truth_std) <= z * sd).mean())


class SharedTrunkEnsemble(torch.nn.Module):
    """One adapter and trunk, N fitness heads. The cheap ensemble.

    Built from a `RewardHead`'s own submodules rather than by editing `head.py`: the
    library's architecture is not what is under test here, the ensembling strategy is.
    """

    def __init__(self, cfg: RewardConfig, members: int) -> None:
        super().__init__()
        base = RewardHead(cfg)
        self.adapter, self.trunk = base.adapter, base.trunk
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(cfg.trunk_dim, 1) for _ in range(members)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, d) -> (members, B)."""
        h = self.trunk(self.adapter.net(z))
        return torch.stack([hd(h).squeeze(-1) for hd in self.heads])


def train_independent(X, y, assay, cens, order_t, cfg, members, epochs, lr, max_pairs):
    """N complete heads, each with its own adapter, trunk and output layer."""
    preds = []
    for m in range(members):
        torch.manual_seed(2_000 + m)
        head = RewardHead(cfg)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        gen = torch.Generator().manual_seed(2_000 + m)
        head.train()
        for _ in range(epochs):
            loss = pairwise_ranking_loss(
                head(X, order=order_t, assay=assay)["fitness"], y, assay,
                censored=cens, max_pairs=max_pairs, generator=gen,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        preds.append(head)
    return preds


def train_shared(X, y, assay, cens, order_t, cfg, members, epochs, lr, max_pairs):
    torch.manual_seed(2_000)
    ens = SharedTrunkEnsemble(cfg, members)
    opt = torch.optim.Adam(ens.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(2_000)
    ens.train()
    for _ in range(epochs):
        out = ens(X)
        loss = sum(
            pairwise_ranking_loss(
                out[m], y, assay, censored=cens, max_pairs=max_pairs, generator=gen
            )
            for m in range(members)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    ens.eval()
    return ens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="benchmark", choices=["benchmark", "planning"])
    ap.add_argument("--seed", type=int, default=0, help="split seed; LOCKED to H1's 0")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-pairs", type=int, default=4096)
    ap.add_argument("--ens-cap", type=int, default=20000,
                    help="test rows per assay for the ensemble comparison")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = assays_for(args.instance)
    limits = detection_limits()

    # ---------- part 1: the GP, whose uncertainty is closed-form ----------
    gp: dict[str, dict] = {}
    if GP_PRED.exists():
        z = np.load(GP_PRED)
        have = sorted({k.split("/")[0] for k in z.files})
        for n in have:
            pred, var = z[f"{n}/pred"].astype(np.float64), z[f"{n}/var"].astype(np.float64)
            truth = z[f"{n}/truth"].astype(np.float64)
            ymu, ysd = float(z[f"{n}/ymu"]), float(z[f"{n}/ysd"])
            sd = np.sqrt(var)
            truth_std = (truth - ymu) / (ysd + 1e-12)
            gp[n] = {
                "rho": rho(pred, truth),
                # the headline: does predicted sd rank realised error? Rank error, so the
                # number is comparable with the ensembles, whose scale is unconstrained
                "rho_sd_vs_error": rho(sd, rank_error(pred, truth)),
                # the GP alone has a meaningful scale -- it is fit with a Gaussian
                # likelihood on standardised y -- so interval coverage is only valid here
                "interval_coverage_90": interval_coverage(pred, truth_std, sd),
                "sd_mean": float(sd.mean()),
                "risk_coverage": risk_coverage(pred, truth, sd),
                "error_vs_uncertainty": error_vs_uncertainty(pred, truth, sd),
            }
            print(f"  GP {n:30s} rho {gp[n]['rho']:+.3f}  "
                  f"sd-ranks-error {gp[n]['rho_sd_vs_error']:+.3f}  "
                  f"90% coverage {gp[n]['interval_coverage_90']:.3f}")
    else:
        print(f"  {GP_PRED} missing -- run scripts/gp_reward.py first; GP part skipped")

    # ---------- part 2: shared versus independent trunk ----------
    Xtr, ytr, atr, ctr, otr, plan = [], [], [], [], [], []
    for k, n in enumerate(names):
        a = load_assay(n, limits)
        tr, te = h1_split(a.order, seed=args.seed)
        if len(te) > args.ens_cap:
            te = np.sort(np.random.default_rng(args.seed).choice(te, args.ens_cap, False))
        zc = np.load(CACHE / f"{BACKBONE}__{n}.npz")
        row = {kk: i for i, kk in enumerate(zc["keys"])}
        emb = zc["emb"]

        def rows(idx):
            return emb[[row[sequence_hash(s)] for s in a.sequence[idx]]].astype(np.float32)

        Xtr.append(rows(tr))
        ytr.append(a.score[tr])
        atr.append(np.full(len(tr), k))
        ctr.append(a.censored[tr])
        otr.append(a.order[tr])
        plan.append((n, rows(te), a.score[te], a.order[te]))
        zc.close()
        print(f"  {n:30s} train {len(tr):>4,}  test(capped) {len(te):>6,}")
    if args.dry_run:
        return 0

    X = torch.from_numpy(np.concatenate(Xtr))
    assay = torch.from_numpy(np.concatenate(atr))
    y = standardise_per_assay(
        torch.from_numpy(np.concatenate(ytr).astype(np.float32)), assay
    )
    cens = torch.from_numpy(np.concatenate(ctr))
    order_t = torch.from_numpy(np.concatenate(otr).astype(np.int64))
    cfg = RewardConfig(backbone_dim=X.shape[1], pool_first=True)

    t0 = time.time()
    indep = train_independent(X, y, assay, cens, order_t, cfg, args.members,
                              args.epochs, args.lr, args.max_pairs)
    print(f"  independent ensemble: {args.members} heads in {time.time() - t0:.0f}s")
    t0 = time.time()
    shared = train_shared(X, y, assay, cens, order_t, cfg, args.members,
                          args.epochs, args.lr, args.max_pairs)
    print(f"  shared-trunk ensemble: {args.members} heads in {time.time() - t0:.0f}s")

    ens: dict[str, dict] = {}
    for n, xte, truth, ordr in plan:
        xt = torch.from_numpy(xte)
        with torch.no_grad():
            pi = np.stack([
                h(xt, order=torch.from_numpy(ordr.astype(np.int64)),
                  assay=torch.zeros(len(xte), dtype=torch.long))["fitness"].numpy()
                for h in indep
            ])
            ps = shared(xt).numpy()
        out = {}
        for tag, p in (("independent", pi), ("shared", ps)):
            mu, sd = p.mean(0), p.std(0)
            out[tag] = {
                "rho": rho(mu, truth),
                "sd_mean": float(sd.mean()),
                "rho_sd_vs_error": rho(sd, rank_error(mu, truth)),
                "risk_coverage": risk_coverage(mu, truth, sd),
            }
        out["dispersion_ratio"] = out["shared"]["sd_mean"] / max(
            out["independent"]["sd_mean"], 1e-12
        )
        ens[n] = out
        print(f"  ens {n:30s} indep sd {out['independent']['sd_mean']:.4f}  "
              f"shared sd {out['shared']['sd_mean']:.4f}  "
              f"ratio {out['dispersion_ratio']:.3f}")

    ratios = np.array([v["dispersion_ratio"] for v in ens.values()])
    metrics = {
        "shared_over_independent_dispersion_mean": float(ratios.mean()),
        "shared_over_independent_dispersion_median": float(np.median(ratios)),
        "ensemble_rho_independent": float(
            np.mean([v["independent"]["rho"] for v in ens.values()])
        ),
        "ensemble_rho_shared": float(np.mean([v["shared"]["rho"] for v in ens.values()])),
        "ensemble_sd_ranks_error_independent": float(
            np.mean([v["independent"]["rho_sd_vs_error"] for v in ens.values()])
        ),
        "ensemble_sd_ranks_error_shared": float(
            np.mean([v["shared"]["rho_sd_vs_error"] for v in ens.values()])
        ),
    }
    if gp:
        metrics["gp_sd_ranks_error"] = float(
            np.mean([v["rho_sd_vs_error"] for v in gp.values()])
        )
        metrics["gp_interval_coverage_90"] = float(
            np.mean([v["interval_coverage_90"] for v in gp.values()])
        )
    print()
    for k, v in metrics.items():
        print(f"{k:45s} {v:+.4f}")

    OUT.write_text(json.dumps(
        {"instance": args.instance, "seed": args.seed, "members": args.members,
         "gp": gp, "ensembles": ens, "metrics": metrics}, indent=2) + "\n")
    print(f"\nwrote {OUT}")

    with tracking.run(
        "2.3", "shared_vs_independent_trunk",
        variant={"instance": args.instance, "members": args.members},
        config={"epochs": args.epochs, "lr": args.lr, "max_pairs": args.max_pairs,
                "ens_cap": args.ens_cap},
        seed=args.seed, datasets=[str(CACHE)],
        notes=("Step 2.3. The headline is whether predicted sd RANKS realised error, "
               "which is the only property the policy needs; interval coverage is "
               "secondary because per-assay Spearman does not use absolute values. The "
               "shared-trunk ensemble is built from RewardHead's own submodules so the "
               "architecture is held fixed and only the ensembling strategy varies. "
               "Ensemble test sets are capped; the GP part is uncapped."),
    ) as r:
        r.log_metrics(metrics)
        r.log_artifact(str(OUT), kind="table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
