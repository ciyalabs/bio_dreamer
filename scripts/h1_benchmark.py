"""Step 2.2 — H1. The reward head against the additive null and a frozen regression.

**This is a gate.** `PREREGISTRATION.md` fixes four conditions and failing them stops the
project at Phase 2. They are checked here and the verdict is written out whichever way it
falls; no threshold is edited after seeing a number.

| condition | threshold |
|---|---|
| `spearman_mean` | >= 0.55 |
| `margin_over_additive_null` | > 0.05 |
| `frac_assays_beating_null` | >= 0.70 |
| `spearman_mean_vs_frozen_regression` | > 0 |

**The additive null is the decisive baseline**, and it is strictly stronger than the
order-only control Step 2.1 used: one fitted coefficient per observed substitution, so it
captures both "which substitution" and, by summing more terms, "how many". A substitution
never seen in training contributes zero, which is the honest additive prediction for an
unmeasured effect rather than an imputation.

`biodreamer.eval.decompose.interaction_design` is NOT reused here. It builds a fixed-site
combinatorial design for GB1-style hypercubes; ProteinGym mutations scatter over hundreds
of positions, so the design is one column per observed substitution instead.

**Uses the `benchmark` instance**, as the pre-registration specifies, which is also the
only instance allowed to touch the banned assays and is therefore barred from producing a
Phase 6 result.

    python -I scripts/h1_benchmark.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

from biodreamer import tracking
from biodreamer.protein_dreamer.cache import sequence_hash
from biodreamer.reward import RewardConfig, RewardHead
from biodreamer.reward.data import assays_for, banned_assays, detection_limits, h1_split, load_assay
from biodreamer.reward.losses import pairwise_ranking_loss, standardise_per_assay

CACHE = Path("data/processed/pooled_cache")
BACKBONE = "westlake-repl_SaProt_650M_AF2"
OUT = Path("data/processed/h1_benchmark.json")
REPORT = Path("results/reward_benchmark.md")
ALPHAS = np.logspace(-1, 5, 13)

# published mean Spearman on the same higher-order extrapolation protocol, as recorded in
# PREREGISTRATION.md. These are cited, NOT recomputed here, and they are reported on
# their own assay sets -- see the comparability caveat in the generated report.
LITERATURE = {"ProteinNPT": 0.56, "MULTI-evolve": 0.61, "Kermut": 0.62, "Cerebra": 0.65}

THRESHOLDS = {
    "spearman_mean": (">=", 0.55),
    "margin_over_additive_null": (">", 0.05),
    "frac_assays_beating_null": (">=", 0.70),
    "spearman_mean_vs_frozen_regression": (">", 0.0),
}


def substitution_design(mutants: np.ndarray, vocab: dict[str, int] | None = None):
    """Rows = variants, columns = observed substitutions. Builds the vocab if absent."""
    if vocab is None:
        vocab = {}
        for m in mutants:
            for sub in m.split(":"):
                vocab.setdefault(sub, len(vocab))
    rows, cols = [], []
    for i, m in enumerate(mutants):
        for sub in m.split(":"):
            j = vocab.get(sub)
            if j is not None:          # unseen substitution -> zero, i.e. neutral
                rows.append(i)
                cols.append(j)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(len(mutants), max(len(vocab), 1)))
    return X, vocab


def additive_null(a, tr: np.ndarray, te: np.ndarray, seed: int = 0) -> np.ndarray:
    """Per-substitution additive model fitted on train, predicting the test variants."""
    Xtr, vocab = substitution_design(a.mutant[tr])
    Xte, _ = substitution_design(a.mutant[te], vocab)
    y = a.score[tr]
    # ridge strength by inner CV on the TRAINING rows only
    best, best_r2 = ALPHAS[0], -np.inf
    for alpha in ALPHAS:
        oof = np.zeros(len(y))
        for i, j in KFold(5, shuffle=True, random_state=seed).split(Xtr):
            oof[j] = Ridge(alpha=alpha).fit(Xtr[i], y[i]).predict(Xtr[j])
        r2 = -np.mean((oof - y) ** 2)
        if r2 > best_r2:
            best, best_r2 = alpha, r2
    return Ridge(alpha=best).fit(Xtr, y).predict(Xte)


def main() -> int:
    names = assays_for("benchmark")
    banned = banned_assays()
    limits = detection_limits()
    seed = 0
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    print(f"H1 on the `benchmark` instance: {len(names)} assays "
          f"({len(set(names) & banned)} banned, so no Phase 6 claim may come from this)\n")

    # ---- the reward head, in the configuration Step 2.1 settled on ----
    Xtr, ytr, atr, ctr, otr, plan = [], [], [], [], [], []
    for k, n in enumerate(names):
        a = load_assay(n, limits)
        tr, te = h1_split(a.order, seed=seed)
        z = np.load(CACHE / f"{BACKBONE}__{n}.npz")
        index = {kk: i for i, kk in enumerate(z["keys"].tolist())}
        rows = lambda idx: np.array([index[sequence_hash(s)] for s in a.sequence[idx]])
        Xtr.append(z["emb"][rows(tr)].astype(np.float32))
        ytr.append(a.score[tr]); atr.append(np.full(len(tr), k))
        ctr.append(a.censored[tr]); otr.append(a.order[tr])
        plan.append((n, k, rows(te), tr, te))
        z.close()

    X = torch.from_numpy(np.concatenate(Xtr))
    assay = torch.from_numpy(np.concatenate(atr))
    y = standardise_per_assay(
        torch.from_numpy(np.concatenate(ytr).astype(np.float32)), assay)
    cens = torch.from_numpy(np.concatenate(ctr))
    order_t = torch.from_numpy(np.concatenate(otr).astype(np.int64))

    head = RewardHead(RewardConfig(backbone_dim=X.shape[1], pool_first=True,
                                   depth_aware=True, depth_transform="linear"))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    head.train()
    t0 = time.time()
    for _ in range(300):
        loss = pairwise_ranking_loss(
            head(X, order=order_t)["fitness"], y, assay, censored=cens,
            max_pairs=4096, generator=gen)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    print(f"reward head trained in {time.time()-t0:.0f}s\n")

    res: dict[str, dict] = {}
    print(f"{'assay':32s} {'head':>7s} {'null':>7s} {'ridge':>7s} {'ridge+d':>8s}  beats null")
    for n, k, rte, tr, te in plan:
        a = load_assay(n, limits)
        z = np.load(CACHE / f"{BACKBONE}__{n}.npz")
        emb = z["emb"]
        od = torch.from_numpy(a.order[te].astype(np.int64))
        ph = np.empty(len(rte), np.float32)
        with torch.no_grad():
            for s in range(0, len(rte), 8192):
                b = rte[s : s + 8192]
                ph[s : s + len(b)] = head(
                    torch.from_numpy(emb[b].astype(np.float32)),
                    order=od[s : s + len(b)])["fitness"].numpy()
        # frozen-encoder regression: a plain ridge on the same pooled features, with and
        # without mutation count, so "what did the architecture add" is separable
        index = {kk: i for i, kk in enumerate(z["keys"].tolist())}
        Etr = emb[[index[sequence_hash(s)] for s in a.sequence[tr]]].astype(np.float32)
        Ete = emb[rte].astype(np.float32)
        z.close()
        ridge = RidgeCV(alphas=ALPHAS).fit(Etr, a.score[tr]).predict(Ete)
        ridge_d = RidgeCV(alphas=ALPHAS).fit(
            np.hstack([Etr, a.order[tr][:, None]]), a.score[tr]
        ).predict(np.hstack([Ete, a.order[te][:, None]]))

        truth = a.score[te]
        r = {
            "head": float(spearmanr(ph, truth).statistic),
            "additive_null": float(spearmanr(additive_null(a, tr, te, seed), truth).statistic),
            "frozen_ridge": float(spearmanr(ridge, truth).statistic),
            "frozen_ridge_depth": float(spearmanr(ridge_d, truth).statistic),
            "n_test": int(len(te)),
            "banned": n in banned,
        }
        r["beats_null"] = bool(r["head"] > r["additive_null"])
        res[n] = r
        print(f"{n:32s} {r['head']:>+7.3f} {r['additive_null']:>+7.3f} "
              f"{r['frozen_ridge']:>+7.3f} {r['frozen_ridge_depth']:>+8.3f}  "
              f"{'yes' if r['beats_null'] else 'NO'}{'   [BANNED]' if r['banned'] else ''}",
              flush=True)

    h = np.array([res[n]["head"] for n in names])
    nul = np.array([res[n]["additive_null"] for n in names])
    rg = np.array([res[n]["frozen_ridge"] for n in names])
    rgd = np.array([res[n]["frozen_ridge_depth"] for n in names])
    metrics = {
        "spearman_mean": float(h.mean()),
        "spearman_median": float(np.median(h)),
        "additive_null_mean": float(nul.mean()),
        "margin_over_additive_null": float(h.mean() - nul.mean()),
        "frac_assays_beating_null": float(np.mean(h > nul)),
        "frozen_regression_mean": float(rg.mean()),
        "frozen_regression_depth_mean": float(rgd.mean()),
        "spearman_mean_vs_frozen_regression": float(h.mean() - rg.mean()),
        "n_assays": float(len(names)),
    }

    print("\n--- H1 verdict against PREREGISTRATION.md ---")
    verdict = {}
    for key, (op, thr) in THRESHOLDS.items():
        got = metrics[key]
        ok = got >= thr if op == ">=" else got > thr
        verdict[key] = {"value": got, "op": op, "threshold": thr, "pass": bool(ok)}
        print(f"  {key:38s} {got:>+7.3f} {op} {thr:<5} {'PASS' if ok else 'FAIL'}")
    passed = all(v["pass"] for v in verdict.values())
    metrics["h1_pass"] = float(passed)
    print(f"\n  H1: {'PASS -- proceed to Phase 3' if passed else 'FAIL -- do not proceed to Phase 3'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"metrics": metrics, "verdict": verdict,
                               "per_assay": res, "literature_cited": LITERATURE,
                               "instance": "benchmark"}, indent=2) + "\n")
    print(f"  wrote {OUT}")

    with tracking.run(
        "2.2", "vs_additive_null",
        variant={"instance": "benchmark", "baseline": "additive+frozen_ridge"},
        config={"epochs": 300, "lr": 1e-3, "depth_aware": True}, seed=seed,
        datasets=[str(CACHE)],
        notes=("H1 gate. Additive null is one coefficient per observed substitution, "
               "fitted on the <=300 order<=2 training rows, predicting unseen order>=3; "
               "unseen substitutions contribute zero. Literature figures are CITED from "
               "PREREGISTRATION.md on their own assay sets, not recomputed."),
    ) as r:
        r.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
        r.log_artifact(str(OUT), kind="table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
