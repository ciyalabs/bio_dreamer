"""Step 2.4 — Kermut-sequence on our own cache: changing the ESTIMATOR, not the features.

Step 2.2 found a ridge and a 266k MLP scoring the same on the pooled vector and concluded
the *features* were saturated. That inference does not hold. A ridge and a small MLP on one
pooled vector are near-equivalent function classes, so their agreeing is close to
tautological -- it is evidence that the features are saturated **for ridge-like
predictors**, and says nothing about the features themselves. Step 2.4 then showed the same
thing from the other side: adding a zero-shot *column* to the ridge bought +0.055, and
+0.000 on nine of eleven assays. The model absorbs a little and saturates.

So the untested axis is the predictor class, and there is a published instance of it to
copy rather than invent. **Kermut-sequence** is Kermut with its structure kernel removed --
the configuration the fine-tuning literature benchmarks against, where it scores 0.614 at
n = 128 on multi-mutants. Its full specification is three things, and this project already
has all three on disk:

    k(x, x') = exp(-||z - z'||^2 / 2s^2)      z = mean-pooled 1280-d embedding
    m(x)     = alpha * f0(x) + beta           f0 = zero-shot log-likelihood ratio
    exact GP regression, hyperparameters by marginal likelihood

`f0` is not an approximation of Kermut's mean function, it **is** it: Kermut defines the
zero-shot value as the summed log-likelihood ratio between variant and wild-type residue at
the mutated positions, which is exactly what `zeroshot_baseline.score_variants` computes.

**Why an exact GP is free here.** A GP costs O(n^3) in training points, and the registered
H1 protocol caps training at 300 per assay. A 300x300 Cholesky is microseconds. The whole
corpus runs on a laptop CPU in minutes -- the thing that made this look expensive was never
the GP, it was the assumption that it needed a library and a GPU.

The ladder separates predictor class from prior mean, because the two were conflated in the
earlier write-up:

    ridge + order            the Step 2.2 baseline, +0.390        (cited, not refitted)
    GP, constant mean        predictor class alone
    GP, zero-shot mean       + Kermut's mean function = Kermut-sequence
    GP, zero-shot mean, order  + this project's depth term, which Kermut does not have

    python -I scripts/gp_reward.py --dry-run
    python -I scripts/gp_reward.py
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
from biodreamer.reward.data import assays_for, detection_limits, h1_split, load_assay

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroshot_baseline import cache_path, within_order_rho  # noqa: E402

CACHE = Path("data/processed/pooled_cache")
BACKBONE = "westlake-repl_SaProt_650M_AF2"
ZS_BACKBONE = "saprot"  # the default fixed in PREREGISTRATION.md, not the better scorer
OUT = Path("data/processed/gp_reward.json")

# cited from scripts/h1_benchmark.py on these same 11 assays and this same split seed
RIDGE_BASELINE = 0.390


class ExactGP(torch.nn.Module):
    """Squared-exponential GP with an affine zero-shot mean function.

    Written out rather than pulled from gpytorch: the model is twenty lines at this size,
    and the mean function is the part that matters most here, so it should be visible
    rather than configured.
    """

    def __init__(self, zero_shot_mean: bool) -> None:
        super().__init__()
        self.log_lengthscale = torch.nn.Parameter(torch.zeros(1))
        self.log_outputscale = torch.nn.Parameter(torch.zeros(1))
        self.log_noise = torch.nn.Parameter(torch.tensor([-2.0]))
        self.beta = torch.nn.Parameter(torch.zeros(1))          # the constant
        self.zero_shot_mean = zero_shot_mean
        # alpha exists only in the zero-shot variant; a constant-mean GP must not be able
        # to read f0 through a back door, or the ablation measures nothing
        self.alpha = torch.nn.Parameter(torch.zeros(1)) if zero_shot_mean else None

    def mean(self, f0: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            return self.beta.expand(f0.shape[0])
        return self.alpha * f0 + self.beta

    def kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(a, b).pow(2)
        return self.log_outputscale.exp() * torch.exp(
            -0.5 * d2 / self.log_lengthscale.exp().pow(2)
        )

    def nll(self, x: torch.Tensor, y: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        k = self.kernel(x, x) + torch.eye(n, dtype=x.dtype) * (
            self.log_noise.exp() + 1e-6
        )
        lo = torch.linalg.cholesky(k)
        r = (y - self.mean(f0)).unsqueeze(-1)
        a = torch.cholesky_solve(r, lo)
        return 0.5 * (r * a).sum() + lo.diagonal().log().sum() + 0.5 * n * np.log(2 * np.pi)

    @torch.no_grad()
    def predict(self, xtr, ytr, f0tr, chunks, with_var: bool = False):
        """`chunks` yields (x, f0) blocks of the test set, already standardised.

        Streamed rather than taking one test matrix, because HIS7 alone has 494,494 test
        variants and 494,494 x 1280 in float64 is **5 GB** -- materialising it, plus the
        hstack copy a ridge needs, is what exhausted this machine's memory on the first
        attempt. The GP never needs more than one block at a time.
        """
        n = xtr.shape[0]
        k = self.kernel(xtr, xtr) + torch.eye(n, dtype=xtr.dtype) * (
            self.log_noise.exp() + 1e-6
        )
        lo = torch.linalg.cholesky(k)
        a = torch.cholesky_solve((ytr - self.mean(f0tr)).unsqueeze(-1), lo)
        out, var = [], []
        for x, f0 in chunks:
            ks = self.kernel(x, xtr)
            out.append((self.mean(f0) + (ks @ a).squeeze(-1)).numpy())
            if with_var:
                # Step 2.3 consumes this. k(x*,x*) is the outputscale for an RBF, so the
                # prior variance is constant and the whole signal is the reduction from
                # being near training points -- which is exactly the epistemic term the
                # policy needs and the thing a deep ensemble only approximates.
                v = torch.linalg.solve_triangular(lo, ks.T, upper=False)
                var.append(
                    (
                        self.log_outputscale.exp()
                        + self.log_noise.exp()
                        - v.pow(2).sum(0)
                    ).clamp(min=1e-12).numpy()
                )
        pred = np.concatenate(out)
        return (pred, np.concatenate(var)) if with_var else pred


def fit(xtr, ytr, f0tr, zero_shot_mean: bool, steps: int, lr: float, seed: int) -> ExactGP:
    torch.manual_seed(seed)
    gp = ExactGP(zero_shot_mean).double()
    # a sensible lengthscale start: the median pairwise distance. Starting at 1.0 on
    # 1280-d standardised vectors puts every off-diagonal kernel entry at ~0, the
    # gradient at ~0, and the GP never leaves its prior.
    with torch.no_grad():
        d = torch.cdist(xtr, xtr)
        gp.log_lengthscale.fill_(float(torch.log(d[d > 0].median())))
    opt = torch.optim.Adam(gp.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = gp.nll(xtr, ytr, f0tr)
        loss.backward()
        opt.step()
    return gp


def rho(pred: np.ndarray, truth: np.ndarray) -> float:
    r = spearmanr(pred, truth).statistic
    return 0.0 if np.isnan(r) else float(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="benchmark", choices=["benchmark", "planning"])
    ap.add_argument("--seed", type=int, default=0, help="split seed; LOCKED to H1's 0")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=20000,
                    help="test rows per block; 20k x 1281 float64 is ~205 MB")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    zs_file = cache_path(ZS_BACKBONE, "masked-marginals")
    if not zs_file.exists():
        raise FileNotFoundError(f"{zs_file} missing -- run scripts/zeroshot_baseline.py")
    zs = np.load(zs_file)

    names = assays_for(args.instance)
    limits = detection_limits()
    # (label, zero_shot_mean, use_order)
    configs = [
        ("gp_constant_mean", False, False),
        ("gp_zeroshot_mean", True, False),
        ("gp_zeroshot_mean_order", True, True),
    ]
    res: dict[str, dict[str, float]] = {c[0]: {} for c in configs}
    wo: dict[str, dict[str, float]] = {c[0]: {} for c in configs}
    saved: dict[str, np.ndarray] = {}  # predictions + variances for Step 2.3

    for n in names:
        a = load_assay(n, limits)
        tr, te = h1_split(a.order, seed=args.seed)
        z = np.load(CACHE / f"{BACKBONE}__{n}.npz")
        row = {k: i for i, k in enumerate(z["keys"])}
        emb = z["emb"]

        def feats(idx):
            return emb[[row[sequence_hash(s)] for s in a.sequence[idx]]].astype(np.float64)

        xtr_raw = feats(tr)
        # standardise embeddings on the TRAIN rows only; the lengthscale would otherwise
        # have to absorb a per-dimension scale that varies by orders of magnitude
        mu, sd = xtr_raw.mean(0), xtr_raw.std(0) + 1e-8
        xtr_s = (xtr_raw - mu) / sd
        # per-assay standardisation of y, as everywhere else in this project
        ytr = a.score[tr]
        ytr_s = (ytr - ytr.mean()) / (ytr.std() + 1e-12)
        f0tr_raw, f0te_raw = zs[f"{n}/tr"], zs[f"{n}/te"]
        fm, fs = f0tr_raw.mean(), f0tr_raw.std() + 1e-12
        f0tr = torch.tensor((f0tr_raw - fm) / fs)
        f0te = torch.tensor((f0te_raw - fm) / fs)
        yt = torch.tensor(ytr_s)

        if args.dry_run:
            print(f"{n:30s} train={len(tr):4d} test={len(te):7,d} dim={xtr_s.shape[1]}")
            continue

        otr = a.order[tr].astype(np.float64)
        om, os_ = otr.mean(), otr.std() + 1e-12

        def test_chunks(use_order: bool, size: int = args.chunk):
            for s0 in range(0, len(te), size):
                blk = te[s0 : s0 + size]
                x = (feats(blk) - mu) / sd
                if use_order:
                    o = ((a.order[blk].astype(np.float64) - om) / os_).reshape(-1, 1)
                    x = np.hstack([x, o])
                yield torch.tensor(x), f0te[s0 : s0 + size]

        for label, zsm, use_order in configs:
            # one extra standardised column for depth; the kernel then measures similarity
            # in sequence AND depth, which is how a GP is given the term the ridge baseline
            # gets as a feature. Kermut has no such term, which is why it is a separate row.
            xtr = (
                np.hstack([xtr_s, ((otr - om) / os_).reshape(-1, 1)])
                if use_order
                else xtr_s
            )
            xt = torch.tensor(xtr)
            t0 = time.time()
            gp = fit(xt, yt, f0tr, zsm, args.steps, args.lr, args.seed)
            want_var = label == "gp_zeroshot_mean"
            if want_var:
                pred, pvar = gp.predict(
                    xt, yt, f0tr, test_chunks(use_order), with_var=True
                )
                saved[f"{n}/pred"] = pred.astype(np.float32)
                saved[f"{n}/var"] = pvar.astype(np.float32)
                saved[f"{n}/truth"] = a.score[te].astype(np.float32)
                saved[f"{n}/order"] = a.order[te].astype(np.int16)
                # predictions live on the per-assay STANDARDISED scale (y was centred and
                # scaled on the train rows), so Step 2.3 needs these two numbers to put
                # truth on the same scale before it can check interval coverage
                saved[f"{n}/ymu"] = np.float32(ytr.mean())
                saved[f"{n}/ysd"] = np.float32(ytr.std())
            else:
                pred = gp.predict(xt, yt, f0tr, test_chunks(use_order))
            res[label][n] = rho(pred, a.score[te])
            wo[label][n] = within_order_rho(pred, a.score[te], a.order[te])
            print(
                f"  {label:24s} {n:30s} {res[label][n]:+.3f} "
                f"(within-order {wo[label][n]:+.3f})  {time.time() - t0:.0f}s"
            )
    if args.dry_run:
        return 0

    print()
    metrics: dict[str, float] = {}
    for label, _, _ in configs:
        v = np.array(list(res[label].values()))
        w = np.array(list(wo[label].values()))
        metrics[f"{label}/mean"] = float(v.mean())
        metrics[f"{label}/median"] = float(np.median(v))
        metrics[f"{label}/within_order_mean"] = float(w.mean())
        metrics[f"{label}/n_positive"] = float((v > 0).sum())
        print(
            f"{label:24s} mean {v.mean():+.3f}  median {np.median(v):+.3f}  "
            f"within-order {w.mean():+.3f}  positive {(v > 0).sum()}/{len(v)}"
        )

    # the two registered quantities. `gp_zeroshot_mean` is Kermut-sequence as published --
    # no order term, because Kermut has none -- so it is the one the thresholds are read on.
    best = metrics["gp_zeroshot_mean/mean"]
    metrics["spearman_mean_gp"] = best
    metrics["gp_margin_over_ridge"] = best - RIDGE_BASELINE
    print(f"\nridge + count (cited)        {RIDGE_BASELINE:+.3f}")
    print(f"spearman_mean_gp             {best:+.3f}   (threshold >= 0.55)")
    print(f"gp_margin_over_ridge         {metrics['gp_margin_over_ridge']:+.3f}   "
          f"(threshold > 0.05)")

    OUT.write_text(json.dumps(
        {"instance": args.instance, "seed": args.seed, "per_assay": res,
         "within_order": wo, "metrics": metrics,
         "ridge_baseline_cited": RIDGE_BASELINE,
         "kermut_sequence_published_n128_multimutant": 0.614}, indent=2) + "\n")
    print(f"wrote {OUT}")
    pred_file = Path("data/processed/gp_predictions.npz")
    np.savez(pred_file, **saved)
    print(f"wrote {pred_file} ({pred_file.stat().st_size / 2**20:.0f} MB) for step 2.3")

    with tracking.run(
        "2.4", "gp_composite_kernel",
        variant={"instance": args.instance, "kernel": "rbf_sequence",
                 "mean": "affine_zeroshot", "structure_kernel": False},
        config={"steps": args.steps, "lr": args.lr, "backbone": BACKBONE,
                "zeroshot": ZS_BACKBONE},
        seed=args.seed, datasets=[str(CACHE), str(zs_file)],
        notes=("Kermut-sequence reproduced on this project's own cache: RBF on mean-pooled "
               "1280-d embeddings, m(x)=alpha*f0(x)+beta with f0 the summed log-likelihood "
               "ratio at mutated positions, exact GP, hyperparameters by marginal "
               "likelihood. No structure kernel -- that is the published Kermut-sequence "
               "configuration and the one the fine-tuning literature benchmarks, not a "
               "shortcut. Thresholds registered in plan.yaml before this ran."),
    ) as r:
        r.log_metrics(metrics)
        r.log_artifact(str(OUT), kind="table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
