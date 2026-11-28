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
from encode_positional import resolve_wild_type  # noqa: E402
from h1_benchmark import substitution_design  # noqa: E402
from structure_kernel import (  # noqa: E402
    StructureKernel,
    assay_vocabulary,
    load_site_data,
    pair_distances,
)
from zeroshot_baseline import cache_path, score_from_matrix, within_order_rho  # noqa: E402

CACHE = Path("data/processed/pooled_cache")
BACKBONE = "westlake-repl_SaProt_650M_AF2"
EMBED_BACKBONES = {  # cache-file prefixes written by scripts/encode_corpus.py
    "saprot": "westlake-repl_SaProt_650M_AF2",
    "esm2": "facebook_esm2_t33_650M_UR50D",
}
ZS_BACKBONE = "saprot"  # the default fixed in PREREGISTRATION.md, not the better scorer
BACKBONES_ZS = ("saprot", "esm2")
# Published zero-shot predictors, read from ProteinGym's own score archive rather than
# recomputed (scripts/extract_pgym_priors.py). These are ALIGNMENT-based: they read a
# multiple sequence alignment of the protein family, which is a category of evidence no
# prior this project has used so far carries at all. On the 11 benchmark assays' order>=3
# test set the computed ESM-2 prior scores +0.249 total / +0.164 within-order; these land
# between +0.397/+0.280 and +0.442/+0.328. The join is exact: our independently computed
# ESM-2 masked marginals reproduce ProteinGym's ESM2_650M column at rho = 1.0000.
PGYM_PRIORS = Path("data/processed/pgym_priors.npz")
PUBLISHED_ZS = {
    "gemme": "GEMME",
    "msa-transformer": "MSA_Transformer_ensemble",
    "trancepteve": "TranceptEVE_L",
    "eve": "EVE_ensemble",
}
OUT = Path("data/processed/gp_reward.json")
PRED = Path("data/processed/gp_predictions.npz")

# cited from scripts/h1_benchmark.py on these same 11 assays and this same split seed
RIDGE_BASELINE = 0.390


class ExactGP(torch.nn.Module):
    """Squared-exponential GP with an affine zero-shot mean function.

    Written out rather than pulled from gpytorch: the model is twenty lines at this size,
    and the mean function is the part that matters most here, so it should be visible
    rather than configured.
    """

    def __init__(self, zero_shot_mean: bool, n_seq: int | None = None,
                 n_mean: int = 1, structure: bool = False) -> None:
        super().__init__()
        self.log_lengthscale = torch.nn.Parameter(torch.zeros(1))
        self.log_outputscale = torch.nn.Parameter(torch.zeros(1))
        self.log_noise = torch.nn.Parameter(torch.tensor([-2.0]))
        self.beta = torch.nn.Parameter(torch.zeros(1))          # the constant
        self.zero_shot_mean = zero_shot_mean
        # COMPOSITE KERNEL. `n_seq` splits the input into [embedding | substitution
        # one-hot] and the kernel becomes pi*k_subs + (1-pi)*k_seq, which is Kermut's
        # composition (eq. 4) with the substitution design standing in for the structure
        # kernel this project has no ProteinMPNN for yet. Worth testing because the
        # additive null -- one coefficient per observed substitution -- reaches +0.324 on
        # its own, so substitution identity carries signal the RBF on a pooled vector may
        # not express. pi is learned through the marginal likelihood like everything else.
        self.n_seq = n_seq
        if n_seq is not None:
            self.log_lengthscale_subs = torch.nn.Parameter(torch.zeros(1))
            self.logit_pi = torch.nn.Parameter(torch.zeros(1))
        # STRUCTURE KERNEL. Kermut eq. 4 proper: pi*k_struct + (1-pi)*k_seq, with
        # k_struct comparing the ProteinMPNN site distributions and CA distances of the
        # mutated positions rather than a scalar summed over them. This is the component
        # Step 2.4 tested as a mean term and found worth nothing, which measured the
        # wrong thing: Kermut's largest ablation (-0.065) is this kernel, not that scalar.
        self.struct = StructureKernel() if structure else None
        if structure:
            self.logit_pi_struct = torch.nn.Parameter(torch.zeros(1))
        # alpha exists only in the zero-shot variant; a constant-mean GP must not be able
        # to read f0 through a back door, or the ablation measures nothing
        # `n_mean` > 1 when a second zero-shot term is supplied -- the ProteinMPNN
        # structure score alongside the sequence one. Kermut uses structure in its KERNEL
        # rather than its mean, but as a mean term it is the cheapest honest test of
        # whether inverse-folding information carries signal this model does not have.
        self.alpha = (
            torch.nn.Parameter(torch.zeros(max(n_mean, 1))) if zero_shot_mean else None
        )

    def mean(self, f0: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            return self.beta.expand(f0.shape[0])
        if f0.ndim == 1:
            return self.alpha[0] * f0 + self.beta
        return f0 @ self.alpha + self.beta

    def kernel(self, a: torch.Tensor, b: torch.Tensor, sa=None, sb=None,
               dists=None) -> torch.Tensor:
        if self.struct is not None:
            k_seq = torch.exp(
                -0.5 * torch.cdist(a, b).pow(2) / self.log_lengthscale.exp().pow(2)
            )
            pi = torch.sigmoid(self.logit_pi_struct)
            k_st = self.struct(sa, sb, dists)
            # k_struct is a SUM over pairs of mutated sites, so its scale grows with
            # mutation depth while k_seq is bounded by 1. Left raw, pi would have to
            # absorb that and the mixture would be dominated by whichever variant has
            # the most mutations. Normalising by the train-block maximum puts the two
            # terms on a comparable scale before they are mixed; the outputscale is
            # still free to rescale the whole kernel.
            k_st = k_st / (self._struct_scale + 1e-12)
            return self.log_outputscale.exp() * (pi * k_st + (1.0 - pi) * k_seq)
        if self.n_seq is None:
            d2 = torch.cdist(a, b).pow(2)
            return self.log_outputscale.exp() * torch.exp(
                -0.5 * d2 / self.log_lengthscale.exp().pow(2)
            )
        ns = self.n_seq
        k_seq = torch.exp(
            -0.5 * torch.cdist(a[:, :ns], b[:, :ns]).pow(2)
            / self.log_lengthscale.exp().pow(2)
        )
        k_sub = torch.exp(
            -0.5 * torch.cdist(a[:, ns:], b[:, ns:]).pow(2)
            / self.log_lengthscale_subs.exp().pow(2)
        )
        pi = torch.sigmoid(self.logit_pi)
        return self.log_outputscale.exp() * (pi * k_sub + (1.0 - pi) * k_seq)

    def nll(self, x: torch.Tensor, y: torch.Tensor, f0: torch.Tensor,
            s=None, dists=None) -> torch.Tensor:
        n = x.shape[0]
        k = self.kernel(x, x, s, s, dists) + torch.eye(n, dtype=x.dtype) * (
            self.log_noise.exp() + 1e-6
        )
        lo = torch.linalg.cholesky(k)
        r = (y - self.mean(f0)).unsqueeze(-1)
        a = torch.cholesky_solve(r, lo)
        return 0.5 * (r * a).sum() + lo.diagonal().log().sum() + 0.5 * n * np.log(2 * np.pi)

    @torch.no_grad()
    def predict(self, xtr, ytr, f0tr, chunks, with_var: bool = False,
                str_tr=None, dists_tr=None, dists_cross=None):
        """`chunks` yields (x, f0) blocks of the test set, already standardised.

        Streamed rather than taking one test matrix, because HIS7 alone has 494,494 test
        variants and 494,494 x 1280 in float64 is **5 GB** -- materialising it, plus the
        hstack copy a ridge needs, is what exhausted this machine's memory on the first
        attempt. The GP never needs more than one block at a time.
        """
        n = xtr.shape[0]
        k = self.kernel(xtr, xtr, str_tr, str_tr, dists_tr) + torch.eye(
            n, dtype=xtr.dtype
        ) * (self.log_noise.exp() + 1e-6)
        lo = torch.linalg.cholesky(k)
        a = torch.cholesky_solve((ytr - self.mean(f0tr)).unsqueeze(-1), lo)
        out, var = [], []
        for x, f0, s_blk in chunks:
            ks = self.kernel(x, xtr, s_blk, str_tr, dists_cross)
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


def fit(xtr, ytr, f0tr, zero_shot_mean: bool, steps: int, lr: float, seed: int,
        n_seq: int | None = None, str_tr=None, dists_tr=None) -> ExactGP:
    torch.manual_seed(seed)
    n_mean = 1 if f0tr.ndim == 1 else f0tr.shape[1]
    structure = str_tr is not None
    gp = ExactGP(zero_shot_mean, n_seq, n_mean, structure=structure).double()
    if structure:
        # k_struct sums over pairs of mutated sites, so its magnitude tracks mutation
        # depth rather than similarity. Fix the scale once, on the training block, so
        # that pi mixes two comparable quantities instead of also having to undo this.
        with torch.no_grad():
            raw = gp.struct(str_tr, str_tr, dists_tr)
            gp.register_buffer("_struct_scale", raw.max().clamp(min=1e-12).clone())
    # a sensible lengthscale start: the median pairwise distance. Starting at 1.0 on
    # 1280-d standardised vectors puts every off-diagonal kernel entry at ~0, the
    # gradient at ~0, and the GP never leaves its prior.
    with torch.no_grad():
        ns = xtr.shape[1] if n_seq is None else n_seq
        d = torch.cdist(xtr[:, :ns], xtr[:, :ns])
        gp.log_lengthscale.fill_(float(torch.log(d[d > 0].median())))
        if n_seq is not None:
            ds = torch.cdist(xtr[:, ns:], xtr[:, ns:])
            pos = ds[ds > 0]
            gp.log_lengthscale_subs.fill_(
                float(torch.log(pos.median())) if pos.numel() else 0.0
            )
    opt = torch.optim.Adam(gp.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = gp.nll(xtr, ytr, f0tr, str_tr, dists_tr)
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
    ap.add_argument(
        "--zs-backbone", default=ZS_BACKBONE,
        choices=sorted(BACKBONES_ZS) + sorted(PUBLISHED_ZS),
        help="which backbone supplies f0. SaProt is this project's registered default and "
             "is the primary; esm2 is a SENSITIVITY, and it is method-specified rather "
             "than test-selected because Kermut names ESM-2 for its mean function.",
    )
    ap.add_argument(
        "--embed-backbone", default="saprot", choices=sorted(EMBED_BACKBONES),
        help="which backbone supplies the 1280-d embeddings the kernel runs on. SaProt is "
             "the registered default; esm2 is the backbone Kermut specifies, and running "
             "it here makes the backbone the ONLY thing that changes between the two "
             "numbers -- H1's own assays, split and protocol either way.",
    )
    ap.add_argument(
        "--structure-mean", action="store_true",
        help="add the ProteinMPNN structure score as a SECOND mean-function term, so "
             "m(x) = a1*f0_seq(x) + a2*f0_struct(x) + b. Needs "
             "data/processed/mpnn_matrix.npz from scripts/structure_score.py.",
    )
    ap.add_argument(
        "--structure-kernel", action="store_true",
        help="add Kermut's STRUCTURE KERNEL as a fifth config: pi*k_struct + "
             "(1-pi)*k_seq, where k_struct compares the ProteinMPNN site distributions "
             "and CA distances of the mutated positions. This is the component Kermut's "
             "ablation ranks first (-0.065); --structure-mean tests a scalar summary of "
             "the same data and is a different, weaker thing. Needs "
             "data/processed/mpnn_matrix.npz and data/processed/wt_structures/.",
    )
    ap.add_argument("--chunk", type=int, default=20000,
                    help="test rows per block; 20k x 1281 float64 is ~205 MB")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # the two instances are different results and must not overwrite each other's files;
    # `benchmark` keeps the original names so earlier references stay valid
    global OUT, PRED, BACKBONE
    BACKBONE = EMBED_BACKBONES[args.embed_backbone]
    tag = "" if args.instance == "benchmark" else f"_{args.instance}"
    if args.embed_backbone != "saprot":
        tag += f"_emb-{args.embed_backbone}"
    if args.zs_backbone != ZS_BACKBONE:
        tag += f"_zs-{args.zs_backbone}"
    if args.structure_mean:
        tag += "_struct"
    if args.structure_kernel:
        tag += "_structkern"
    if tag:
        OUT = OUT.with_name(f"gp_reward{tag}.json")
        PRED = PRED.with_name(f"gp_predictions{tag}.npz")

    published = args.zs_backbone in PUBLISHED_ZS
    zs_file = PGYM_PRIORS if published else cache_path(args.zs_backbone, "masked-marginals")
    if not zs_file.exists():
        raise FileNotFoundError(
            f"{zs_file} missing -- run scripts/"
            + ("extract_pgym_priors.py" if published else "zeroshot_baseline.py")
        )
    zs = np.load(zs_file)
    if published:
        col = PUBLISHED_ZS[args.zs_backbone]

        def f0_for(name, tr, te):
            # published scores are stored per variant over the WHOLE assay, so they are
            # sliced by the split here; the computed caches are already split-shaped
            v = zs[f"{name}/{col}"]
            if not np.isfinite(v).all():
                raise ValueError(f"{col} has non-finite scores for {name}")
            return v[tr], v[te]
    else:

        def f0_for(name, tr, te):
            return zs[f"{name}/tr"], zs[f"{name}/te"]

    names = assays_for(args.instance)
    limits = detection_limits()
    mpnn = None
    if args.structure_mean:
        mp = Path("data/processed/mpnn_matrix.npz")
        if not mp.exists():
            raise FileNotFoundError(f"{mp} missing -- run scripts/structure_score.py")
        mpnn = np.load(mp)
        missing = [n for n in names if n not in mpnn.files]
        if missing:
            raise KeyError(f"{mp} has no structure matrix for {missing}")
        print(f"structure mean: {mp} covering {len(mpnn.files)} assays")
    # (label, zero_shot_mean, use_order, composite, structure)
    configs = [
        ("gp_constant_mean", False, False, False, False),
        ("gp_zeroshot_mean", True, False, False, False),
        ("gp_zeroshot_mean_order", True, True, False, False),
        ("gp_composite_kernel", True, False, True, False),
    ]
    if args.structure_kernel:
        configs.append(("gp_structure_kernel", True, False, False, True))
    res: dict[str, dict[str, float]] = {c[0]: {} for c in configs}
    wo: dict[str, dict[str, float]] = {c[0]: {} for c in configs}
    pis: dict[str, float] = {}
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
        f0tr_raw, f0te_raw = f0_for(n, tr, te)
        fm, fs = f0tr_raw.mean(), f0tr_raw.std() + 1e-12
        f0tr = torch.tensor((f0tr_raw - fm) / fs)
        f0te = torch.tensor((f0te_raw - fm) / fs)
        if mpnn is not None:
            wt = resolve_wild_type(a, n)
            st = score_from_matrix(a.mutant, mpnn[n], wt)
            sm, ss = st[tr].mean(), st[tr].std() + 1e-12
            f0tr = torch.stack([f0tr, torch.tensor((st[tr] - sm) / ss)], dim=1)
            f0te = torch.stack([f0te, torch.tensor((st[te] - sm) / ss)], dim=1)
        yt = torch.tensor(ytr_s)

        if args.dry_run:
            print(f"{n:30s} train={len(tr):4d} test={len(te):7,d} dim={xtr_s.shape[1]}")
            continue

        structure = args.structure_kernel
        S_tr = S_te = dists_tr = dists_cross = str_tr = None
        if structure:
            site = load_site_data(n)
            S_tr, S_te, keys, train_cols = assay_vocabulary(a.mutant[tr], a.mutant[te])
            # columns are the TRAINING mutations only -- the cross kernel never needs
            # more, and (L*19) x (L*19) would be 17M entries for the largest assay
            train_keys = [keys[i] for i in train_cols]
            dists_cross = [
                torch.tensor(d) for d in pair_distances(keys, train_keys, site)
            ]
            dists_tr = [d[train_cols] for d in dists_cross]
            str_tr = torch.tensor(S_tr[:, train_cols].toarray())

        otr = a.order[tr].astype(np.float64)
        om, os_ = otr.mean(), otr.std() + 1e-12
        dtr, vocab = substitution_design(a.mutant[tr])
        subs_tr = dtr.toarray()
        n_seq = xtr_s.shape[1]

        def test_chunks(use_order: bool, composite: bool, size: int = args.chunk):
            for s0 in range(0, len(te), size):
                blk = te[s0 : s0 + size]
                x = (feats(blk) - mu) / sd
                if use_order:
                    o = ((a.order[blk].astype(np.float64) - om) / os_).reshape(-1, 1)
                    x = np.hstack([x, o])
                if composite:
                    # built per block: HIS7's full one-hot would be 494,494 x |vocab|
                    x = np.hstack([x, substitution_design(a.mutant[blk], vocab)[0].toarray()])
                # the structure indicator stays SPARSE across the full mutation
                # vocabulary; see StructureKernel.forward for why it is not densified
                s_blk = S_te[s0 : s0 + size] if structure else None
                yield torch.tensor(x), f0te[s0 : s0 + size], s_blk

        for label, zsm, use_order, composite, use_struct in configs:
            # one extra standardised column for depth; the kernel then measures similarity
            # in sequence AND depth, which is how a GP is given the term the ridge baseline
            # gets as a feature. Kermut has no such term, which is why it is a separate row.
            xtr = (
                np.hstack([xtr_s, ((otr - om) / os_).reshape(-1, 1)])
                if use_order
                else xtr_s
            )
            if composite:
                xtr = np.hstack([xtr, subs_tr])
            xt = torch.tensor(xtr)
            t0 = time.time()
            gp = fit(xt, yt, f0tr, zsm, args.steps, args.lr, args.seed,
                     n_seq=n_seq if composite else None,
                     str_tr=str_tr if use_struct else None,
                     dists_tr=dists_tr if use_struct else None)
            if composite:
                pis[n] = float(torch.sigmoid(gp.logit_pi).detach())
            want_var = label == "gp_zeroshot_mean"
            if want_var:
                pred, pvar = gp.predict(
                    xt, yt, f0tr, test_chunks(use_order, composite), with_var=True,
                    str_tr=str_tr if use_struct else None,
                    dists_tr=dists_tr if use_struct else None,
                    dists_cross=dists_cross if use_struct else None,
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
                pred = gp.predict(
                    xt, yt, f0tr, test_chunks(use_order, composite),
                    str_tr=str_tr if use_struct else None,
                    dists_tr=dists_tr if use_struct else None,
                    dists_cross=dists_cross if use_struct else None,
                )
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
    for label, *_ in configs:
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
    print(f"\nspearman_mean_gp             {best:+.3f}   (threshold >= 0.55)")
    # RIDGE_BASELINE is scripts/h1_benchmark.py's number on the 11 BENCHMARK assays, so
    # subtracting it from a 7-assay planning mean would compare two different assay lists
    # -- the exact error this step promoted vs_literature to a precondition over.
    if args.instance == "benchmark":
        metrics["gp_margin_over_ridge"] = best - RIDGE_BASELINE
        print(f"ridge + count (cited)        {RIDGE_BASELINE:+.3f}")
        print(f"gp_margin_over_ridge         {metrics['gp_margin_over_ridge']:+.3f}   "
              f"(threshold > 0.05)")
    else:
        print(f"gp_margin_over_ridge         not computed: the cited ridge baseline is "
              f"the 11-assay benchmark mean, not {args.instance}'s")

    OUT.write_text(json.dumps(
        {"instance": args.instance, "seed": args.seed, "per_assay": res,
         "within_order": wo, "metrics": metrics,
         "composite_pi_per_assay": pis,
         "ridge_baseline_cited": RIDGE_BASELINE,
         "kermut_sequence_published_n128_multimutant": 0.614}, indent=2) + "\n")
    print(f"wrote {OUT}")
    np.savez(PRED, **saved)
    print(f"wrote {PRED} ({PRED.stat().st_size / 2**20:.0f} MB) for step 2.3")

    with tracking.run(
        "2.4", "gp_composite_kernel",
        variant={"instance": args.instance, "kernel": "rbf_sequence+composite",
                 "mean": "affine_zeroshot", "structure_kernel": False,
                 "zeroshot_backbone": args.zs_backbone,
                 "embedding_backbone": args.embed_backbone},
        config={"steps": args.steps, "lr": args.lr, "backbone": BACKBONE,
                "zeroshot": args.zs_backbone},
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
