"""Step 2.4 — `vs_literature` on a MATCHED assay list and a matched protocol.

Step 2.2 recorded this as "not done rather than estimated", and the re-scope promoted it
from an unchecked box to a **precondition**. The reason is simple and it invalidates every
comparison made so far: every published figure is a mean over *its own* assay list, and
+0.409 is a mean over *ours*. An unknown part of the shortfall is therefore assay
difficulty rather than method quality, and until that is separated **neither a pass nor a
fail of H1 is attributable to the method**.

This closes it by running our estimators on the 5 multi-mutant ProteinGym datasets the
fine-tuning literature benchmarks, under its protocol: n sequences sampled at random for
training, 5,000 more for test, averaged over 3 seeds. Two of those datasets (GRB2, PABP)
top out at order 2 and so were never in this project's corpus -- which is exactly the kind
of difference that makes assay-list means incomparable.

**The OHE row is the control that validates the harness, and it is the point of including
a baseline we do not care about.** The published one-hot number is 0.473 at n = 128. If our
OHE lands there, the protocol is implemented correctly and the other rows can be read; if
it does not, nothing else on the table means anything and the discrepancy is the result.

Note what this protocol is NOT. It is a *random* split over all mutation orders, so
multi-mutants appear in training. H1's protocol trains on order <= 2 and tests on order >=
3, which is strictly harder. Numbers here are therefore not comparable with H1's +0.409 and
are not presented as such; they answer one question only, which is whether this project's
estimator is where the literature is when both are measured on the same thing.

    python -I scripts/matched_literature.py --dry-run
    python -I scripts/matched_literature.py
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
from sklearn.linear_model import RidgeCV

from biodreamer import tracking
from biodreamer.protein_dreamer.cache import sequence_hash
from biodreamer.reward.data import (
    LITERATURE_MULTIMUTANT,
    LITERATURE_SEEDS,
    LITERATURE_TRAIN_SIZES,
    literature_split,
    load_assay,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode_positional import resolve_wild_type  # noqa: E402
from gp_reward import ExactGP, fit  # noqa: E402
from h1_benchmark import substitution_design  # noqa: E402
from zeroshot_baseline import matrix_path, score_from_matrix  # noqa: E402

CACHE = Path("data/processed/pooled_cache")
OUT = Path("data/processed/matched_literature.json")
ALPHAS = np.logspace(-1, 5, 13)

# Multi-mutant Spearman from bioRxiv 2024.05.28.596156v4, Tables 4 and 5, on THESE 5
# datasets under THIS protocol. Cited, never recomputed. `Kermut` there is Kermut-sequence
# -- the paper states it is run without a structure model -- which is the configuration
# scripts/gp_reward.py reproduces, so `gp_zeroshot` is the row to read against it.
PUBLISHED = {
    "Kermut-sequence":            {24: 0.397, 32: 0.426, 48: 0.504, 128: 0.614, 512: 0.724},
    "ProteinNPT (MSAT)":          {24: 0.505, 32: 0.511, 48: 0.591, 128: 0.696, 512: 0.791},
    "Emb. aug. (MSAT)":           {24: 0.577, 32: 0.581, 48: 0.614, 128: 0.696, 512: 0.764},
    "OHE aug. (MSAT)":            {24: 0.607, 32: 0.616, 48: 0.622, 128: 0.684, 512: 0.763},
    "OHE":                        {24: 0.255, 32: 0.268, 48: 0.314, 128: 0.473, 512: 0.664},
    "ESM-2 linear head, ranking": {24: 0.429, 32: 0.447, 48: 0.503, 128: 0.648, 512: 0.773},
    "ESM-2 fine-tuned, ranking":  {24: 0.577, 32: 0.593, 48: 0.603, 128: 0.651, 512: 0.758},
}


def rho(pred: np.ndarray, truth: np.ndarray) -> float:
    r = spearmanr(pred, truth).statistic
    return 0.0 if np.isnan(r) else float(r)


def ridge_rho(xtr, ytr, xte, truth) -> float:
    m = RidgeCV(alphas=ALPHAS).fit(xtr, ytr)
    return rho(m.predict(xte), truth)


def gp_rho(xtr, ytr, f0tr, xte, f0te, truth, zero_shot_mean, steps, lr, seed) -> float:
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
    xt = torch.tensor((xtr - mu) / sd)
    xv = torch.tensor((xte - mu) / sd)
    ys = torch.tensor((ytr - ytr.mean()) / (ytr.std() + 1e-12))
    fm, fs = f0tr.mean(), f0tr.std() + 1e-12
    ftr = torch.tensor((f0tr - fm) / fs)
    fte = torch.tensor((f0te - fm) / fs)
    gp = fit(xt, ys, ftr, zero_shot_mean, steps, lr, seed)
    return rho(gp.predict(xt, ys, ftr, [(xv, fte)]), truth)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="saprot", choices=["saprot", "esm2"],
                    help="backbone for the EMBEDDINGS the kernel runs on")
    ap.add_argument("--zs-backbone", default=None, choices=["saprot", "esm2"],
                    help="backbone for f0, the zero-shot prior mean. Defaults to "
                         "--backbone, which is how this script ran originally and is "
                         "what Kermut specifies. Setting it separately de-confounds the "
                         "two: on H1's protocol the representation and the zero-shot "
                         "scorer prefer DIFFERENT backbones, so a single --backbone flag "
                         "cannot attribute a difference to either one.")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    zs_bb = args.zs_backbone or args.backbone
    bb = {"saprot": "westlake-repl_SaProt_650M_AF2",
          "esm2": "facebook_esm2_t33_650M_UR50D"}[args.backbone]
    mat_file = matrix_path(zs_bb)
    if not args.dry_run and not mat_file.exists():
        raise FileNotFoundError(
            f"{mat_file} missing -- run scripts/zeroshot_baseline.py "
            f"--save-logprob-matrix --backbones {zs_bb}"
        )
    global OUT
    if args.zs_backbone and args.zs_backbone != args.backbone:
        OUT = OUT.with_name(f"matched_literature_emb-{args.backbone}_zs-{zs_bb}.json")
    elif args.backbone != "saprot":
        OUT = OUT.with_name(f"matched_literature_{args.backbone}.json")
    mats = np.load(mat_file) if not args.dry_run else None

    methods = ["ohe", "ohe_aug", "emb_ridge", "emb_aug", "gp_constant", "gp_zeroshot"]
    # method -> n -> list of per-(dataset, seed) rho
    acc: dict[str, dict[int, list[float]]] = {m: {n: [] for n in LITERATURE_TRAIN_SIZES}
                                              for m in methods}
    per_dataset: dict[str, dict] = {}

    for name in LITERATURE_MULTIMUTANT:
        a = load_assay(name)
        f = CACHE / f"{bb}__{name}.npz"
        if args.dry_run:
            print(f"{name:34s} {len(a.sequence):>7,} variants  cache={'OK' if f.exists() else 'MISSING'}")
            continue
        wt = resolve_wild_type(a, name)
        f0_all = score_from_matrix(a.mutant, mats[name], wt)
        z = np.load(f)
        row = {k: i for i, k in enumerate(z["keys"])}
        emb = z["emb"]
        miss = [s for s in a.sequence if sequence_hash(s) not in row]
        if miss:
            print(f"  {name}: {len(miss):,} sequences absent from the cache")
        per_dataset[name] = {}

        for seed in LITERATURE_SEEDS:
            _, te = literature_split(len(a.sequence), seed)
            xte = emb[[row[sequence_hash(s)] for s in a.sequence[te]]].astype(np.float64)
            truth = a.score[te]
            for n in LITERATURE_TRAIN_SIZES:
                tr, _ = literature_split(len(a.sequence), seed, n)
                xtr = emb[[row[sequence_hash(s)] for s in a.sequence[tr]]].astype(np.float64)
                ytr = a.score[tr]
                f0tr, f0te = f0_all[tr], f0_all[te]
                t0 = time.time()

                # one-hot over substitutions observed in THIS training set; an unseen
                # substitution contributes zero, the honest additive prediction
                dtr, vocab = substitution_design(a.mutant[tr])
                dte, _ = substitution_design(a.mutant[te], vocab)
                otr, ote = dtr.toarray(), dte.toarray()
                r = {
                    "ohe": ridge_rho(otr, ytr, ote, truth),
                    "ohe_aug": ridge_rho(
                        np.hstack([otr, f0tr.reshape(-1, 1)]), ytr,
                        np.hstack([ote, f0te.reshape(-1, 1)]), truth),
                    "emb_ridge": ridge_rho(xtr, ytr, xte, truth),
                    "emb_aug": ridge_rho(
                        np.hstack([xtr, f0tr.reshape(-1, 1)]), ytr,
                        np.hstack([xte, f0te.reshape(-1, 1)]), truth),
                    "gp_constant": gp_rho(xtr, ytr, f0tr, xte, f0te, truth, False,
                                          args.steps, args.lr, seed),
                    "gp_zeroshot": gp_rho(xtr, ytr, f0tr, xte, f0te, truth, True,
                                          args.steps, args.lr, seed),
                }
                for m, v in r.items():
                    acc[m][n].append(v)
                per_dataset[name].setdefault(n, []).append(r)
                print(f"  {name:28s} seed {seed} n={n:<4d} "
                      + "  ".join(f"{m}={v:+.3f}" for m, v in r.items())
                      + f"  ({time.time() - t0:.0f}s)", flush=True)
        z.close()
    if args.dry_run:
        return 0

    print(f"\n{'method':30s}" + "".join(f"{n:>9d}" for n in LITERATURE_TRAIN_SIZES))
    summary = {}
    for m in methods:
        summary[m] = {n: float(np.mean(acc[m][n])) for n in LITERATURE_TRAIN_SIZES}
        print(f"{m:30s}" + "".join(f"{summary[m][n]:>+9.3f}" for n in LITERATURE_TRAIN_SIZES))
    print(f"\n{'PUBLISHED (their numbers)':30s}")
    for k, v in PUBLISHED.items():
        print(f"{k:30s}" + "".join(f"{v[n]:>+9.3f}" for n in LITERATURE_TRAIN_SIZES))

    # the harness check: our OHE against their OHE, same protocol, same datasets
    ohe_gap = {n: summary["ohe"][n] - PUBLISHED["OHE"][n] for n in LITERATURE_TRAIN_SIZES}
    kermut_gap = {n: summary["gp_zeroshot"][n] - PUBLISHED["Kermut-sequence"][n]
                  for n in LITERATURE_TRAIN_SIZES}
    print(f"\n{'OHE control gap (ours - theirs)':30s}"
          + "".join(f"{ohe_gap[n]:>+9.3f}" for n in LITERATURE_TRAIN_SIZES))
    print(f"{'gp_zeroshot - Kermut-sequence':30s}"
          + "".join(f"{kermut_gap[n]:>+9.3f}" for n in LITERATURE_TRAIN_SIZES))

    metrics = {f"{m}/n{n}": summary[m][n] for m in methods for n in LITERATURE_TRAIN_SIZES}
    metrics["ohe_control_gap_n128"] = ohe_gap[128]
    metrics["gp_minus_kermut_n128"] = kermut_gap[128]
    OUT.write_text(json.dumps(
        {"backbone": args.backbone, "zs_backbone": zs_bb,
         "protocol": "literature random split, 5000 test",
         "seeds": list(LITERATURE_SEEDS), "train_sizes": list(LITERATURE_TRAIN_SIZES),
         "summary": summary, "per_dataset": per_dataset, "published": PUBLISHED,
         "ohe_control_gap": ohe_gap, "gp_minus_kermut": kermut_gap}, indent=2) + "\n")
    print(f"\nwrote {OUT}")

    with tracking.run(
        "2.4", "matched_assay_literature",
        variant={"backbone": args.backbone, "zs_backbone": zs_bb,
                 "protocol": "literature_random_split"},
        config={"steps": args.steps, "lr": args.lr,
                "train_sizes": list(LITERATURE_TRAIN_SIZES)},
        seed=0, datasets=[str(CACHE), str(mat_file)],
        notes=("vs_literature on a matched assay list AND a matched protocol. The OHE row "
               "is the harness control: the published value is 0.473 at n=128, and if ours "
               "does not land there the rest of the table is uninterpretable. This is a "
               "RANDOM split with multi-mutants in training, so it is NOT comparable with "
               "H1's order<=2 -> order>=3 extrapolation and is not presented as such."),
    ) as r:
        r.log_metrics(metrics)
        r.log_artifact(str(OUT), kind="table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
