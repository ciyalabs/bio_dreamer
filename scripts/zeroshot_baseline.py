"""Step 2.2 — `vs_zeroshot`, and the component every leading method has and we lack.

H1 failed at +0.409 against a registered 0.55, and the recorded diagnosis was that the
frozen encoder's *representation* is the ceiling. The literature says otherwise. On this
exact protocol (<300 order<=2 for training, unseen order>=3 for test) Kermut scores 0.62
from **mean-pooled** frozen embeddings -- the same input this project feeds its head --
and ProteinNPT 0.56. What those methods have that this one does not is a **zero-shot
likelihood**: Kermut uses it as the GP prior mean, ProteinNPT as an input feature. This
project has never measured one.

So before any fine-tune is costed, measure the free thing:

1. **The zero-shot score alone**, as a ranker on H1's own test set.
2. **The zero-shot score as a feature**, added to the frozen ridge that already ties with
   the whole reward head -- the cheapest form of Kermut's prior mean.

**It is cheap because the scoring rule is additive over positions.** Masked marginals need
one forward pass per *mutated position*, not per variant: 1,201 passes cover all 943,411
test variants, because a substitution's score is read from the wild type's logits at that
one site. No rental, no re-encoding.

**That additivity is also the limitation, and it is the point of the measurement.** This
score cannot represent epistasis: a variant's value is the sum of its substitutions'
values, exactly like `h1_benchmark.additive_null` but with coefficients taken from a
pretrained likelihood instead of fitted on <=300 labels. So it is a direct test of whether
the gap to the literature is a *better additive estimate* or genuinely non-additive
structure. The fine-tuning literature predicts the former matters: at n=128 multi-mutants,
fine-tuning ESM-2 end to end scores 0.651 against 0.648 for a frozen linear head with the
same ranking loss -- +0.003, inside noise -- while the zero-shot rule it is fine-tuning
*towards* is itself additive.

Two scoring rules, because the cheap one is nearly free and the standard one is the
comparable one:

    wt-marginals      one forward pass of the wild type, logits read at every position
    masked-marginals  one pass per mutated position, that position masked (ProteinGym's
                      standard for ESM-family models, and the stronger of the two)

    python -I scripts/zeroshot_baseline.py --dry-run
    python -I scripts/zeroshot_baseline.py
"""

from __future__ import annotations

import argparse
import gc
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
from biodreamer.protein_dreamer.encoder import UNKNOWN_3DI
from biodreamer.reward.data import assays_for, detection_limits, h1_split, load_assay

# `python -I` drops the script's own directory from sys.path, so a sibling script is not
# importable by default -- the same isolation that broke the first pod run. Re-adding it is
# two lines; copying `parse_subs` would be a second definition of the mutant-string format.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode_positional import parse_subs, resolve_wild_type  # noqa: E402

CACHE = Path("data/processed/pooled_cache")
POOLED_BACKBONE = "westlake-repl_SaProt_650M_AF2"
OUT = Path("data/processed/zeroshot_baseline.json")
AA = "ACDEFGHIKLMNPQRSTVWY"
ALPHAS = np.logspace(-1, 5, 13)

BACKBONES = {
    "saprot": "westlake-repl/SaProt_650M_AF2",
    "esm2": "facebook/esm2_t33_650M_UR50D",
}
# PREREGISTRATION.md fixes SaProt as the default and ESM-2 as the comparator. The
# prior-mean test uses the default so that the registered margin is not quietly selected
# on the test set it is scored against.
DEFAULT_BACKBONE = "saprot"

# measured in this project, on these same 11 assays and this same split seed, by
# scripts/h1_benchmark.py. Quoted so the comparison is visible in one table; not recomputed.
OURS = {
    "reward head (depth-aware)": 0.409,
    "frozen ridge + mutation count": 0.390,
    "additive null": 0.324,
    "frozen ridge": 0.301,
}
# published mean Spearman on this same higher-order extrapolation protocol, on THEIR assay
# sets, as recorded in PREREGISTRATION.md. Cited, not recomputed -- see the caveat in the
# report: a mean over a different assay list is not a like-for-like number.
LITERATURE = {"ProteinNPT": 0.56, "MULTI-evolve": 0.61, "Kermut": 0.62, "Cerebra": 0.65}


def aa_token_ids(tok, structure_aware: bool) -> torch.Tensor:
    """Vocabulary ids for the 20 amino acids, in `AA` order.

    SaProt's vocabulary is the product of amino acid and 3Di token, so a sequence-only run
    reads the `aa#` column -- the same unknown-structure token the encoder uses, and the
    only way these log-probs are comparable to the cached embeddings.
    """
    vocab = tok.get_vocab()
    toks = [a + UNKNOWN_3DI if structure_aware else a for a in AA]
    missing = [t for t in toks if t not in vocab]
    if missing:
        raise KeyError(f"tokenizer is missing {missing}")
    return torch.tensor([vocab[t] for t in toks])


def load_backbone(key: str, device: str, dtype: str):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    name = BACKBONES[key]
    tok = AutoTokenizer.from_pretrained(name, cache_dir="models/hf")
    model = AutoModelForMaskedLM.from_pretrained(
        name, cache_dir="models/hf", dtype=getattr(torch, dtype)
    ).eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        device, model = "cpu", model.float()
    return tok, model, device


def encode_wt(tok, wt: str, structure_aware: bool, device: str) -> torch.Tensor:
    text = "".join(a + UNKNOWN_3DI.lower() for a in wt) if structure_aware else wt
    ids = tok(text, return_tensors="pt")["input_ids"]
    if ids.shape[1] != len(wt) + 2:
        raise ValueError(f"expected {len(wt) + 2} tokens for a {len(wt)}-mer, got {ids.shape[1]}")
    return ids.to(device)


def logprobs(
    tok, model, wt: str, positions: list[int], structure_aware: bool, device: str,
    rule: str, batch: int,
) -> dict[int, np.ndarray]:
    """1-based position -> log-probability over the 20 amino acids, in `AA` order."""
    ids = encode_wt(tok, wt, structure_aware, device)
    cols = aa_token_ids(tok, structure_aware).to(device)

    if rule == "wt-marginals":
        with torch.no_grad():
            lg = model(input_ids=ids).logits[0]
        lp = torch.log_softmax(lg[:, cols].float(), dim=-1).cpu().numpy()
        return {p: lp[p] for p in positions}  # token index p == 1-based position p

    out: dict[int, np.ndarray] = {}
    for s in range(0, len(positions), batch):
        chunk = positions[s : s + batch]
        rows = ids.repeat(len(chunk), 1)
        for r, p in enumerate(chunk):
            rows[r, p] = tok.mask_token_id
        with torch.no_grad():
            lg = model(input_ids=rows).logits
        for r, p in enumerate(chunk):
            out[p] = torch.log_softmax(lg[r, p, cols].float(), dim=-1).cpu().numpy()
    return out


def score_variants(mutants: np.ndarray, lp: dict[int, np.ndarray], wt: str) -> np.ndarray:
    """Sum of per-substitution log-odds. Additive by construction -- see the docstring."""
    idx = {a: i for i, a in enumerate(AA)}
    out = np.zeros(len(mutants))
    for k, m in enumerate(mutants):
        tot = 0.0
        for ref, pos, alt in parse_subs(m):
            if wt[pos - 1] != ref:
                raise ValueError(f"{m}: wild type has {wt[pos - 1]} at {pos}, not {ref}")
            tot += float(lp[pos][idx[alt]] - lp[pos][idx[ref]])
        out[k] = tot
    return out


def cache_path(key: str, rule: str) -> Path:
    return Path("data/processed") / f"zeroshot_{key}_{rule.replace('-', '_')}.npz"


def matrix_path(key: str) -> Path:
    """Per-position log-probabilities, which supersede the per-variant score cache.

    The score cache holds one number per variant for ONE split, so it cannot serve a
    different split. The (L, 20) masked-marginal matrix holds everything needed to score
    **any** variant of that protein and is 15 KB, so caching it instead makes the matched
    literature protocol -- which uses random splits over all variants -- free rather than a
    second GPU pass.
    """
    return Path("data/processed") / f"logprob_matrix_{key}.npz"


def score_from_matrix(mutants: np.ndarray, mat: np.ndarray, wt: str) -> np.ndarray:
    """Score variants from a full (L, 20) log-probability matrix."""
    return score_variants(mutants, {p + 1: mat[p] for p in range(mat.shape[0])}, wt)


def rho(pred: np.ndarray, truth: np.ndarray) -> float:
    r = spearmanr(pred, truth).statistic
    return 0.0 if np.isnan(r) else float(r)


def within_order_rho(pred: np.ndarray, truth: np.ndarray, order: np.ndarray) -> float:
    """Mean rho computed at each fixed mutation depth, weighted by group size.

    **This is the number that decides whether a zero-shot score is worth anything here.**
    The scoring rule sums one log-odds term per substitution, and those terms are mostly
    negative, so a deeper variant scores lower *by construction* -- the score is partly a
    mutation counter. Step 2.1 measured that counting mutations alone reaches 0.31 on this
    test set, so a total rho near 0.32 is consistent with a score that has no
    sequence-specific skill at all. At fixed depth, depth carries no information and only
    skill is left.
    """
    num = den = 0.0
    for d in np.unique(order):
        m = order == d
        if m.sum() < 15:  # the order-group floor the Cerebra protocol uses
            continue
        r = spearmanr(pred[m], truth[m]).statistic
        if not np.isnan(r):
            num += r * m.sum()
            den += m.sum()
    return float(num / den) if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="benchmark", choices=["benchmark", "planning"])
    ap.add_argument("--backbones", default="saprot,esm2")
    ap.add_argument("--rules", default="masked-marginals,wt-marginals")
    ap.add_argument("--seed", type=int, default=0, help="split seed; LOCKED to H1's 0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--save-logprob-matrix", action="store_true",
        help="mask EVERY position and cache the (L, 20) matrix per assay, so any split "
             "can be scored later without another GPU pass",
    )
    ap.add_argument(
        "--assays", default="",
        help="comma-separated assay names, overriding the instance list (for the matched "
             "literature protocol, whose datasets are not all in this corpus)",
    )
    args = ap.parse_args()

    names = args.assays.split(",") if args.assays else assays_for(args.instance)
    limits = detection_limits()
    rules = args.rules.split(",")
    keys = args.backbones.split(",")

    if args.save_logprob_matrix:
        for key in keys:
            dest = matrix_path(key)
            have = dict(np.load(dest)) if dest.exists() else {}
            tok = model = None
            device, sa = args.device, key == "saprot"
            for n in names:
                if n in have:
                    print(f"  {key:7s} {n:30s} cached ({have[n].shape})")
                    continue
                a = load_assay(n, limits)
                wt = resolve_wild_type(a, n)
                if model is None:
                    tok, model, device = load_backbone(key, args.device, args.dtype)
                t0 = time.time()
                lp = logprobs(tok, model, wt, list(range(1, len(wt) + 1)), sa, device,
                              "masked-marginals", args.batch)
                have[n] = np.stack([lp[p] for p in range(1, len(wt) + 1)]).astype(np.float32)
                np.savez(dest, **have)  # written per assay, so an interrupt loses one
                print(f"  {key:7s} {n:30s} L={len(wt):4d} "
                      f"{time.time() - t0:.0f}s -> {dest.name}")
            if model is not None:
                model.to("cpu")
                del model, tok
                gc.collect()
                torch.cuda.empty_cache()
        return 0

    loaded = {}
    for n in names:
        a = load_assay(n, limits)
        tr, te = h1_split(a.order, seed=args.seed)
        wt = resolve_wild_type(a, n)
        pos = sorted({p for m in a.mutant[te] for _, p, _ in parse_subs(m)})
        loaded[n] = (a, tr, te, wt, pos)
        print(f"{n:30s} L={len(wt):4d} test={len(te):7,d} positions={len(pos):4d}")
    total = sum(len(v[4]) for v in loaded.values())
    print(f"\n{len(names)} assays, {sum(len(v[2]) for v in loaded.values()):,} test variants, "
          f"{total:,} masked forward passes")
    if args.dry_run:
        return 0

    results: dict[str, dict] = {}
    for key in keys:
        # loaded on the first cache MISS, not up front: a fully cached backbone needs no
        # model at all, and two 650M backbones do not fit on this card at once
        tok = model = None
        device = args.device
        sa = key == "saprot"
        for rule in rules:
            t0 = time.time()
            per: dict[str, float] = {}
            scores: dict[str, np.ndarray] = {}
            scores_tr: dict[str, np.ndarray] = {}
            wo: dict[str, float] = {}
            # the log-probs are a deterministic function of (backbone, rule, wild type), so
            # a re-run reuses them rather than paying for 1,201 forward passes again --
            # the same resume discipline scripts/encode_positional.py uses for the cache
            cached = cache_path(key, rule)
            have = np.load(cached) if cached.exists() else None
            if have is not None and set(have.files) != {f"{n}/{w}" for n in loaded for w in ("tr", "te")}:
                print(f"  {cached.name}: assay list differs from this run -- recomputing")
                have = None
            for n, (a, tr, te, wt, pos) in loaded.items():
                if have is not None:
                    scores[n], scores_tr[n] = have[f"{n}/te"], have[f"{n}/tr"]
                    per[n] = rho(scores[n], a.score[te])
                    wo[n] = within_order_rho(scores[n], a.score[te], a.order[te])
                    print(f"  {key:7s} {rule:16s} {n:30s} {per[n]:+.3f} "
                          f"(within-order {wo[n]:+.3f}) (cached)")
                    continue
                if model is None:
                    tok, model, device = load_backbone(key, args.device, args.dtype)
                lp = logprobs(tok, model, wt, pos, sa, device, rule, args.batch)
                s = score_variants(a.mutant[te], lp, wt)
                scores[n] = s
                # the train rows are needed by the prior-mean test below, and the
                # log-probs are already in hand -- recomputing them later would mean a
                # second pass over the model for nothing
                tr_pos = sorted({p for m in a.mutant[tr] for _, p, _ in parse_subs(m)})
                lp_tr = logprobs(
                    tok, model, wt, [p for p in tr_pos if p not in lp], sa, device,
                    rule, args.batch,
                )
                scores_tr[n] = score_variants(a.mutant[tr], {**lp, **lp_tr}, wt)
                per[n] = rho(s, a.score[te])
                wo[n] = within_order_rho(s, a.score[te], a.order[te])
                print(f"  {key:7s} {rule:16s} {n:30s} {per[n]:+.3f} "
                      f"(within-order {wo[n]:+.3f})")
            vals = np.array(list(per.values()))
            tag = f"{key}/{rule}"
            wov = np.array([wo[n] for n in per])
            results[tag] = {
                "per_assay": per,
                "within_order": wo,
                "within_order_mean": float(wov.mean()),
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "min": float(vals.min()),
                "n_positive": int((vals > 0).sum()),
                "seconds": round(time.time() - t0, 1),
            }
            print(f"  == {tag}: mean {vals.mean():+.3f}  median {np.median(vals):+.3f}  "
                  f"within-order {wov.mean():+.3f}  ({time.time() - t0:.0f}s)")
            if have is None:
                np.savez(
                    cache_path(key, rule),
                    **{f"{n}/te": v for n, v in scores.items()},
                    **{f"{n}/tr": v for n, v in scores_tr.items()},
                )
        # a 3.6 GiB card with a desktop session on it cannot hold two 650M backbones, and
        # `del` alone leaves the weights resident until the allocator is told to release
        if model is not None:
            model.to("cpu")
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    # ---- the prior-mean test: does the zero-shot score add to the frozen features? ----
    # read from the cache rather than from this run's results, so the two backbones may be
    # run in separate processes -- which on this card they must be
    want = {f"{n}/{w}" for n in loaded for w in ("tr", "te")}
    avail = {}
    for key in BACKBONES:
        f = cache_path(key, "masked-marginals")
        if not f.exists():
            continue
        z = np.load(f)
        if set(z.files) != want:
            continue
        avail[key] = float(
            np.mean([rho(z[f"{n}/te"], loaded[n][0].score[loaded[n][2]]) for n in loaded])
        )
    if not avail:
        print("\nno complete zero-shot cache -- skipping the prior-mean test")
        return 0
    # DEFAULT_BACKBONE, never `max(avail)`. Picking the backbone that scores best on the
    # order>=3 set would select a pre-registered quantity on its own test set; SaProt is
    # the default fixed in PREREGISTRATION.md and ESM-2 is the registered comparator, so
    # the choice is already made and is not ours to make here.
    zs_key = DEFAULT_BACKBONE if DEFAULT_BACKBONE in avail else sorted(avail)[0]
    if zs_key != DEFAULT_BACKBONE:
        print(f"\nWARNING: {DEFAULT_BACKBONE} has no complete cache; falling back to {zs_key}")
    print(f"\nprior-mean test uses {zs_key} masked-marginals (the pre-specified default, "
          f"mean {avail[zs_key]:+.3f}); all backbones: "
          f"{({k: round(v, 3) for k, v in avail.items()})}")
    zs = np.load(cache_path(zs_key, "masked-marginals"))
    combo: dict[str, dict[str, float]] = {}
    for n, (a, tr, te, wt, pos) in loaded.items():
        z = np.load(CACHE / f"{POOLED_BACKBONE}__{n}.npz")
        emb = {k: i for i, k in enumerate(z["keys"])}

        def rows(idx):
            return z["emb"][[emb[sequence_hash(s)] for s in a.sequence[idx]]].astype(np.float64)

        def predict_chunked(model, extra_te, size: int = 20000) -> np.ndarray:
            """Predict the test split in blocks instead of one matrix.

            HIS7 has 494,494 test rows and 494,494 x 1280 in float64 is **5 GB**; the
            hstack needed to append the order and zero-shot columns makes a second copy.
            Building both at once is what exhausted this machine's memory.
            """
            out = []
            for s0 in range(0, len(te), size):
                blk = te[s0 : s0 + size]
                cols = [rows(blk)] + [e[s0 : s0 + size] for e in extra_te]
                out.append(model.predict(np.hstack(cols)))
            return np.concatenate(out)

        xtr = rows(tr)
        otr = a.order[tr].reshape(-1, 1).astype(float)
        ote = a.order[te].reshape(-1, 1).astype(float)
        ztr = zs[f"{n}/tr"].reshape(-1, 1)
        zte = zs[f"{n}/te"].reshape(-1, 1)
        y = a.score[tr]
        base = RidgeCV(alphas=ALPHAS).fit(np.hstack([xtr, otr]), y)
        with_zs = RidgeCV(alphas=ALPHAS).fit(np.hstack([xtr, otr, ztr]), y)
        p_base = predict_chunked(base, [ote])
        p_zs = predict_chunked(with_zs, [ote, zte])
        # within-order rho for the ridge too, so the comparison against the GP is on the
        # depth-free diagnostic and not only on a total that a mutation counter inflates.
        # Without this the GP looks like +0.038 over the ridge; the skill gap is the number
        # that matters and it cannot be read off the totals.
        combo[n] = {
            "ridge_count": rho(p_base, a.score[te]),
            "ridge_count_zeroshot": rho(p_zs, a.score[te]),
            "ridge_count_within_order": within_order_rho(p_base, a.score[te], a.order[te]),
            "ridge_count_zeroshot_within_order": within_order_rho(
                p_zs, a.score[te], a.order[te]
            ),
        }
        # SENSITIVITY, not an alternative to be selected on: `masked-marginals` is
        # ProteinGym's standard for ESM-family models and is the pre-specified rule. The
        # prediction being checked is that wt-marginals adds LESS despite scoring higher
        # standalone, because its extra total rho is depth (within-order is the same to
        # 0.006) and the ridge already has an `order` column.
        alt = cache_path(zs_key, "wt-marginals")
        if alt.exists():
            za = np.load(alt)
            if f"{n}/te" in za.files:
                w = RidgeCV(alphas=ALPHAS).fit(
                    np.hstack([xtr, otr, za[f"{n}/tr"].reshape(-1, 1)]), y
                )
                combo[n]["ridge_count_zeroshot_wt"] = rho(
                    predict_chunked(w, [ote, za[f"{n}/te"].reshape(-1, 1)]),
                    a.score[te],
                )
        print(f"  prior-mean {n:30s} {combo[n]['ridge_count']:+.3f} -> "
              f"{combo[n]['ridge_count_zeroshot']:+.3f}")

    base_mean = float(np.mean([v["ridge_count"] for v in combo.values()]))
    zs_mean = float(np.mean([v["ridge_count_zeroshot"] for v in combo.values()]))
    margin = zs_mean - base_mean
    deltas = {n: v["ridge_count_zeroshot"] - v["ridge_count"] for n, v in combo.items()}
    dv = np.array(list(deltas.values()))
    # the mean is the registered quantity, but it is not the honest summary if a couple of
    # assays carry it, so the concentration is reported next to it rather than left to a
    # reader of the per-assay table
    carried = sorted(deltas, key=lambda n: -deltas[n])[:2]
    rest = float(np.mean([deltas[n] for n in deltas if n not in carried]))
    print(f"\nridge + count            mean {base_mean:+.3f}")
    print(f"ridge + count + zeroshot mean {zs_mean:+.3f}")
    print(f"zeroshot_feature_margin       {margin:+.3f}  (step 2.4 threshold: > 0.05)")
    print(f"  median delta                {np.median(dv):+.3f}")
    print(f"  improved >0.02              {(dv > 0.02).sum()}/{len(dv)}; "
          f"hurt >0.02 {(dv < -0.02).sum()}/{len(dv)}")
    print(f"  top two assays             {carried}")
    print(f"  mean delta over the rest    {rest:+.3f}")
    for k in ("ridge_count_within_order", "ridge_count_zeroshot_within_order"):
        print(f"  {k:45s} {np.mean([v[k] for v in combo.values()]):+.3f}")
    alt_vals = [v["ridge_count_zeroshot_wt"] for v in combo.values()
                if "ridge_count_zeroshot_wt" in v]
    alt_mean = float(np.mean(alt_vals)) if len(alt_vals) == len(combo) else float("nan")
    if np.isfinite(alt_mean):
        print(f"  SENSITIVITY wt-marginals as the feature instead: mean {alt_mean:+.3f} "
              f"(margin {alt_mean - base_mean:+.3f})")

    payload = {
        "instance": args.instance,
        "seed": args.seed,
        "assays": names,
        "zeroshot": results,
        "prior_mean": combo,
        "zeroshot_backbone_for_prior_mean": zs_key,
        "zeroshot_means_all_backbones": avail,
        "ours": OURS,
        "literature": LITERATURE,
        "metrics": {
            "ridge_count_mean": base_mean,
            "ridge_count_zeroshot_mean": zs_mean,
            "zeroshot_feature_margin": margin,
            "zeroshot_feature_margin_median": float(np.median(dv)),
            "zeroshot_feature_margin_excluding_top_two": rest,
            "zeroshot_feature_margin_wt_marginals_sensitivity": alt_mean - base_mean,
            "ridge_count_within_order_mean": float(
                np.mean([v["ridge_count_within_order"] for v in combo.values()])
            ),
            "ridge_count_zeroshot_within_order_mean": float(
                np.mean([v["ridge_count_zeroshot_within_order"] for v in combo.values()])
            ),
        },
        "delta_per_assay": deltas,
        "carried_by": carried,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")

    with tracking.run(
        "2.4", "zeroshot_feature",
        variant={"instance": args.instance, "zeroshot": zs_key, "rule": "masked-marginals"},
        config={"alphas": "logspace(-1,5,13)", "features": "pooled ++ order ++ zeroshot"},
        seed=args.seed,
        datasets=[str(CACHE)],
        notes=(
            "Step 2.4. The component H1 never had: a pretrained likelihood as a separate "
            "term. Masked marginals cost one forward pass per MUTATED POSITION (1,201 for "
            "the corpus), not per variant, because the scoring rule is additive over "
            "positions -- which is also why it cannot represent epistasis and is the right "
            "comparison for the additive null rather than for the head. "
            "zeroshot_feature_margin is the pre-registered quantity: the gain from adding "
            "the zero-shot column to the frozen ridge that already ties the whole head."
        ),
    ) as r:
        r.log_metrics(
            {k: v for k, v in payload["metrics"].items() if np.isfinite(v)}
            | {f"zeroshot_mean/{t}": results[t]["mean"] for t in results}
        )
        r.log_artifact(str(OUT), kind="table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
