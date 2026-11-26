"""Step 1.1 — D3: epistasis headroom audit.

The experiment the project is gated on. If the target landscapes are close to
additive there is no valley to cross, H4 is vacuous, and no architecture fixes it.

Step 0.2 produced a first pass: `1 - additive_r2` on a full-data fit, per
landscape. This replaces those numbers with ones that mean something, because the
first pass conflated three sources of a low additive R^2 — epistasis, measurement
noise, and floor censoring — and on TEV the figure was almost entirely noise.

What this measures, per landscape:

    order decomposition   out-of-fold R^2 at interaction order 1, 2, 3
    noise ceiling         the highest R^2 the assay's own error bars permit
    high-confidence fit   the same decomposition on above-median read depth
    greedy reachability   how often hill-climbing with restarts finds the optimum
    path accessibility    fraction of monotone shortest paths to the optimum
    reciprocal sign       count of the traps greedy provably cannot cross
    stack-top-k headroom  what the forty-year-old heuristic already captures

Gate metrics, pre-registered in PREREGISTRATION.md and unchanged here:

    nonadditive_variance_fraction >= 0.15   aggregate max   (existential)
    greedy_optimum_hit_rate       <= 0.80   aggregate min   (existential)

Both keep the Step 0.2 definition — a full-data additive fit on the raw
landscape — so the gate is judged against the numbers it was registered against.
Everything else here is additional reporting.

Run from the repo root:

    python -I scripts/epistasis_audit.py                # all landscapes
    python -I scripts/epistasis_audit.py --only TEV TrpB4
    python -I scripts/epistasis_audit.py --no-track     # no tracker runs
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from biodreamer import tracking
from biodreamer.eval import metrics as M
from biodreamer.eval.baselines import AdditiveNull, CountingOracle, GreedyWalk, StackTopK
from biodreamer.eval.decompose import decompose, noise_ceiling
from biodreamer.eval.landscape import Landscape

SSMULA = Path("data/raw/ssmula/extracted/data")
OUT_JSON = Path("data/processed/epistasis_audit.json")

# Restart counts for the greedy reachability curve. 5 is the pre-registered
# definition of "greedy" (PREREGISTRATION.md); the others show the trend.
RESTART_COUNTS = (1, 5, 20, 100)
SEEDS = tuple(range(10))
GREEDY_MAX_STEPS = 20


@dataclass
class Spec:
    """One landscape, and which published columns carry its confidence signal."""

    name: str
    path: Path
    reads_col: str | None = None  # depth, used for the high-confidence subset
    variance_col: str | None = None  # per-variant variance of the fitness estimate
    control: bool = False
    note: str = ""


def registry() -> list[Spec]:
    trpb = SSMULA / "TrpB/fitness_landscape"
    specs = [
        Spec(
            "TrpB4",
            trpb / "TrpB4.csv",
            note="4 sites, 0% dead, no published error bars",
        ),
        Spec(
            "TEV",
            SSMULA / "TEV/fitness_landscape/TEV.csv",
            reads_col="Number of Reads",
            variance_col="Fitness Variance",
            note="4 sites, primary landscape as registered in Step 0.5",
        ),
        Spec(
            "GB1",
            SSMULA / "GB1/fitness_landscape/GB1.csv",
            reads_col="Count input",
            note="4 sites, 92% dead; input count is independent of the measured fitness",
        ),
        Spec(
            "T7",
            SSMULA / "T7/fitness_landscape/T7.csv",
            reads_col="Number of Reads",
            variance_col="Fitness Variance",
            note="3 sites, median 11 reads per variant",
        ),
        Spec("DHFR", SSMULA / "DHFR/fitness_landscape/DHFR.csv", note="9 nucleotide sites"),
    ]
    specs += [
        Spec(f"TrpB3{c}", trpb / f"TrpB3{c}.csv", note="3 sites")
        for c in "ABCDEFGHI"
    ]
    specs += [
        Spec(
            "ParD2",
            SSMULA / "ParD2/fitness_landscape/ParD2.csv",
            control=True,
            note="negative control: near-additive",
        ),
        Spec(
            "ParD3",
            SSMULA / "ParD3/fitness_landscape/ParD3.csv",
            control=True,
            note="negative control: near-additive",
        ),
    ]
    return specs


# ---- loading ----


def _aux(spec: Spec, landscape: Landscape) -> pd.DataFrame | None:
    """Per-sequence read depth and measurement variance, aligned to the landscape.

    `Landscape.from_csv` averages replicate rows of the same genotype, so the
    confidence columns are aggregated the same way: depth sums, variance averages.
    """
    if spec.reads_col is None and spec.variance_col is None:
        return None
    df = pd.read_csv(spec.path)
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}
    seq_col = next(
        cols[a]
        for a in ("variants", "variant", "aas", "sequences", "sequence", "seq", "genotype")
        if a in cols
    )
    out = pd.DataFrame({"seq": df[seq_col].astype(str).str.strip().str.upper()})
    agg: dict[str, str] = {}
    if spec.reads_col:
        out["reads"] = pd.to_numeric(df[spec.reads_col], errors="coerce")
        agg["reads"] = "sum"
    if spec.variance_col:
        out["mvar"] = pd.to_numeric(df[spec.variance_col], errors="coerce")
        agg["mvar"] = "mean"
    grouped = out.groupby("seq", as_index=True).agg(agg)
    return grouped.reindex(landscape.sequences())


def load(spec: Spec) -> tuple[Landscape, pd.DataFrame | None]:
    ls = Landscape.from_csv(spec.path, spec.name)
    return ls, _aux(spec, ls)


# ---- per-landscape measurement ----


def dead_fraction(y: np.ndarray) -> float:
    """Fraction sitting within 1% of the landscape's span above its minimum."""
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    return float((y <= lo + 0.01 * span).mean()) if span > 0 else float("nan")


def additive_fit(ls: Landscape, aux: pd.DataFrame | None) -> dict[str, float]:
    """The gate metric, plus the three things that explain it."""
    y = ls.fitness
    full = AdditiveNull().fit(ls)
    r2_full = full.r2(ls)

    singles = AdditiveNull().fit_on_singles(ls)
    high = ls.order() >= 2
    sp_high = (
        M.spearman(y[high], singles.predict(ls.variants[high]))
        if high.sum() >= 2
        else float("nan")
    )

    out = {
        "n_genotypes": float(ls.n),
        "n_sites": float(ls.n_sites),
        "alphabet_size": float(len(ls.alphabet)),
        "additive_r2_full": r2_full,
        # the pre-registered gate metric, definition unchanged from Step 0.2
        "nonadditive_variance_fraction": 1.0 - r2_full,
        # fitted on wild type and singles only, scored on order>=2: the honest
        # question of whether single-mutant effects extrapolate
        "additive_spearman_higher_order": sp_high,
        "frac_order_ge2": float(high.mean()),
        "frac_dead": dead_fraction(y),
        "frac_above_wild_type": float((y > ls.wild_type_fitness()).mean()),
    }

    # the same figure with the floor removed: on GB1, 92% of variants are at the
    # detection limit, and a floored distribution is not epistasis
    alive = y > float(y.min()) + 0.01 * (float(y.max()) - float(y.min()))
    if 0 < alive.sum() < ls.n and alive.sum() > 50:
        sub = ls.subset(alive, "alive")
        out["nonadditive_variance_fraction_alive"] = 1.0 - AdditiveNull().fit(sub).r2(sub)
        out["n_alive"] = float(alive.sum())

    if aux is not None and "mvar" in aux:
        out |= {k: v for k, v in noise_ceiling(y, aux["mvar"].to_numpy()).items()}
    return out


def pairwise_fit(ls: Landscape, aux: pd.DataFrame | None, seed: int) -> dict[str, float]:
    """Out-of-fold R^2 by interaction order — epistasis or noise, decided."""
    dec = decompose(ls.variants, ls.fitness, ls.alphabet, ls.name, max_order=3, seed=seed)
    out = dec.as_metrics()
    out["max_order_fitted"] = float(dec.max_order)

    # repeat on above-median read depth: if the increment grows, the full-landscape
    # figure was noise-limited rather than epistasis-limited
    if aux is not None and "reads" in aux:
        reads = aux["reads"].to_numpy()
        thr = float(np.nanmedian(reads))
        keep = np.nan_to_num(reads, nan=-1.0) >= thr
        if keep.sum() > 500:
            sub = ls.subset(keep, "hiconf")
            dec_hi = decompose(
                sub.variants, sub.fitness, sub.alphabet, sub.name, max_order=2, seed=seed
            )
            out |= {
                "hiconf_read_threshold": thr,
                "n_hiconf": float(keep.sum()),
                "hiconf_cv_r2_additive": dec_hi.additive,
                "hiconf_cv_r2_pairwise": dec_hi.r2_by_order.get(2, float("nan")),
                "hiconf_pairwise_increment": dec_hi.pairwise_increment,
            }
    return out


def greedy_reachability(ls: Landscape) -> dict[str, float]:
    """How often hill-climbing reaches the global optimum, and at what oracle cost.

    The cost is recorded because H4 is pre-registered as a margin over the
    strongest baseline at B = 100 true evaluations, while "greedy" is
    pre-registered as 5 restarts of up to 20 steps. On a four-site landscape over
    20 residues each step evaluates 76 neighbours, so the two definitions differ
    by orders of magnitude and cannot both hold.
    """
    out: dict[str, float] = {"n_seeds": float(len(SEEDS))}
    for r in RESTART_COUNTS:
        hits, calls = [], []
        for s in SEEDS:
            oracle = CountingOracle(ls)
            proposals = GreedyWalk(
                max_steps=GREEDY_MAX_STEPS, restarts=r, seed=s
            ).propose(ls, 10**6, oracle=oracle)
            hits.append(M.global_optimum_hit_rate(proposals, ls))
            calls.append(oracle.distinct)
        rate = float(np.mean(hits))
        out[f"greedy_hit_rate_restarts_{r}"] = rate
        out[f"greedy_oracle_calls_restarts_{r}"] = float(np.mean(calls))
        if r == 5:  # the pre-registered definition of "greedy"
            out["greedy_optimum_hit_rate"] = rate
            out["greedy_oracle_calls"] = float(np.mean(calls))
    out["path_accessibility"] = M.path_accessibility(ls)
    out["n_singles_available"] = float(ls.singles().n - 1)
    return out


def stack_topk_headroom(ls: Landscape) -> dict[str, float]:
    """What stacking the best single substitutions already achieves."""
    wt = ls.wild_type_fitness()
    _, best = ls.global_optimum()
    gain = best - wt

    oracle = CountingOracle(ls)
    proposals = StackTopK(k=ls.n_sites).propose(ls, budget=ls.n_sites, oracle=oracle)
    stacked = M.best_fitness_found(proposals, ls)
    null = AdditiveNull().fit_on_singles(ls)
    return {
        # what each baseline costs in true evaluations, in the same currency as the
        # pre-registered B = 100 budget
        "stack_topk_oracle_calls": float(oracle.distinct),
        "additive_null_fitting_cost": float(ls.singles().n),
        "stack_topk_gain_fraction": float((stacked - wt) / gain) if gain > 0 else float("nan"),
        "stack_topk_hit_rate": M.global_optimum_hit_rate(proposals, ls),
        "additive_null_hit_rate_b100": M.global_optimum_hit_rate(null.propose(ls, 100), ls),
        "additive_null_gain_fraction_b100": (
            float((M.best_fitness_found(null.propose(ls, 100), ls) - wt) / gain)
            if gain > 0
            else float("nan")
        ),
        "wild_type_fitness": wt,
        "global_optimum_fitness": best,
    }


def rse_count(ls: Landscape) -> dict[str, float]:
    """Reciprocal sign epistasis: both singles down, the double up. The trap."""
    rep = M.reciprocal_sign_epistasis(ls)
    return {
        "rse_pairs": float(rep.n_reciprocal_sign),
        "rse_pairs_examined": float(rep.n_pairs),
        "rse_fraction": rep.fraction,
    }


EXPERIMENTS = {
    "additive_fit": lambda ls, aux, seed: additive_fit(ls, aux),
    "pairwise_fit": lambda ls, aux, seed: pairwise_fit(ls, aux, seed),
    "greedy_reachability": lambda ls, aux, seed: greedy_reachability(ls),
    "stack_topk_headroom": lambda ls, aux, seed: stack_topk_headroom(ls),
    "rse_count": lambda ls, aux, seed: rse_count(ls),
}


def run_one(spec: Spec, seed: int, track: bool) -> dict:
    ls, aux = load(spec)
    rec: dict = {"name": spec.name, "control": spec.control, "note": spec.note}
    for exp, fn in EXPERIMENTS.items():
        t0 = time.time()
        vals = fn(ls, aux, seed)
        rec[exp] = vals
        rec.setdefault("timing_s", {})[exp] = round(time.time() - t0, 1)
        if track:
            # controls are logged under their own experiment id so a reader cannot
            # mistake a near-additive control for evidence of headroom
            exp_id = "negative_control" if spec.control else exp
            with tracking.run(
                "1.1",
                exp_id,
                variant={"landscape": spec.name, "measurement": exp},
                datasets=[str(spec.path)],
                seed=seed,
                notes=spec.note,
            ) as r:
                r.log_metrics({k: v for k, v in vals.items() if np.isfinite(v)})
        print(f"  {spec.name:<8} {exp:<20} {rec['timing_s'][exp]:>6.1f}s", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="landscape names to run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    specs = registry()
    if args.only:
        want = {s.lower() for s in args.only}
        specs = [s for s in specs if s.name.lower() in want]
        if not specs:
            print("no landscape matched --only")
            return 1

    results = []
    if OUT_JSON.exists() and args.only:
        results = [r for r in json.loads(OUT_JSON.read_text()) if r["name"] not in {s.name for s in specs}]

    for spec in specs:
        if not spec.path.exists():
            print(f"  {spec.name:<8} MISSING {spec.path}")
            continue
        results.append(run_one(spec, args.seed, not args.no_track))
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\nwrote {OUT_JSON}  ({len(results)} landscapes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
