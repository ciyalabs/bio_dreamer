"""Step 1.1, path-level half: does valley-crossing get harder with depth?

Every SSMuLA landscape tops out at four mutable sites, so every path-level
question the project can ask of them is capped at horizon 4. The premise behind
multi-step planning is that the difficulty grows with depth — that a five- or
nine-step rearrangement hides optima a one-step-at-a-time method cannot reach.
Nothing in the project has tested that, because nothing had the data.

The HIS3 extraction does: 13.5 million combinatorially complete sub-landscapes
up to dimension 9, from one mutagenesis experiment. Each is a complete 2^d
hypercube, so for each dimension this measures

    variance by order     exact, via the Walsh-Hadamard transform
    greedy reachability    how often hill-climbing finds the best corner
    path accessibility     fraction of monotone shortest paths to it
    reciprocal sign        whether the trap greedy cannot cross is present
    stack-top-k headroom   what the forty-year-old heuristic already gets

and reports it as a function of d. A flat curve would undercut the premise; a
falling one supports it. Either way it is the first depth-resolved evidence the
project has.

Run from the repo root:

    python -I scripts/hypercube_paths.py
    python -I scripts/hypercube_paths.py --segment 140_160 --dims 3 4 5 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biodreamer import tracking
from biodreamer.eval import metrics as M
from biodreamer.eval.baselines import AdditiveNull, CountingOracle, GreedyWalk, StackTopK
from biodreamer.eval.decompose import walsh_variance_by_order
from biodreamer.eval.hypercube import load_genotype_table, sample
from biodreamer.eval.landscape import Landscape

HC = Path("data/processed/hypercubeme")
OUT_JSON = Path("data/processed/hypercube_paths.json")

SEEDS = tuple(range(10))
GREEDY_RESTARTS = 5  # the pre-registered definition
GREEDY_MAX_STEPS = 20
MAX_PATHS = 5_000  # 9! = 362,880 shortest paths at dimension 9; sample instead

# fewer at the deep end, where there are fewer hypercubes and each costs more
SAMPLE_SIZE = {3: 200, 4: 200, 5: 200, 6: 200, 7: 200, 8: 100, 9: 100}


def measure(ls: Landscape) -> dict[str, float]:
    """Everything one hypercube can say, before aggregating over the sample."""
    frac = walsh_variance_by_order(ls.fitness, ls.n_sites)
    hits, calls = [], []
    for s in SEEDS:
        oracle = CountingOracle(ls)
        proposals = GreedyWalk(
            max_steps=GREEDY_MAX_STEPS, restarts=GREEDY_RESTARTS, seed=s
        ).propose(ls, 10**6, oracle=oracle)
        hits.append(M.global_optimum_hit_rate(proposals, ls))
        calls.append(oracle.distinct)

    wt = ls.wild_type_fitness()
    _, best = ls.global_optimum()
    gain = best - wt
    stacked = M.best_fitness_found(StackTopK(k=ls.n_sites).propose(ls, ls.n_sites), ls)
    null = AdditiveNull().fit(ls)

    return {
        "order_1": frac.get(1, float("nan")),
        "order_2": frac.get(2, float("nan")),
        "order_3plus": float(sum(v for k, v in frac.items() if k >= 3)),
        "nonadditive": 1.0 - frac.get(1, float("nan")),
        "additive_r2": null.r2(ls),
        "greedy_hit_rate": float(np.mean(hits)),
        "greedy_oracle_calls": float(np.mean(calls)),
        "path_accessibility": M.path_accessibility(ls, max_paths=MAX_PATHS),
        "rse_pairs": float(M.reciprocal_sign_epistasis(ls).n_reciprocal_sign),
        "stack_topk_gain_fraction": float((stacked - wt) / gain) if gain > 0 else float("nan"),
        "wt_is_best": float(wt >= best),
        "corners": float(ls.n),
    }


def aggregate(per_cube: list[dict[str, float]]) -> dict[str, float]:
    """Means, plus the fractions that say how often the hard case actually occurs."""

    def col(k: str) -> np.ndarray:
        return np.array([c[k] for c in per_cube], float)

    greedy, path, rse = col("greedy_hit_rate"), col("path_accessibility"), col("rse_pairs")
    out = {
        "n_hypercubes": float(len(per_cube)),
        "n_seeds": float(len(SEEDS)),
        "mean_order_1_fraction": float(np.nanmean(col("order_1"))),
        "mean_order_2_fraction": float(np.nanmean(col("order_2"))),
        "mean_order_3plus_fraction": float(np.nanmean(col("order_3plus"))),
        "mean_nonadditive_variance_fraction": float(np.nanmean(col("nonadditive"))),
        "mean_greedy_hit_rate": float(np.nanmean(greedy)),
        "frac_hypercubes_greedy_never_solves": float(np.mean(greedy == 0.0)),
        "frac_hypercubes_greedy_always_solves": float(np.mean(greedy == 1.0)),
        "mean_greedy_oracle_calls": float(np.nanmean(col("greedy_oracle_calls"))),
        "mean_path_accessibility": float(np.nanmean(path)),
        "frac_hypercubes_no_monotone_path": float(np.nanmean(path == 0.0)),
        "frac_hypercubes_with_reciprocal_sign": float(np.mean(rse > 0)),
        "mean_reciprocal_sign_pairs": float(np.mean(rse)),
        "mean_stack_topk_gain_fraction": float(np.nanmean(col("stack_topk_gain_fraction"))),
        "frac_wild_type_corner_is_best": float(np.nanmean(col("wt_is_best"))),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default="6_40")
    ap.add_argument("--dims", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    table_path = HC / f"his3_seg_{args.segment}.txt"
    out_dir = HC / f"out_{args.segment}"
    if not table_path.exists() or not out_dir.exists():
        print(f"missing {table_path} or {out_dir}")
        return 1

    t0 = time.time()
    table = load_genotype_table(table_path)
    print(f"segment {args.segment}: {len(table):,} genotypes ({time.time()-t0:.1f}s)")

    dims = args.dims or sorted(SAMPLE_SIZE)
    results = {}
    for d in dims:
        f = out_dir / f"hypercubes_{d}.txt"
        if not f.exists():
            print(f"  dim {d}: no file (gitignored or not extracted)")
            continue
        t0 = time.time()
        cubes, stats = sample(f, table, SAMPLE_SIZE.get(d, 100), seed=args.seed)
        if not cubes:
            print(f"  dim {d}: nothing usable {stats}")
            continue
        per_cube = [measure(ls) for ls in cubes]
        agg = aggregate(per_cube) | {"dimension": float(d), "population": float(stats["seen"])}
        agg["sampling_fraction"] = agg["n_hypercubes"] / stats["seen"]
        results[str(d)] = {"aggregate": agg, "sampling": stats}
        print(
            f"  dim {d}: {stats['seen']:>9,} available, {len(cubes):>3} sampled  "
            f"add {agg['mean_order_1_fraction']:.3f}  greedy {agg['mean_greedy_hit_rate']:.3f}  "
            f"path {agg['mean_path_accessibility']:.3f}  "
            f"rse {agg['frac_hypercubes_with_reciprocal_sign']:.2f}  ({time.time()-t0:.0f}s)",
            flush=True,
        )

        if not args.no_track:
            with tracking.run(
                "1.1",
                "greedy_reachability",
                variant={
                    "dataset": "HIS3",
                    "segment": args.segment,
                    "dimension": d,
                    "measurement": "hypercube_paths",
                },
                datasets=[str(table_path)],
                seed=args.seed,
                notes=(
                    f"{len(cubes)} of {stats['seen']:,} dimension-{d} hypercubes from HIS3 "
                    f"segment {args.segment}; exact Walsh variance decomposition"
                ),
            ) as r:
                r.log_metrics({k: v for k, v in agg.items() if np.isfinite(v)})

    if results:
        existing = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
        existing[args.segment] = results
        OUT_JSON.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
