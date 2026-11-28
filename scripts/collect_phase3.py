"""Aggregate the per-assay Step 3.1/3.2 runs into one gate reading.

`scripts/phase3_pod.sh` trains one assay at a time so a dying pod still leaves finished
work behind. That means the gate has to be read ACROSS the per-assay files rather than
inside one run, and it is read here so the aggregation is one auditable place.

Both readings are reported: the superseded `latent_effective_rank_ratio >= 0.5` against
`d_model`, and the amended `rank_preserved_vs_input` mean >= 0.7 and min >= 0.5
(PREREGISTRATION.md, 2026-11-28). Assays below `--min-transitions` are listed as skipped
and excluded from both, because averaging a starved assay into a collapse gate measures
starvation.

    python -I scripts/collect_phase3.py --dir results/pod_phase3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RANK_FLOOR = 0.5
PRESERVED_FLOOR_MEAN = 0.7
PRESERVED_FLOOR_MIN = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/pod_phase3")
    ap.add_argument("--min-transitions", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("wm_residue_*.json"))
    if not files:
        raise SystemExit(f"no wm_residue_*.json in {args.dir}")

    rows, skipped = {}, {}
    for f in files:
        d = json.loads(f.read_text())
        for assay, r in d.get("per_assay", {}).items():
            fin = r["final"]
            if r["n_transitions"] < args.min_transitions:
                skipped[assay] = r["n_transitions"]
                continue
            rows[assay] = {
                "n_transitions": r["n_transitions"],
                "length": r.get("length"),
                "latent_rank": fin["latent_effective_rank"],
                "latent_rank_ratio": fin["latent_effective_rank_ratio"],
                "input_rank": fin["input_effective_rank"],
                "preserved": fin["rank_preserved_vs_input"],
                "inverse_position_top1": fin["inverse_position_top1"],
                "inverse_aa_top1": fin["inverse_aa_top1"],
                "one_step_vs_identity": fin["one_step_vs_identity"],
                "sigreg_stat": fin["sigreg_stat"],
            }
        for assay, r in d.get("skipped_for_too_few_transitions", {}).items():
            skipped[assay] = r.get("n_transitions", 0)

    if not rows:
        raise SystemExit(
            f"every assay in {args.dir} fell below --min-transitions "
            f"{args.min_transitions}; nothing to read the gate on"
        )

    print(f"{'assay':34s} {'L':>5} {'trans':>9} {'latent':>8} {'input':>8} "
          f"{'preserved':>10} {'inv pos@1':>10} {'1-step/id':>10}")
    for a in sorted(rows, key=lambda k: -rows[k]["n_transitions"]):
        r = rows[a]
        print(f"{a:34s} {str(r['length'] or '?'):>5} {r['n_transitions']:>9,} "
              f"{r['latent_rank']:>8.1f} {r['input_rank']:>8.1f} "
              f"{r['preserved']:>10.3f} {r['inverse_position_top1']:>10.3f} "
              f"{r['one_step_vs_identity']:>10.3f}")
    for a, n in sorted(skipped.items()):
        print(f"{a:34s} {'':>5} {n:>9,}   SKIPPED, below --min-transitions "
              f"{args.min_transitions:,}")

    pres = np.array([r["preserved"] for r in rows.values()])
    ratio = np.array([r["latent_rank_ratio"] for r in rows.values()])
    summary = {
        "n_assays": len(rows),
        "latent_effective_rank_ratio": float(ratio.mean()),
        "rank_preserved_vs_input": float(pres.mean()),
        "min_rank_preserved_vs_input": float(pres.min()),
        "inverse_position_top1": float(
            np.mean([r["inverse_position_top1"] for r in rows.values()])),
        "one_step_vs_identity": float(
            np.mean([r["one_step_vs_identity"] for r in rows.values()])),
    }
    sup_pass = summary["latent_effective_rank_ratio"] >= RANK_FLOOR
    mean_ok = summary["rank_preserved_vs_input"] >= PRESERVED_FLOOR_MEAN
    min_ok = summary["min_rank_preserved_vs_input"] >= PRESERVED_FLOOR_MIN
    passed = mean_ok and min_ok

    print(f"\n{'=' * 78}")
    print(f"  SUPERSEDED  latent_effective_rank_ratio >= {RANK_FLOOR}   "
          f"{summary['latent_effective_rank_ratio']:+.4f}  "
          f"{'PASS' if sup_pass else 'FAIL as registered'}")
    print(f"  GATE        rank_preserved_vs_input mean >= {PRESERVED_FLOOR_MEAN}  "
          f"{summary['rank_preserved_vs_input']:+.4f}  {'PASS' if mean_ok else 'FAIL'}")
    print(f"  GATE        rank_preserved_vs_input min  >= {PRESERVED_FLOOR_MIN}  "
          f"{summary['min_rank_preserved_vs_input']:+.4f}  "
          f"{'PASS' if min_ok else 'FAIL'}")
    print(f"\n  STEP 3.2 -> {'PASS -- no collapse' if passed else 'FAIL'}"
          f"   (read on the per-residue state, {len(rows)} "
          f"assay{'s' if len(rows) != 1 else ''})")

    summary.update({"collapse_gate_passed": passed,
                    "superseded_gate_passed": sup_pass,
                    "thresholds": {"latent_effective_rank_ratio": RANK_FLOOR,
                                   "rank_preserved_vs_input_mean": PRESERVED_FLOOR_MEAN,
                                   "rank_preserved_vs_input_min": PRESERVED_FLOOR_MIN},
                    "per_assay": rows, "skipped": skipped,
                    "state": "residue"})
    out = args.out or str(Path(args.dir) / "summary.json")
    Path(out).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
