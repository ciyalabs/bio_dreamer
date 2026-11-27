"""Step 2.4 — re-read H1's four registered conditions against the current reward model.

H1 was scored at Step 2.2 on the 266k reward head, and failed. The reward model has since
changed: Step 2.4 replaced it with a Gaussian process. The gate asks about *the reward
model*, so leaving the verdict attached to a component that is no longer the reward model
would be reporting a stale answer to a live question.

**No threshold is touched and no baseline is recomputed.** The additive null and the frozen
ridge are read from `data/processed/h1_benchmark.json` exactly as Step 2.2 measured them,
on the same assays and the same split seed; only the model column changes. The four
conditions and their values come from `plan.yaml`, so they cannot drift from the
pre-registration.

The primary row is `gp_zeroshot_mean` — Kermut-sequence as published, which is what
`spearman_mean_gp` was registered against. `gp_composite_kernel` is reported beside it and
is an extension beyond the published configuration, so it is labelled rather than promoted.

    python -I scripts/h1_verdict_gp.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from biodreamer import tracking

H1 = Path("data/processed/h1_benchmark.json")
GP = Path("data/processed/gp_reward.json")
OUT = Path("data/processed/h1_verdict_gp.json")
PRIMARY = "gp_zeroshot_mean"


def thresholds() -> dict[str, tuple[str, float]]:
    """Read H1's conditions from plan.yaml rather than restating them here."""
    plan = tracking.load_plan()
    for ph in plan.phases:
        for st in ph.steps:
            if st.id == "2.2":
                return {t.metric: (t.op, t.value) for t in st.thresholds}
    raise KeyError("step 2.2 not found in plan.yaml")


def passes(op: str, got: float, want: float) -> bool:
    return {">": got > want, ">=": got >= want}[op]


def main() -> int:
    h1 = json.loads(H1.read_text())
    gp = json.loads(GP.read_text())
    per = h1["per_assay"]
    assays = sorted(per)

    out: dict[str, dict] = {}
    for model in ("gp_zeroshot_mean", "gp_composite_kernel"):
        pred = gp["per_assay"][model]
        rows = [(a, pred[a], per[a]["additive_null"], per[a]["frozen_ridge"])
                for a in assays if a in pred]
        mean = float(np.mean([r[1] for r in rows]))
        null_mean = float(np.mean([r[2] for r in rows]))
        ridge_mean = float(np.mean([r[3] for r in rows]))
        beat = sum(1 for r in rows if r[1] > r[2])
        m = {
            "spearman_mean": mean,
            "margin_over_additive_null": mean - null_mean,
            "frac_assays_beating_null": beat / len(rows),
            "spearman_mean_vs_frozen_regression": mean - ridge_mean,
        }
        th = thresholds()
        verdict = {k: passes(th[k][0], v, th[k][1]) for k, v in m.items()}
        out[model] = {"metrics": m, "verdict": verdict, "n_assays": len(rows),
                      "assays_beating_null": beat}
        print(f"\n=== {model} ===")
        for k, v in m.items():
            op, want = th[k]
            print(f"  {k:38s} {op:>2s} {want:<5} measured {v:+.3f}   "
                  f"{'PASS' if verdict[k] else 'FAIL'}")
        n_fail = sum(1 for p in verdict.values() if not p)
        print(f"  --> {'H1 PASSES' if n_fail == 0 else f'H1 FAILS ({n_fail} of 4)'}")

    # the comparison Step 2.2 recorded as the finding that mattered more than the verdict
    print(f"\nreference, measured at 2.2 on the same assays:")
    print(f"  reward head (266k)            +0.409")
    print(f"  frozen ridge + mutation count +0.390")
    print(f"  additive null                 {np.mean([per[a]['additive_null'] for a in assays]):+.3f}")
    print(f"  frozen ridge                  {np.mean([per[a]['frozen_ridge'] for a in assays]):+.3f}")

    OUT.write_text(json.dumps(
        {"primary": PRIMARY, "models": out,
         "baselines_from": str(H1), "note":
         "thresholds read from plan.yaml step 2.2; baselines reused from the 2.2 run, "
         "not recomputed; only the model column changes"}, indent=2) + "\n")
    print(f"\nwrote {OUT}")

    with tracking.run(
        "2.4", "h1_verdict_rescored",
        variant={"instance": "benchmark", "model": PRIMARY, "rescored": True},
        config={"source": str(GP)}, seed=0, datasets=[str(H1), str(GP)],
        notes=("H1's four registered conditions re-read against the Step 2.4 reward model. "
               "Thresholds come from plan.yaml and baselines are reused from the 2.2 run "
               "rather than recomputed, so only the model column changes. Primary is "
               "gp_zeroshot_mean (Kermut-sequence as published); the composite kernel is "
               "an extension beyond that configuration and is labelled, not promoted."),
    ) as r:
        r.log_metrics({f"{k}": v for k, v in out[PRIMARY]["metrics"].items()})
        r.log_artifact(str(OUT), kind="table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
