"""plan and experiment tracking for the ProteinDreamer research plan.

the ledger answers "what did step 3.4 conclude, and which variant was that?"
months later, without a search through logs. per-step training curves stay in
W&B; this records the plan structure, variants, provenance, summary metrics,
pre-registered threshold outcomes and the decision that was taken.

    from biodreamer import tracking

    with tracking.run("1.1", "additive_fit",
                      variant={"landscape": "GB1-4site"},
                      config=cfg, seed=0) as r:
        r.log_metric("nonadditive_variance_fraction", 0.17)
        r.log_artifact("figures/gb1_epistasis.png", kind="figure")
        r.attach_wandb()
"""

from __future__ import annotations

from .api import Run, run
from .plan import Plan, Step, Threshold, load_plan
from .status import PlanStatus, StepStatus, compute
from .store import RunRecord, Store

__all__ = [
    "Plan",
    "PlanStatus",
    "Run",
    "RunRecord",
    "Step",
    "StepStatus",
    "Store",
    "Threshold",
    "compute",
    "load_plan",
    "run",
]
