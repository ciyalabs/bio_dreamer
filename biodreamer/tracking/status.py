"""derives plan status from the ledger: what has run, what passed, what is flagged.

kept separate from both the CLI and the report so the two cannot disagree about
what the state of the project is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .plan import Plan, Step
from .store import RunRecord, Store


@dataclass
class ThresholdResult:
    description: str
    hypothesis: str
    observed: float | None
    outcome: str  # pass | fail | no-data
    aggregate: str = "latest"
    # every (value, variant) behind `observed`, newest first
    values: list[tuple[float, str]] = field(default_factory=list)

    @property
    def failing_variants(self) -> list[str]:
        """variants that individually violate the condition (only meaningful for 'all')."""
        from .plan import _OPS  # local import keeps plan the single source of the ops

        if not self.values:
            return []
        op, target = self._op, self._target
        return [lab for v, lab in self.values if not _OPS[op](v, target)]

    _op: str = ">="
    _target: float = 0.0


@dataclass
class StepStatus:
    step: Step
    runs: list[RunRecord] = field(default_factory=list)
    thresholds: list[ThresholdResult] = field(default_factory=list)
    verdict: dict[str, str] | None = None

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.runs if r.status == "completed")

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.runs if r.status == "failed")

    @property
    def n_running(self) -> int:
        return sum(1 for r in self.runs if r.status == "running")

    @property
    def experiments_with_runs(self) -> set[str]:
        return {r.experiment_id for r in self.runs if r.status == "completed"}

    @property
    def missing_experiments(self) -> list[str]:
        return sorted(self.step.experiment_ids() - self.experiments_with_runs)

    @property
    def state(self) -> str:
        """one of: not-started, running, partial, thresholds-failed, complete, blocked."""
        if self.verdict and self.verdict["decision"] in {"no-go", "rescope"}:
            return "blocked"
        if any(t.outcome == "fail" for t in self.thresholds):
            return "thresholds-failed"
        if not self.runs:
            return "not-started"
        if self.n_running:
            return "running"
        if self.missing_experiments:
            return "partial"
        if self.step.thresholds and any(t.outcome == "no-data" for t in self.thresholds):
            return "partial"
        return "complete"

    @property
    def needs_prereg(self) -> bool:
        """a gate or pre-registered step with no thresholds declared yet."""
        return self.step.prereg_required and not self.step.thresholds


@dataclass
class PlanStatus:
    plan: Plan
    steps: list[StepStatus]
    duplicate_variants: list[tuple[str, str, str, int]]

    def by_id(self, step_id: str) -> StepStatus:
        for s in self.steps:
            if s.step.id == step_id:
                return s
        raise KeyError(step_id)

    # ---- cross-cutting flags: the things easy to miss ----

    @property
    def dirty_runs(self) -> list[RunRecord]:
        return [
            r
            for s in self.steps
            for r in s.runs
            if r.git_dirty and r.status == "completed"
        ]

    @property
    def failed_runs(self) -> list[RunRecord]:
        return [r for s in self.steps for r in s.runs if r.status == "failed"]

    @property
    def stale_plan_runs(self) -> list[RunRecord]:
        """completed runs recorded against an older plan.yaml than the current one."""
        return [
            r
            for s in self.steps
            for r in s.runs
            if r.status == "completed"
            and r.plan_hash
            and r.plan_hash != self.plan.plan_hash
        ]

    @property
    def missing_prereg(self) -> list[StepStatus]:
        return [s for s in self.steps if s.needs_prereg]

    @property
    def open_gates(self) -> list[StepStatus]:
        """gate steps that are not complete and have no recorded verdict."""
        return [
            s
            for s in self.steps
            if s.step.gate and s.state != "complete" and not s.verdict
        ]

    @property
    def next_steps(self) -> list[StepStatus]:
        """the first not-started step in each phase that has unfinished work."""
        out = []
        for phase in self.plan.phases:
            for step in phase.steps:
                s = self.by_id(step.id)
                if s.state in {"not-started", "partial", "thresholds-failed"}:
                    out.append(s)
                    break
        return out

    def summary(self) -> dict[str, Any]:
        states = [s.state for s in self.steps]
        return {
            "project": self.plan.project,
            "plan_hash": self.plan.plan_hash,
            "steps_total": len(self.steps),
            "steps_complete": states.count("complete"),
            "steps_running": states.count("running"),
            "steps_partial": states.count("partial"),
            "steps_not_started": states.count("not-started"),
            "steps_thresholds_failed": states.count("thresholds-failed"),
            "steps_blocked": states.count("blocked"),
            "runs_total": sum(len(s.runs) for s in self.steps),
            "runs_failed": len(self.failed_runs),
            "runs_dirty": len(self.dirty_runs),
            "runs_stale_plan": len(self.stale_plan_runs),
            "gates_open": len(self.open_gates),
            "steps_missing_prereg": len(self.missing_prereg),
        }


def compute(plan: Plan, store: Store) -> PlanStatus:
    """joins the plan against the ledger."""
    verdicts = store.verdicts()
    steps: list[StepStatus] = []

    for step in plan.steps():
        runs = store.runs(step_id=step.id)
        thresholds = []
        for t in step.thresholds:
            pairs = store.metric_values(step.id, t.metric)
            values = [v for v, _ in pairs]
            thresholds.append(
                ThresholdResult(
                    description=t.describe(),
                    hypothesis=t.hypothesis,
                    observed=t.reduce(values),
                    outcome=t.check(values),
                    aggregate=t.aggregate,
                    values=pairs,
                    _op=t.op,
                    _target=t.value,
                )
            )
        steps.append(
            StepStatus(
                step=step,
                runs=runs,
                thresholds=thresholds,
                verdict=verdicts.get(step.id),
            )
        )

    return PlanStatus(
        plan=plan,
        steps=steps,
        duplicate_variants=store.duplicate_variants(),
    )
