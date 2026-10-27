"""loads plan.yaml into a queryable structure.

the plan is data, not code: phases contain steps, steps contain experiments and
optional machine-checked thresholds. RESEARCH_PLAN.md holds the narrative; this
module only cares about the parts the tracker needs to validate runs against.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_OPS: dict[str, Any] = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


AGGREGATES = ("latest", "min", "max", "mean", "all")


@dataclass(frozen=True)
class Threshold:
    """a pre-registered condition on a logged metric.

    `aggregate` says how to combine the metric across a step's variants:
    'latest' (default) uses the most recent run, 'min'/'max'/'mean' reduce over
    all of them, and 'all' requires every variant to satisfy the condition.
    for a step that audits several landscapes, 'all' or the pessimistic
    'min'/'max' is usually what you mean -- 'latest' would judge the step on
    whichever landscape happened to run last.
    """

    metric: str
    op: str
    value: float
    hypothesis: str = ""
    aggregate: str = "latest"

    def reduce(self, values: list[float]) -> float | None:
        """collapses observed values to the single number shown as 'observed'."""
        if not values:
            return None
        if self.aggregate == "latest":
            return values[0]  # callers pass newest-first
        if self.aggregate == "min":
            return min(values)
        if self.aggregate == "max":
            return max(values)
        if self.aggregate == "mean":
            return sum(values) / len(values)
        if self.aggregate == "all":
            # the worst value with respect to this operator
            return min(values) if self.op in (">=", ">") else max(values)
        raise ValueError(f"unknown aggregate {self.aggregate!r}")

    def check(self, values: list[float]) -> str:
        """returns 'pass', 'fail' or 'no-data'."""
        if not values:
            return "no-data"
        if self.aggregate == "all":
            ok = all(_OPS[self.op](v, self.value) for v in values)
        else:
            ok = _OPS[self.op](self.reduce(values), self.value)
        return "pass" if ok else "fail"

    def describe(self) -> str:
        agg = "" if self.aggregate == "latest" else f" [{self.aggregate}]"
        return f"{self.metric} {self.op} {self.value:g}{agg}"


@dataclass(frozen=True)
class Experiment:
    id: str
    name: str


@dataclass
class Step:
    id: str
    name: str
    phase_id: str
    size: str = ""
    why: str = ""
    gate: bool = False
    prereg_required: bool = False
    experiments: list[Experiment] = field(default_factory=list)
    thresholds: list[Threshold] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def experiment_ids(self) -> set[str]:
        return {e.id for e in self.experiments}


@dataclass
class Phase:
    id: str
    name: str
    goal: str = ""
    steps: list[Step] = field(default_factory=list)


@dataclass
class Plan:
    project: str
    phases: list[Phase]
    reminders: list[dict[str, str]]
    plan_hash: str
    source: Path

    # ---- lookups ----

    def steps(self) -> list[Step]:
        return [s for p in self.phases for s in p.steps]

    def step(self, step_id: str) -> Step:
        for s in self.steps():
            if s.id == step_id:
                return s
        raise KeyError(
            f"step {step_id!r} is not in {self.source.name}. "
            f"known steps: {', '.join(s.id for s in self.steps())}"
        )

    def phase(self, phase_id: str) -> Phase:
        for p in self.phases:
            if p.id == phase_id:
                return p
        raise KeyError(f"phase {phase_id!r} is not in {self.source.name}")

    def has_step(self, step_id: str) -> bool:
        return any(s.id == step_id for s in self.steps())


def load_plan(path: str | Path = "plan.yaml") -> Plan:
    """parses plan.yaml. raises if the file is missing or malformed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. the tracker needs it; run `bdtrack init` from the repo root."
        )

    raw = path.read_bytes()
    plan_hash = hashlib.sha256(raw).hexdigest()[:12]
    data = yaml.safe_load(raw)

    phases: list[Phase] = []
    seen_steps: set[str] = set()

    for p in data.get("phases", []):
        phase = Phase(id=str(p["id"]), name=p["name"], goal=p.get("goal", ""))
        for s in p.get("steps", []):
            step_id = str(s["id"])
            if step_id in seen_steps:
                raise ValueError(f"duplicate step id {step_id!r} in {path}")
            seen_steps.add(step_id)

            thresholds = []
            for t in s.get("thresholds") or []:
                if t["op"] not in _OPS:
                    raise ValueError(
                        f"step {step_id}: unknown op {t['op']!r}; "
                        f"expected one of {', '.join(_OPS)}"
                    )
                agg = t.get("aggregate", "latest")
                if agg not in AGGREGATES:
                    raise ValueError(
                        f"step {step_id}: unknown aggregate {agg!r}; "
                        f"expected one of {', '.join(AGGREGATES)}"
                    )
                thresholds.append(
                    Threshold(
                        metric=t["metric"],
                        op=t["op"],
                        value=float(t["value"]),
                        hypothesis=t.get("hypothesis", ""),
                        aggregate=agg,
                    )
                )

            phase.steps.append(
                Step(
                    id=step_id,
                    name=s["name"],
                    phase_id=phase.id,
                    size=s.get("size", ""),
                    why=s.get("why", ""),
                    gate=bool(s.get("gate", False)),
                    prereg_required=bool(s.get("prereg_required", False)),
                    experiments=[
                        Experiment(id=e["id"], name=e["name"])
                        for e in (s.get("experiments") or [])
                    ],
                    thresholds=thresholds,
                    outputs=list(s.get("outputs") or []),
                )
            )
        phases.append(phase)

    return Plan(
        project=data.get("project", "unnamed"),
        phases=phases,
        reminders=list(data.get("reminders") or []),
        plan_hash=plan_hash,
        source=path,
    )
