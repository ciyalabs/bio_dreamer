"""the recording API used from experiment code.

usage:

    from biodreamer import tracking

    with tracking.run("1.1", "additive_fit",
                      variant={"landscape": "GB1-4site"},
                      config=cfg, seed=0,
                      datasets=["data/gb1_4site.csv"]) as r:
        ...
        r.log_metric("nonadditive_variance_fraction", 0.17)
        r.log_artifact("figures/gb1_epistasis.png", kind="figure")

provenance, timing and failure state are captured automatically. a run that
raises is still written, marked 'failed', with the traceback -- so a crashed
experiment leaves a record instead of a gap.
"""

from __future__ import annotations

import datetime as _dt
import logging
import traceback
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any

from . import provenance
from .plan import Plan, load_plan
from .store import DEFAULT_DB, Store

logger = logging.getLogger(__name__)


def _now() -> str:
    # microseconds, not seconds: two runs can otherwise share a timestamp
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class Run:
    """an open run. created by `run()`, not directly."""

    def __init__(
        self,
        store: Store,
        run_id: str,
        step_id: str,
        experiment_id: str,
        close_store: bool,
    ) -> None:
        self._store = store
        self._close_store = close_store
        self.id = run_id
        self.step_id = step_id
        self.experiment_id = experiment_id
        self._started = _dt.datetime.now(_dt.timezone.utc)
        self._metrics: dict[str, float] = {}

    # ---- recording ----

    def log_metric(self, name: str, value: float) -> None:
        """records a summary metric. re-logging the same name overwrites it."""
        self._metrics[name] = float(value)
        self._store.set_metric(self.id, name, value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_artifact(
        self, path: str | Path, kind: str = "", description: str = ""
    ) -> None:
        """records a pointer to a file on disk (figure, table, checkpoint).

        the file is not copied; a warning is logged if it does not exist yet, so a
        typo surfaces now rather than when the report is built.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("artifact %s does not exist (recorded anyway)", p)
        self._store.add_artifact(self.id, str(p), kind or None, description or None)

    def note(self, text: str) -> None:
        self._store.update_run(self.id, notes=text)

    def attach_wandb(self, url: str | None = None, run_id: str | None = None) -> None:
        """links this run to its W&B run, so training curves stay one click away.

        with no arguments it picks up the active `wandb.run` if there is one.
        """
        if url is None and run_id is None:
            try:
                import wandb  # noqa: PLC0415 — optional dependency, imported on demand

                if wandb.run is not None:
                    url, run_id = wandb.run.url, wandb.run.id
            except Exception:  # pragma: no cover - wandb absent or not initialised
                logger.debug("no active wandb run to attach")
                return
        if url or run_id:
            self._store.update_run(self.id, wandb_url=url, wandb_run_id=run_id)

    # ---- lifecycle ----

    def _finish(self, status: str, error: str | None = None) -> None:
        ended = _dt.datetime.now(_dt.timezone.utc)
        self._store.update_run(
            self.id,
            status=status,
            ended_at=ended.isoformat(),
            duration_s=(ended - self._started).total_seconds(),
            error=error,
        )

    def __enter__(self) -> Run:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc is None:
            self._finish("completed")
            logger.info(
                "run %s (%s/%s) completed with %d metric(s)",
                self.id[:8],
                self.step_id,
                self.experiment_id,
                len(self._metrics),
            )
        elif isinstance(exc, KeyboardInterrupt):
            self._finish("aborted", "KeyboardInterrupt")
        else:
            self._finish("failed", "".join(traceback.format_exception_only(exc_type, exc)).strip())
            logger.error("run %s failed: %s", self.id[:8], exc)
        if self._close_store:
            self._store.close()
        return False  # never swallow the exception


def run(
    step_id: str,
    experiment_id: str,
    variant: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    datasets: list[str | Path] | None = None,
    notes: str = "",
    plan: Plan | None = None,
    db: str | Path = DEFAULT_DB,
    store: Store | None = None,
) -> Run:
    """opens a tracked run for a plan step.

    args:
        step_id: a step id from plan.yaml, e.g. "1.1".
        experiment_id: an experiment id declared under that step.
        variant: what distinguishes this run from its siblings, e.g.
            {"landscape": "GB1-4site", "seed": 1}. hashed for identity.
        config: full configuration, stored and hashed.
        seed: random seed, also recorded in the variant if absent there.
        datasets: files or directories whose contents identify the data used.
        notes: free text.
        plan: pre-loaded plan, if you have one.
        db: ledger path.
        store: an open store to reuse; it will not be closed on exit.

    raises:
        KeyError: if step_id is not in the plan.
        ValueError: if experiment_id is not declared under that step.
    """
    plan = plan or load_plan()
    step = plan.step(step_id)  # raises with the list of valid ids

    known = step.experiment_ids()
    if known and experiment_id not in known:
        raise ValueError(
            f"experiment {experiment_id!r} is not declared under step {step_id} "
            f"in {plan.source.name}. declared: {', '.join(sorted(known)) or '(none)'}. "
            "add it to plan.yaml rather than logging to an undeclared id."
        )

    variant = dict(variant or {})
    if seed is not None and "seed" not in variant:
        variant["seed"] = seed

    prov = provenance.collect()
    own_store = store is None
    store = store or Store(db)
    run_id = uuid.uuid4().hex

    import json

    store.insert_run(
        {
            "id": run_id,
            "step_id": step_id,
            "experiment_id": experiment_id,
            "variant": json.dumps(variant, sort_keys=True, default=str),
            "variant_key": provenance.variant_key(variant),
            "status": "running",
            "started_at": _now(),
            "git_sha": prov.git_sha,
            "git_branch": prov.git_branch,
            "git_dirty": int(prov.git_dirty),
            "plan_hash": plan.plan_hash,
            "config": json.dumps(config, sort_keys=True, default=str) if config else None,
            "config_hash": provenance.hash_object(config) if config else None,
            "dataset_hash": provenance.hash_paths(datasets) if datasets else None,
            "seed": seed,
            "host": prov.host,
            "user": prov.user,
            "python": prov.python,
            "platform": prov.platform,
            "notes": notes or None,
        }
    )

    if prov.git_dirty:
        logger.warning(
            "run %s started with a dirty or unverifiable working tree; "
            "it will be flagged as non-reproducible in the report",
            run_id[:8],
        )

    return Run(store, run_id, step_id, experiment_id, close_store=own_store)
