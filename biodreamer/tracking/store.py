"""sqlite ledger for runs, metrics, artifacts and recorded decisions.

one file, no server. per-step training curves stay in W&B; this stores the
summary metrics, the provenance, and the decision that was taken -- the things
you need months later when you have forgotten which run was which.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DB = Path("results/tracking.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    step_id       TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    variant       TEXT NOT NULL,
    variant_key   TEXT NOT NULL,
    status        TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    duration_s    REAL,
    git_sha       TEXT,
    git_branch    TEXT,
    git_dirty     INTEGER NOT NULL DEFAULT 1,
    plan_hash     TEXT,
    config        TEXT,
    config_hash   TEXT,
    dataset_hash  TEXT,
    seed          INTEGER,
    wandb_url     TEXT,
    wandb_run_id  TEXT,
    host          TEXT,
    user          TEXT,
    python        TEXT,
    platform      TEXT,
    notes         TEXT,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_step    ON runs(step_id);
CREATE INDEX IF NOT EXISTS idx_runs_variant ON runs(step_id, experiment_id, variant_key);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name   TEXT NOT NULL,
    value  REAL NOT NULL,
    PRIMARY KEY (run_id, name)
);

CREATE TABLE IF NOT EXISTS artifacts (
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    kind        TEXT,
    description TEXT,
    PRIMARY KEY (run_id, path)
);

CREATE TABLE IF NOT EXISTS verdicts (
    step_id    TEXT PRIMARY KEY,
    decision   TEXT NOT NULL,
    rationale  TEXT,
    decided_at TEXT NOT NULL,
    decided_by TEXT
);
"""


@dataclass
class RunRecord:
    """a row from `runs`, with its metrics and artifacts attached."""

    id: str
    step_id: str
    experiment_id: str
    variant: dict[str, Any]
    variant_key: str
    status: str
    started_at: str
    ended_at: str | None
    duration_s: float | None
    git_sha: str | None
    git_branch: str | None
    git_dirty: bool
    plan_hash: str | None
    config: dict[str, Any] | None
    config_hash: str | None
    dataset_hash: str | None
    seed: int | None
    wandb_url: str | None
    host: str | None
    notes: str | None
    error: str | None
    metrics: dict[str, float]
    artifacts: list[dict[str, str]]

    @property
    def short_id(self) -> str:
        return self.id[:8]


class Store:
    """thin sqlite wrapper. safe to open concurrently from short-lived writers."""

    def __init__(self, path: str | Path = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- writes ----

    def insert_run(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self._conn.execute(
            f"INSERT INTO runs ({cols}) VALUES ({marks})", list(row.values())
        )
        self._conn.commit()

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._conn.execute(
            f"UPDATE runs SET {sets} WHERE id = ?", [*fields.values(), run_id]
        )
        self._conn.commit()

    def set_metric(self, run_id: str, name: str, value: float) -> None:
        self._conn.execute(
            "INSERT INTO metrics (run_id, name, value) VALUES (?, ?, ?) "
            "ON CONFLICT(run_id, name) DO UPDATE SET value = excluded.value",
            (run_id, name, float(value)),
        )
        self._conn.commit()

    def add_artifact(
        self, run_id: str, path: str, kind: str | None, description: str | None
    ) -> None:
        self._conn.execute(
            "INSERT INTO artifacts (run_id, path, kind, description) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(run_id, path) DO UPDATE SET "
            "kind = excluded.kind, description = excluded.description",
            (run_id, path, kind, description),
        )
        self._conn.commit()

    def set_verdict(
        self,
        step_id: str,
        decision: str,
        rationale: str,
        decided_at: str,
        decided_by: str | None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO verdicts (step_id, decision, rationale, decided_at, decided_by) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(step_id) DO UPDATE SET "
            "decision = excluded.decision, rationale = excluded.rationale, "
            "decided_at = excluded.decided_at, decided_by = excluded.decided_by",
            (step_id, decision, rationale, decided_at, decided_by),
        )
        self._conn.commit()

    # ---- reads ----

    def _hydrate(self, row: sqlite3.Row) -> RunRecord:
        metrics = {
            r["name"]: r["value"]
            for r in self._conn.execute(
                "SELECT name, value FROM metrics WHERE run_id = ? ORDER BY name",
                (row["id"],),
            )
        }
        artifacts = [
            {"path": r["path"], "kind": r["kind"] or "", "description": r["description"] or ""}
            for r in self._conn.execute(
                "SELECT path, kind, description FROM artifacts WHERE run_id = ? ORDER BY path",
                (row["id"],),
            )
        ]
        return RunRecord(
            id=row["id"],
            step_id=row["step_id"],
            experiment_id=row["experiment_id"],
            variant=json.loads(row["variant"]),
            variant_key=row["variant_key"],
            status=row["status"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            duration_s=row["duration_s"],
            git_sha=row["git_sha"],
            git_branch=row["git_branch"],
            git_dirty=bool(row["git_dirty"]),
            plan_hash=row["plan_hash"],
            config=json.loads(row["config"]) if row["config"] else None,
            config_hash=row["config_hash"],
            dataset_hash=row["dataset_hash"],
            seed=row["seed"],
            wandb_url=row["wandb_url"],
            host=row["host"],
            notes=row["notes"],
            error=row["error"],
            metrics=metrics,
            artifacts=artifacts,
        )

    def runs(
        self,
        step_id: str | None = None,
        experiment_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[RunRecord]:
        where, params = [], []
        if step_id:
            where.append("step_id = ?")
            params.append(step_id)
        if experiment_id:
            where.append("experiment_id = ?")
            params.append(experiment_id)
        if status:
            where.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM runs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY started_at DESC, rowid DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [self._hydrate(r) for r in self._conn.execute(sql, params)]

    def run(self, run_id_prefix: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id LIKE ? ORDER BY started_at DESC, rowid DESC LIMIT 1",
            (run_id_prefix + "%",),
        ).fetchone()
        return self._hydrate(row) if row else None

    def verdicts(self) -> dict[str, dict[str, str]]:
        return {
            r["step_id"]: {
                "decision": r["decision"],
                "rationale": r["rationale"] or "",
                "decided_at": r["decided_at"],
                "decided_by": r["decided_by"] or "",
            }
            for r in self._conn.execute("SELECT * FROM verdicts")
        }

    def duplicate_variants(self) -> list[tuple[str, str, str, int]]:
        """(step_id, experiment_id, variant_key, n) where the same variant ran more than once.

        repeats are legitimate for seed replication and suspicious otherwise, so
        these are surfaced rather than blocked.
        """
        return [
            (r["step_id"], r["experiment_id"], r["variant_key"], r["n"])
            for r in self._conn.execute(
                "SELECT step_id, experiment_id, variant_key, COUNT(*) AS n FROM runs "
                "WHERE status = 'completed' "
                "GROUP BY step_id, experiment_id, variant_key HAVING n > 1"
            )
        ]

    def metric_values(self, step_id: str, metric: str) -> list[tuple[float, str]]:
        """every completed (value, variant_label) for a metric within a step.

        returned newest-first. thresholds aggregate over these according to their
        declared rule, so a step that logs one metric across several variants is
        not silently judged on whichever ran last.
        """
        rows = self._conn.execute(
            "SELECT m.value, r.variant FROM metrics m JOIN runs r ON r.id = m.run_id "
            "WHERE r.step_id = ? AND m.name = ? AND r.status = 'completed' "
            "ORDER BY r.started_at DESC, r.rowid DESC",
            (step_id, metric),
        ).fetchall()
        out = []
        for row in rows:
            from .provenance import variant_label

            out.append((row["value"], variant_label(json.loads(row["variant"]))))
        return out

    def best_metric(self, step_id: str, metric: str) -> float | None:
        """most recent completed value of a metric within a step."""
        row = self._conn.execute(
            "SELECT m.value FROM metrics m JOIN runs r ON r.id = m.run_id "
            "WHERE r.step_id = ? AND m.name = ? AND r.status = 'completed' "
            # rowid breaks ties deterministically: two runs can share a timestamp,
            # and insertion order is the only correct answer to "which is later".
            "ORDER BY r.started_at DESC, r.rowid DESC LIMIT 1",
            (step_id, metric),
        ).fetchone()
        return row["value"] if row else None
