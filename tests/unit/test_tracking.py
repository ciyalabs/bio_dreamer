"""tests for the plan/experiment tracker.

covers the behaviours that matter for not mixing results up: variant identity,
automatic provenance, crashed runs leaving a record, threshold evaluation, and
the refusal to record 'go' over a failing gate.
"""

from __future__ import annotations

import json

import pytest
import yaml

from biodreamer.tracking import compute, load_plan, run
from biodreamer.tracking.cli import main as cli_main
from biodreamer.tracking.provenance import variant_key, variant_label
from biodreamer.tracking.report import export_json, export_markdown, render_html
from biodreamer.tracking.store import Store

PLAN = {
    "project": "TestPlan",
    "phases": [
        {
            "id": "1",
            "name": "Diagnostics",
            "goal": "cheap kill shots",
            "steps": [
                {
                    "id": "1.1",
                    "name": "Epistasis audit",
                    "size": "S",
                    "gate": True,
                    "why": "no valley means no H4",
                    "experiments": [
                        {"id": "additive_fit", "name": "Additive fit"},
                        {"id": "greedy_reachability", "name": "Greedy reachability"},
                    ],
                    "thresholds": [
                        {
                            "metric": "nonadditive_variance_fraction",
                            "op": ">=",
                            "value": 0.15,
                            "hypothesis": "H4-precondition",
                        }
                    ],
                },
                {
                    "id": "1.2",
                    "name": "Needs prereg",
                    "prereg_required": True,
                    "experiments": [{"id": "probe", "name": "Probe"}],
                },
            ],
        }
    ],
    "reminders": [{"id": "r1", "text": "additive null on every claim"}],
}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """a plan file and ledger in an isolated directory."""
    monkeypatch.chdir(tmp_path)
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(PLAN))
    return {"plan": str(plan_path), "db": str(tmp_path / "t.db"), "tmp": tmp_path}


# ---- plan ----


def test_plan_loads_and_indexes(env):
    plan = load_plan(env["plan"])
    assert plan.project == "TestPlan"
    assert [s.id for s in plan.steps()] == ["1.1", "1.2"]
    assert plan.step("1.1").gate is True
    assert plan.step("1.1").thresholds[0].metric == "nonadditive_variance_fraction"
    assert len(plan.plan_hash) == 12


def test_unknown_step_lists_valid_ids(env):
    plan = load_plan(env["plan"])
    with pytest.raises(KeyError, match="1.1, 1.2"):
        plan.step("9.9")


def test_duplicate_step_id_rejected(tmp_path):
    bad = dict(PLAN)
    bad["phases"] = [
        {
            "id": "1",
            "name": "x",
            "steps": [{"id": "1.1", "name": "a"}, {"id": "1.1", "name": "b"}],
        }
    ]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="duplicate step id"):
        load_plan(p)


def test_unknown_threshold_aggregate_rejected(tmp_path):
    bad = json.loads(json.dumps(PLAN))
    bad["phases"][0]["steps"][0]["thresholds"][0]["aggregate"] = "median"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="unknown aggregate"):
        load_plan(p)


def test_aggregate_all_requires_every_variant(env):
    """a step sweeping landscapes must not be judged on whichever ran last."""
    spec = json.loads(json.dumps(PLAN))
    spec["phases"][0]["steps"][0]["thresholds"][0]["aggregate"] = "all"
    plan_path = env["tmp"] / "agg.yaml"
    plan_path.write_text(yaml.safe_dump(spec))
    plan = load_plan(plan_path)

    for lan, val in [("GB1", 0.30), ("TrpB", 0.28), ("PhoQ", 0.02)]:
        with run("1.1", "additive_fit", variant={"landscape": lan},
                 plan=plan, db=env["db"]) as r:
            r.log_metric("nonadditive_variance_fraction", val)

    with Store(env["db"]) as store:
        st = compute(plan, store).by_id("1.1")
    t0 = st.thresholds[0]
    assert t0.outcome == "fail"          # PhoQ drags it down
    assert t0.observed == 0.02           # worst value reported, not the newest
    assert t0.failing_variants == ["landscape=PhoQ"]
    assert len(t0.values) == 3


def test_unknown_threshold_op_rejected(tmp_path):
    bad = json.loads(json.dumps(PLAN))
    bad["phases"][0]["steps"][0]["thresholds"][0]["op"] = "~="
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="unknown op"):
        load_plan(p)


# ---- variant identity ----


def test_variant_key_is_order_independent():
    assert variant_key({"a": 1, "b": 2}) == variant_key({"b": 2, "a": 1})
    assert variant_key({"a": 1}) != variant_key({"a": 2})


def test_variant_label_is_sorted_and_readable():
    assert variant_label({"seed": 1, "landscape": "GB1"}) == "landscape=GB1, seed=1"
    assert variant_label({}) == "(default)"


# ---- recording ----


def test_run_records_metrics_and_provenance(env):
    plan = load_plan(env["plan"])
    with run(
        "1.1",
        "additive_fit",
        variant={"landscape": "GB1-4site"},
        config={"lr": 0.1},
        seed=7,
        plan=plan,
        db=env["db"],
    ) as r:
        r.log_metric("nonadditive_variance_fraction", 0.21)
        r.log_metrics({"additive_r2": 0.79})

    with Store(env["db"]) as store:
        rec = store.runs(step_id="1.1")[0]
    assert rec.status == "completed"
    assert rec.metrics == {"additive_r2": 0.79, "nonadditive_variance_fraction": 0.21}
    assert rec.variant == {"landscape": "GB1-4site", "seed": 7}  # seed folded in
    assert rec.seed == 7
    assert rec.config_hash and rec.plan_hash == plan.plan_hash
    assert rec.duration_s is not None
    assert isinstance(rec.git_dirty, bool)


def test_undeclared_experiment_is_refused(env):
    plan = load_plan(env["plan"])
    with pytest.raises(ValueError, match="not declared under step 1.1"):
        run("1.1", "typo_experiment", plan=plan, db=env["db"])


def test_crashed_run_still_leaves_a_record(env):
    plan = load_plan(env["plan"])
    with pytest.raises(RuntimeError, match="boom"):
        with run("1.1", "additive_fit", plan=plan, db=env["db"]) as r:
            r.log_metric("partial", 1.0)
            raise RuntimeError("boom")

    with Store(env["db"]) as store:
        rec = store.runs(step_id="1.1")[0]
    assert rec.status == "failed"
    assert "boom" in rec.error
    assert rec.metrics["partial"] == 1.0  # metrics logged before the crash survive


def test_relogging_a_metric_overwrites(env):
    plan = load_plan(env["plan"])
    with run("1.1", "additive_fit", plan=plan, db=env["db"]) as r:
        r.log_metric("x", 1.0)
        r.log_metric("x", 2.0)
    with Store(env["db"]) as store:
        assert store.runs(step_id="1.1")[0].metrics["x"] == 2.0


def test_artifacts_and_notes_recorded(env):
    plan = load_plan(env["plan"])
    fig = env["tmp"] / "fig.png"
    fig.write_bytes(b"x")
    with run("1.1", "additive_fit", plan=plan, db=env["db"]) as r:
        r.log_artifact(fig, kind="figure", description="epistasis")
        r.note("looks additive")
    with Store(env["db"]) as store:
        rec = store.runs(step_id="1.1")[0]
    assert rec.artifacts[0]["kind"] == "figure"
    assert rec.notes == "looks additive"


# ---- status and thresholds ----


def test_threshold_passes_and_fails_on_latest_value(env):
    plan = load_plan(env["plan"])
    with run("1.1", "additive_fit", variant={"v": 1}, plan=plan, db=env["db"]) as r:
        r.log_metric("nonadditive_variance_fraction", 0.30)

    with Store(env["db"]) as store:
        st = compute(plan, store).by_id("1.1")
    assert st.thresholds[0].outcome == "pass"
    assert st.thresholds[0].observed == 0.30

    # a later run supersedes the earlier number rather than competing with it
    with run("1.1", "additive_fit", variant={"v": 2}, plan=plan, db=env["db"]) as r:
        r.log_metric("nonadditive_variance_fraction", 0.02)

    with Store(env["db"]) as store:
        st = compute(plan, store).by_id("1.1")
    assert st.thresholds[0].outcome == "fail"
    assert st.state == "thresholds-failed"


def test_no_data_threshold_and_partial_state(env):
    plan = load_plan(env["plan"])
    with run("1.1", "additive_fit", plan=plan, db=env["db"]) as r:
        r.log_metric("unrelated", 1.0)
    with Store(env["db"]) as store:
        st = compute(plan, store).by_id("1.1")
    assert st.thresholds[0].outcome == "no-data"
    assert st.state == "partial"
    assert "greedy_reachability" in st.missing_experiments


def test_missing_prereg_is_surfaced(env):
    plan = load_plan(env["plan"])
    with Store(env["db"]) as store:
        ps = compute(plan, store)
    assert [s.step.id for s in ps.missing_prereg] == ["1.2"]
    assert ps.summary()["steps_missing_prereg"] == 1


def test_duplicate_variants_surfaced(env):
    plan = load_plan(env["plan"])
    for _ in range(2):
        with run("1.1", "additive_fit", variant={"seed": 1}, plan=plan, db=env["db"]) as r:
            r.log_metric("x", 1.0)
    with Store(env["db"]) as store:
        ps = compute(plan, store)
    assert ps.duplicate_variants and ps.duplicate_variants[0][3] == 2


def test_open_gates_and_next_steps(env):
    plan = load_plan(env["plan"])
    with Store(env["db"]) as store:
        ps = compute(plan, store)
    assert "1.1" in [s.step.id for s in ps.open_gates]
    assert ps.next_steps[0].step.id == "1.1"


# ---- verdicts ----


def test_verdict_go_refused_over_failing_threshold(env):
    plan = load_plan(env["plan"])
    with run("1.1", "additive_fit", plan=plan, db=env["db"]) as r:
        r.log_metric("nonadditive_variance_fraction", 0.01)

    args = ["--plan", env["plan"], "--db", env["db"], "verdict", "1.1", "go", "-m", "ship it"]
    assert cli_main(args) == 2  # refused

    assert cli_main([*args, "--force"]) == 0  # allowed, on the record
    with Store(env["db"]) as store:
        assert store.verdicts()["1.1"]["decision"] == "go"


def test_verdict_no_go_always_allowed(env):
    assert (
        cli_main(
            ["--plan", env["plan"], "--db", env["db"], "verdict", "1.1", "no-go", "-m", "additive"]
        )
        == 0
    )
    with Store(env["db"]) as store:
        st = compute(load_plan(env["plan"]), store).by_id("1.1")
    assert st.state == "blocked"


# ---- outputs ----


def test_html_report_is_self_contained(env):
    plan = load_plan(env["plan"])
    with run("1.1", "additive_fit", variant={"landscape": "GB1"}, plan=plan, db=env["db"]) as r:
        r.log_metric("nonadditive_variance_fraction", 0.21)
    with Store(env["db"]) as store:
        html = render_html(compute(plan, store))

    assert html.startswith("<!doctype html>")
    assert "<script" not in html.lower()  # no external or inline JS needed
    assert "prefers-color-scheme" in html  # dark mode
    assert "Epistasis audit" in html and "landscape=GB1" in html
    assert "0.21" in html


def test_exports_written_and_parseable(env):
    plan = load_plan(env["plan"])
    with run("1.1", "additive_fit", plan=plan, db=env["db"]) as r:
        r.log_metric("nonadditive_variance_fraction", 0.21)
    with Store(env["db"]) as store:
        ps = compute(plan, store)
        j = export_json(ps, env["tmp"] / "summary.json")
        m = export_markdown(ps, env["tmp"] / "RESULTS.md")

    payload = json.loads(j.read_text())
    assert payload["summary"]["runs_total"] == 1
    step = next(s for s in payload["steps"] if s["id"] == "1.1")
    assert step["thresholds"][0]["outcome"] == "pass"
    assert "GATE" in m.read_text()


def test_cli_status_and_report_run_clean(env, capsys):
    plan = load_plan(env["plan"])
    with run("1.1", "additive_fit", plan=plan, db=env["db"]) as r:
        r.log_metric("nonadditive_variance_fraction", 0.21)

    base = ["--plan", env["plan"], "--db", env["db"]]
    assert cli_main([*base, "status"]) == 0
    out = capsys.readouterr().out
    assert "Epistasis audit" in out and "PASS" in out

    assert cli_main([*base, "show", "1.1"]) == 0
    assert cli_main([*base, "runs"]) == 0
    assert cli_main([*base, "report", "-o", str(env["tmp"] / "r.html")]) == 0
    assert (env["tmp"] / "r.html").exists()


def test_compare_reports_spread(env, capsys):
    plan = load_plan(env["plan"])
    for seed, val in [(1, 0.20), (2, 0.24), (3, 0.22)]:
        with run("1.1", "additive_fit", variant={"seed": seed}, plan=plan, db=env["db"]) as r:
            r.log_metric("nonadditive_variance_fraction", val)

    rc = cli_main(
        ["--plan", env["plan"], "--db", env["db"], "compare", "1.1",
         "-m", "nonadditive_variance_fraction"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "seed=2" in out and "mean" in out and "sd" in out


def test_compare_lists_available_metrics_on_miss(env, capsys):
    plan = load_plan(env["plan"])
    with run("1.1", "additive_fit", plan=plan, db=env["db"]) as r:
        r.log_metric("actual_metric", 1.0)
    rc = cli_main(["--plan", env["plan"], "--db", env["db"], "compare", "1.1", "-m", "nope"])
    assert rc == 1
    assert "actual_metric" in capsys.readouterr().out


def test_real_project_plan_is_valid():
    """the committed plan.yaml must parse and stay consistent with itself."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    plan = load_plan(root / "plan.yaml")
    assert plan.project == "ProteinDreamer"
    ids = [s.id for s in plan.steps()]
    assert len(ids) == len(set(ids))
    assert "6.3" in ids  # H4, the central claim
    assert plan.step("6.3").gate is True
    assert any(t.metric == "correlation_with_additive" for t in plan.step("6.3").thresholds)
