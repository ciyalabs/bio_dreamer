"""command line interface: `bdtrack <command>`.

    bdtrack init                       create the ledger, validate plan.yaml
    bdtrack status [--phase N]         plan tree with gates, thresholds, flags
    bdtrack show <step>                every run and variant for one step
    bdtrack compare <step> -m METRIC   variants ranked by a metric
    bdtrack runs [--failed] [--dirty]  recent runs across the whole plan
    bdtrack verdict <step> <decision>  record a go / no-go / rescope decision
    bdtrack export                     write the committed json + markdown record
    bdtrack report [--open]            render the html dashboard
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import webbrowser
from pathlib import Path

from . import report as _report
from . import status as _status
from .plan import load_plan
from .provenance import variant_label
from .store import DEFAULT_DB, Store

DEFAULT_HTML = Path("results/tracker.html")
DEFAULT_JSON = Path("results/summary.json")
DEFAULT_MD = Path("results/RESULTS.md")

_MARK = {"pass": "PASS", "fail": "FAIL", "no-data": "   -"}
_STATE_TAG = {
    "complete": "[done]",
    "running": "[run ]",
    "partial": "[part]",
    "thresholds-failed": "[FAIL]",
    "blocked": "[STOP]",
    "not-started": "[    ]",
}


def _load(args: argparse.Namespace) -> _status.PlanStatus:
    plan = load_plan(args.plan)
    with Store(args.db) as store:
        return _status.compute(plan, store)


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4g}"


# ---- commands ----


def cmd_init(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    with Store(args.db) as store:
        pass
    n_steps = len(plan.steps())
    n_exp = sum(len(s.experiments) for s in plan.steps())
    print(f"plan      {plan.source} ({plan.plan_hash})")
    print(f"ledger    {Path(args.db)}")
    print(f"loaded    {len(plan.phases)} phases, {n_steps} steps, {n_exp} experiments")
    gates = [s.id for s in plan.steps() if s.gate]
    print(f"gates     {', '.join(gates)}")
    pending = [s.id for s in plan.steps() if s.prereg_required and not s.thresholds]
    if pending:
        print(f"prereg    thresholds still needed for: {', '.join(pending)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    ps = _load(args)
    s = ps.summary()

    print(f"{ps.plan.project}  ·  plan {ps.plan.plan_hash}")
    print(
        f"steps {s['steps_complete']}/{s['steps_total']} complete  ·  "
        f"{s['runs_total']} runs  ·  {s['runs_failed']} failed  ·  "
        f"{s['gates_open']} gates open"
    )
    print()

    for phase in ps.plan.phases:
        if args.phase and phase.id != args.phase:
            continue
        print(f"Phase {phase.id} — {phase.name}")
        for step in phase.steps:
            st = ps.by_id(step.id)
            tag = _STATE_TAG[st.state]
            gate = " GATE" if step.gate else ""
            done = len(st.experiments_with_runs)
            counts = f"{done}/{len(step.experiments)}" if step.experiments else "—"
            print(f"  {tag} {step.id:<5} {step.name[:46]:<46} {counts:>7}{gate}")
            for t in st.thresholds:
                extra = f" over {len(t.values)} variant(s)" if len(t.values) > 1 else ""
                print(
                    f"        {_MARK[t.outcome]}  {t.description:<44} "
                    f"observed {_fmt(t.observed)}{extra}"
                )
                if t.outcome == "fail" and t.aggregate == "all" and t.failing_variants:
                    print(f"              failing: {', '.join(t.failing_variants[:4])}")
            if st.needs_prereg:
                print("        !!   no thresholds declared (prereg_required)")
            if st.verdict:
                print(f"        verdict: {st.verdict['decision']} — {st.verdict['rationale'][:60]}")
        print()

    flags = []
    if ps.missing_prereg:
        flags.append(
            f"{len(ps.missing_prereg)} step(s) need pre-registered thresholds: "
            + ", ".join(s.step.id for s in ps.missing_prereg)
        )
    if ps.dirty_runs:
        flags.append(f"{len(ps.dirty_runs)} completed run(s) from a dirty tree — not reproducible")
    if ps.stale_plan_runs:
        flags.append(f"{len(ps.stale_plan_runs)} run(s) recorded against an older plan.yaml")
    if ps.open_gates:
        flags.append("open gates: " + ", ".join(s.step.id for s in ps.open_gates))
    if ps.duplicate_variants:
        flags.append(
            "repeated variants: "
            + ", ".join(f"{a}/{b}x{n}" for a, b, _k, n in ps.duplicate_variants[:5])
        )
    if flags:
        print("Flags")
        for f in flags:
            print(f"  ! {f}")
        print()

    if ps.next_steps:
        print("Next unfinished step per phase")
        for st in ps.next_steps:
            print(f"  {st.step.id:<5} {st.step.name}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    ps = _load(args)
    try:
        st = ps.by_id(args.step)
    except KeyError:
        print(f"no such step: {args.step}", file=sys.stderr)
        return 1

    print(f"{st.step.id} — {st.step.name}   [{st.state}]")
    if st.step.why:
        print(f"why: {st.step.why}")
    print(f"size {st.step.size or '—'}   gate {st.step.gate}   runs {len(st.runs)}")
    print()

    if st.thresholds:
        print("Pre-registered thresholds")
        for t in st.thresholds:
            print(f"  {_MARK[t.outcome]}  {t.description:<40} observed {_fmt(t.observed)}  ({t.hypothesis})")
        print()

    if st.missing_experiments:
        print("Declared but not yet run")
        for m in st.missing_experiments:
            print(f"  - {m}")
        print()

    if not st.runs:
        print("No runs recorded.")
        return 0

    print("Runs")
    for r in st.runs:
        dirty = " DIRTY" if r.git_dirty else ""
        print(
            f"  {r.short_id}  {r.status:<9} {r.experiment_id:<26} "
            f"{variant_label(r.variant)[:40]:<40} {(r.git_sha or '—')[:7]}{dirty}"
        )
        for k, v in r.metrics.items():
            print(f"            {k} = {_fmt(v)}")
        if r.wandb_url:
            print(f"            curves: {r.wandb_url}")
        for a in r.artifacts:
            print(f"            artifact: {a['path']}" + (f" ({a['kind']})" if a["kind"] else ""))
        if r.error:
            print(f"            error: {r.error}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    ps = _load(args)
    try:
        st = ps.by_id(args.step)
    except KeyError:
        print(f"no such step: {args.step}", file=sys.stderr)
        return 1

    rows = [
        (r, r.metrics[args.metric])
        for r in st.runs
        if r.status == "completed" and args.metric in r.metrics
    ]
    if not rows:
        print(f"no completed runs in {args.step} logged metric {args.metric!r}")
        known = sorted({m for r in st.runs for m in r.metrics})
        if known:
            print("available metrics: " + ", ".join(known))
        return 1

    rows.sort(key=lambda x: x[1], reverse=not args.ascending)
    width = max(len(variant_label(r.variant)) for r, _ in rows)
    print(f"{st.step.id} — {args.metric}   ({len(rows)} runs)")
    for r, v in rows:
        dirty = " DIRTY" if r.git_dirty else ""
        print(f"  {_fmt(v):>12}  {variant_label(r.variant):<{width}}  {r.short_id}{dirty}")

    vals = [v for _, v in rows]
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        print(f"\n  mean {mean:.4g}   sd {var ** 0.5:.4g}   n {n}")
    else:
        print(f"\n  single run (n=1) — not evidence; see step 7.2 on seed replication")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    ps = _load(args)
    runs = [r for st in ps.steps for r in st.runs]
    if args.failed:
        runs = [r for r in runs if r.status == "failed"]
    if args.dirty:
        runs = [r for r in runs if r.git_dirty]
    runs.sort(key=lambda r: r.started_at, reverse=True)
    runs = runs[: args.limit]

    if not runs:
        print("no matching runs")
        return 0
    for r in runs:
        dirty = " DIRTY" if r.git_dirty else ""
        print(
            f"{r.started_at}  {r.short_id}  {r.status:<9} "
            f"{r.step_id:<5} {r.experiment_id:<26} {variant_label(r.variant)[:34]}{dirty}"
        )
    return 0


def cmd_verdict(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    if not plan.has_step(args.step):
        print(f"no such step: {args.step}", file=sys.stderr)
        return 1
    with Store(args.db) as store:
        ps = _status.compute(plan, store)
        st = ps.by_id(args.step)

        failing = [t for t in st.thresholds if t.outcome == "fail"]
        nodata = [t for t in st.thresholds if t.outcome == "no-data"]
        if args.decision == "go" and (failing or nodata) and not args.force:
            print(f"refusing to record 'go' for {args.step}:", file=sys.stderr)
            for t in failing:
                print(f"  FAILING  {t.description} (observed {_fmt(t.observed)})", file=sys.stderr)
            for t in nodata:
                print(f"  NO DATA  {t.description}", file=sys.stderr)
            print("\nrecord the real outcome, or pass --force with a rationale.", file=sys.stderr)
            return 2

        store.set_verdict(
            step_id=args.step,
            decision=args.decision,
            rationale=args.message,
            decided_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            decided_by=os.environ.get("USER") or "unknown",
        )
    print(f"recorded: {args.step} → {args.decision}")
    if args.force and args.decision == "go":
        print("(forced over failing or missing thresholds — rationale is on the record)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    ps = _load(args)
    j = _report.export_json(ps, args.json)
    m = _report.export_markdown(ps, args.markdown)
    print(f"wrote {j}")
    print(f"wrote {m}")
    print("these are the git-committed record; the sqlite ledger stays ignored")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    ps = _load(args)
    out = _report.write_html(ps, args.out)
    print(f"wrote {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


# ---- wiring ----


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bdtrack",
        description="ProteinDreamer plan and experiment tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--plan", default="plan.yaml", help="plan file (default: plan.yaml)")
    p.add_argument("--db", default=str(DEFAULT_DB), help=f"ledger (default: {DEFAULT_DB})")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the ledger and validate the plan").set_defaults(func=cmd_init)

    sp = sub.add_parser("status", help="plan tree with gates, thresholds and flags")
    sp.add_argument("--phase", help="restrict to one phase id")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("show", help="every run and variant for one step")
    sp.add_argument("step")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("compare", help="rank a step's variants by a metric")
    sp.add_argument("step")
    sp.add_argument("-m", "--metric", required=True)
    sp.add_argument("--ascending", action="store_true", help="lower is better")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("runs", help="recent runs across the plan")
    sp.add_argument("--limit", type=int, default=30)
    sp.add_argument("--failed", action="store_true")
    sp.add_argument("--dirty", action="store_true")
    sp.set_defaults(func=cmd_runs)

    sp = sub.add_parser("verdict", help="record a go / no-go / rescope decision")
    sp.add_argument("step")
    sp.add_argument("decision", choices=["go", "no-go", "rescope", "pending"])
    sp.add_argument("-m", "--message", default="", help="rationale (strongly recommended)")
    sp.add_argument("--force", action="store_true", help="allow 'go' despite failing thresholds")
    sp.set_defaults(func=cmd_verdict)

    sp = sub.add_parser("export", help="write the committed json + markdown record")
    sp.add_argument("--json", default=str(DEFAULT_JSON))
    sp.add_argument("--markdown", default=str(DEFAULT_MD))
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("report", help="render the html dashboard")
    sp.add_argument("-o", "--out", default=str(DEFAULT_HTML))
    sp.add_argument("--open", action="store_true", help="open in a browser")
    sp.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
