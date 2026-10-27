"""renders the ledger to a single self-contained HTML file, and to committed exports.

no external assets, no network, no build step: open the file in a browser. the
json/markdown exports are the git-committed record, so results diff readably even
though the sqlite file itself is ignored.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
from pathlib import Path

from .provenance import variant_label
from .status import PlanStatus, StepStatus

_STATE_ORDER = [
    "complete",
    "running",
    "partial",
    "thresholds-failed",
    "blocked",
    "not-started",
]

_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
  --panel: #f9fafb; --accent: #1f6feb;
  --ok: #15803d; --okbg: #dcfce7; --warn: #b45309; --warnbg: #fef3c7;
  --bad: #b91c1c; --badbg: #fee2e2; --idle: #6b7280; --idlebg: #f3f4f6;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e; --line: #30363d;
    --panel: #161b22; --accent: #58a6ff;
    --ok: #3fb950; --okbg: #0f2d1a; --warn: #d29922; --warnbg: #2d2206;
    --bad: #f85149; --badbg: #3d1518; --idle: #8b949e; --idlebg: #1c2128;
  }
}
:root[data-theme="dark"] {
  --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e; --line: #30363d;
  --panel: #161b22; --accent: #58a6ff;
  --ok: #3fb950; --okbg: #0f2d1a; --warn: #d29922; --warnbg: #2d2206;
  --bad: #f85149; --badbg: #3d1518; --idle: #8b949e; --idlebg: #1c2128;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 32px 16px 80px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 1.15rem; margin: 40px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
h3 { font-size: 0.98rem; margin: 0; font-weight: 600; }
.sub { color: var(--muted); font-size: 0.85rem; margin-bottom: 28px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 10px; margin-bottom: 8px; }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
.tile .n { font-size: 1.5rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.tile .k { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 2px; }
.tile.bad .n { color: var(--bad); } .tile.warn .n { color: var(--warn); } .tile.ok .n { color: var(--ok); }
.phase { margin: 26px 0 0; }
.phase-h { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.phase-h .goal { color: var(--muted); font-size: 0.82rem; }
.step { border: 1px solid var(--line); border-radius: 8px; margin: 10px 0; background: var(--panel); overflow: hidden; }
.step > summary { padding: 11px 14px; cursor: pointer; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.step > summary::-webkit-details-marker { display: none; }
.step > summary:hover { background: color-mix(in srgb, var(--accent) 7%, transparent); }
.step-body { padding: 0 14px 16px; border-top: 1px solid var(--line); }
.why { color: var(--muted); font-size: 0.86rem; margin: 12px 0; }
.pill { font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.03em; white-space: nowrap; }
.p-complete { background: var(--okbg); color: var(--ok); }
.p-running { background: var(--warnbg); color: var(--warn); }
.p-partial { background: var(--warnbg); color: var(--warn); }
.p-thresholds-failed { background: var(--badbg); color: var(--bad); }
.p-blocked { background: var(--badbg); color: var(--bad); }
.p-not-started { background: var(--idlebg); color: var(--idle); }
.gate { background: var(--badbg); color: var(--bad); }
.sid { font-family: var(--mono); font-size: 0.8rem; color: var(--muted); min-width: 34px; }
.grow { flex: 1 1 auto; min-width: 140px; }
.count { color: var(--muted); font-size: 0.78rem; font-variant-numeric: tabular-nums; }
.scroller { overflow-x: auto; margin: 10px 0; }
table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.scroller table { min-width: max-content; }
th, td { text-align: left; padding: 6px 9px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; }
td.num { font-variant-numeric: tabular-nums; font-family: var(--mono); }
code { font-family: var(--mono); font-size: 0.85em; background: color-mix(in srgb, var(--fg) 7%, transparent); padding: 1px 5px; border-radius: 4px; }
.flag { background: var(--badbg); color: var(--bad); border-radius: 8px; padding: 10px 14px; margin: 8px 0; font-size: 0.86rem; }
.flag.warn { background: var(--warnbg); color: var(--warn); }
.flag b { font-weight: 650; }
.note { background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 6px; padding: 10px 14px; margin: 10px 0; font-size: 0.86rem; }
ul.rem { margin: 8px 0; padding-left: 20px; font-size: 0.85rem; color: var(--muted); }
ul.rem li { margin: 3px 0; }
a { color: var(--accent); }
.empty { color: var(--muted); font-size: 0.85rem; font-style: italic; }
@media (max-width: 640px) {
  .wrap { padding: 20px 16px 60px; }
  table { font-size: 0.78rem; }
  th, td { padding: 5px 6px; }
}
"""


def _e(s: object) -> str:
    return html.escape(str(s if s is not None else ""))


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.4g}"


def _threshold_rows(st: StepStatus) -> str:
    if not st.step.thresholds:
        if st.needs_prereg:
            return (
                '<div class="flag">⚠ <b>No thresholds declared.</b> This step is marked '
                "<code>prereg_required</code> — fix the numbers in "
                "<code>PREREGISTRATION.md</code> and add them to <code>plan.yaml</code> "
                "before running it.</div>"
            )
        return ""
    rows = "".join(
        f"<tr><td><code>{_e(t.description)}</code></td><td>{_e(t.hypothesis)}</td>"
        f'<td class="num">{_fmt(t.observed)}</td>'
        f'<td><span class="pill p-{"complete" if t.outcome == "pass" else "thresholds-failed" if t.outcome == "fail" else "not-started"}">'
        f"{_e(t.outcome)}</span></td>"
        f'<td class="num">{_e(" · ".join(f"{v:.4g} ({lab})" for v, lab in t.values[:6])) or "—"}</td>'
        "</tr>"
        for t in st.thresholds
    )
    return (
        '<div class="scroller"><table><thead><tr><th>Pre-registered condition</th>'
        "<th>Hypothesis</th><th>Observed</th><th>Outcome</th><th>Per variant</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _runs_table(st: StepStatus) -> str:
    if not st.runs:
        return '<p class="empty">No runs recorded.</p>'

    metric_names = sorted({m for r in st.runs for m in r.metrics})
    head = (
        "<tr><th>Run</th><th>Experiment</th><th>Variant</th><th>Status</th>"
        + "".join(f"<th>{_e(m)}</th>" for m in metric_names)
        + "<th>Code</th><th>Links</th></tr>"
    )

    body = []
    for r in st.runs:
        flags = []
        if r.git_dirty:
            flags.append('<span class="pill p-thresholds-failed" title="uncommitted or unverifiable working tree">dirty</span>')
        sha = f"<code>{_e(r.git_sha[:7])}</code>" if r.git_sha else "—"
        links = []
        if r.wandb_url:
            links.append(f'<a href="{_e(r.wandb_url)}">curves</a>')
        for a in r.artifacts:
            links.append(f'<a href="{_e(a["path"])}">{_e(a["kind"] or Path(a["path"]).name)}</a>')
        state = (
            "complete" if r.status == "completed"
            else "running" if r.status == "running"
            else "thresholds-failed"
        )
        body.append(
            f'<tr><td><code>{_e(r.short_id)}</code></td>'
            f"<td>{_e(r.experiment_id)}</td>"
            f"<td>{_e(variant_label(r.variant))}</td>"
            f'<td><span class="pill p-{state}">{_e(r.status)}</span></td>'
            + "".join(f'<td class="num">{_fmt(r.metrics.get(m))}</td>' for m in metric_names)
            + f"<td>{sha} {' '.join(flags)}</td>"
            f"<td>{' '.join(links) or '—'}</td></tr>"
        )

    return (
        f'<div class="scroller"><table><thead>{head}</thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _step_html(st: StepStatus) -> str:
    s = st.step
    pills = [f'<span class="pill p-{st.state}">{_e(st.state)}</span>']
    if s.gate:
        pills.append('<span class="pill gate">gate</span>')
    if s.size:
        pills.append(f'<span class="count">{_e(s.size)}</span>')

    counts = []
    if st.n_completed:
        counts.append(f"{st.n_completed} done")
    if st.n_running:
        counts.append(f"{st.n_running} running")
    if st.n_failed:
        counts.append(f"{st.n_failed} failed")
    if st.missing_experiments:
        counts.append(f"{len(st.missing_experiments)} experiments pending")

    verdict = ""
    if st.verdict:
        v = st.verdict
        cls = "complete" if v["decision"] == "go" else "thresholds-failed"
        verdict = (
            f'<div class="note"><span class="pill p-{cls}">verdict: {_e(v["decision"])}</span> '
            f'<b>{_e(v["decided_at"])}</b>'
            + (f" — {_e(v['rationale'])}" if v["rationale"] else "")
            + "</div>"
        )

    missing = ""
    if st.missing_experiments:
        missing = (
            '<div class="flag warn">Not yet run: '
            + ", ".join(f"<code>{_e(m)}</code>" for m in st.missing_experiments)
            + "</div>"
        )

    return (
        f'<details class="step"{" open" if st.state in {"running", "thresholds-failed"} else ""}>'
        f'<summary><span class="sid">{_e(s.id)}</span>'
        f'<h3 class="grow">{_e(s.name)}</h3>'
        f'{"".join(pills)}<span class="count">{_e(" · ".join(counts))}</span></summary>'
        f'<div class="step-body">'
        f'{f"<p class=why>{_e(s.why)}</p>" if s.why else ""}'
        f"{verdict}{_threshold_rows(st)}{missing}{_runs_table(st)}"
        f"</div></details>"
    )


def _flags_html(ps: PlanStatus) -> str:
    out = []
    if ps.missing_prereg:
        ids = ", ".join(f"<code>{_e(s.step.id)}</code>" for s in ps.missing_prereg)
        out.append(
            f'<div class="flag">⚠ <b>{len(ps.missing_prereg)} step(s) need pre-registered '
            f"thresholds</b> before running: {ids}</div>"
        )
    if ps.dirty_runs:
        out.append(
            f'<div class="flag">⚠ <b>{len(ps.dirty_runs)} completed run(s) have a dirty or '
            "unverifiable working tree.</b> These are not reproducible from a commit and "
            "must not be used as headline numbers.</div>"
        )
    if ps.stale_plan_runs:
        out.append(
            f'<div class="flag warn">{len(ps.stale_plan_runs)} run(s) were recorded against an '
            "older <code>plan.yaml</code>. Their thresholds may have changed since.</div>"
        )
    if ps.failed_runs:
        out.append(
            f'<div class="flag warn">{len(ps.failed_runs)} failed run(s) on record. '
            "Kept deliberately — a crashed experiment should leave a trace, not a gap.</div>"
        )
    if ps.duplicate_variants:
        rows = ", ".join(
            f"<code>{_e(s)}/{_e(e)}</code>×{n}" for s, e, _k, n in ps.duplicate_variants[:8]
        )
        out.append(
            f'<div class="flag warn">Repeated identical variants: {rows}. '
            "Expected for seed replication; otherwise a sign of an accidental re-run.</div>"
        )
    if ps.open_gates:
        ids = ", ".join(f"<code>{_e(s.step.id)}</code>" for s in ps.open_gates)
        out.append(
            f'<div class="flag warn">Open gate(s) with no recorded verdict: {ids}. '
            "Record one with <code>bdtrack verdict &lt;step&gt; go|no-go|rescope -m …</code></div>"
        )
    return "".join(out) or '<p class="empty">No flags.</p>'


def render_html(ps: PlanStatus) -> str:
    """builds the whole dashboard as one string."""
    s = ps.summary()
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    tiles = [
        ("complete", s["steps_complete"], f"of {s['steps_total']} steps", "ok"),
        ("runs", s["runs_total"], "recorded", ""),
        ("gates open", s["gates_open"], "undecided", "warn" if s["gates_open"] else ""),
        ("prereg missing", s["steps_missing_prereg"], "steps", "bad" if s["steps_missing_prereg"] else ""),
        ("dirty runs", s["runs_dirty"], "not reproducible", "bad" if s["runs_dirty"] else ""),
        ("failed runs", s["runs_failed"], "on record", "warn" if s["runs_failed"] else ""),
    ]
    tiles_html = "".join(
        f'<div class="tile {cls}"><div class="n">{v}</div><div class="k">{_e(k)}</div>'
        f'<div class="k" style="text-transform:none;letter-spacing:0">{_e(sub)}</div></div>'
        for k, v, sub, cls in tiles
    )

    nxt = ps.next_steps
    next_html = (
        '<div class="note"><b>Next unfinished step per phase:</b> '
        + ", ".join(f"<code>{_e(x.step.id)}</code> {_e(x.step.name)}" for x in nxt[:6])
        + "</div>"
        if nxt
        else ""
    )

    phases = []
    for phase in ps.plan.phases:
        steps_html = "".join(_step_html(ps.by_id(st.id)) for st in phase.steps)
        phases.append(
            f'<div class="phase"><div class="phase-h">'
            f"<h2 style='margin:18px 0 4px;border:0'>Phase {_e(phase.id)} — {_e(phase.name)}</h2>"
            f"</div><div class='goal sub' style='margin:0 0 6px'>{_e(phase.goal)}</div>"
            f"{steps_html}</div>"
        )

    reminders = "".join(f"<li>{_e(r['text'])}</li>" for r in ps.plan.reminders)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(ps.plan.project)} Tracker</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>{_e(ps.plan.project)} — experiment tracker</h1>
  <div class="sub">
    Generated {_e(generated)} · plan <code>{_e(ps.plan.plan_hash)}</code> ·
    {_e(s["runs_total"])} runs · source <code>{_e(ps.plan.source.name)}</code>
  </div>

  <div class="tiles">{tiles_html}</div>
  {next_html}

  <h2>Flags</h2>
  {_flags_html(ps)}

  <h2>Plan</h2>
  {"".join(phases)}

  <h2>Standing reminders</h2>
  <ul class="rem">{reminders}</ul>
</div>
</body>
</html>
"""


def write_html(ps: PlanStatus, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(ps), encoding="utf-8")
    return path


def export_json(ps: PlanStatus, path: str | Path) -> Path:
    """machine-readable export. this is the git-committed record."""
    payload = {
        "summary": ps.summary(),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "steps": [
            {
                "id": st.step.id,
                "name": st.step.name,
                "phase": st.step.phase_id,
                "gate": st.step.gate,
                "state": st.state,
                "verdict": st.verdict,
                "thresholds": [
                    {
                        "condition": t.description,
                        "hypothesis": t.hypothesis,
                        "observed": t.observed,
                        "outcome": t.outcome,
                        "aggregate": t.aggregate,
                        "per_variant": [
                            {"value": v, "variant": lab} for v, lab in t.values
                        ],
                    }
                    for t in st.thresholds
                ],
                "missing_experiments": st.missing_experiments,
                "runs": [
                    {
                        "id": r.id,
                        "experiment": r.experiment_id,
                        "variant": r.variant,
                        "status": r.status,
                        "started_at": r.started_at,
                        "duration_s": r.duration_s,
                        "git_sha": r.git_sha,
                        "git_dirty": r.git_dirty,
                        "config_hash": r.config_hash,
                        "dataset_hash": r.dataset_hash,
                        "seed": r.seed,
                        "wandb_url": r.wandb_url,
                        "metrics": r.metrics,
                        "artifacts": r.artifacts,
                        "error": r.error,
                    }
                    for r in st.runs
                ],
            }
            for st in ps.steps
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def export_markdown(ps: PlanStatus, path: str | Path) -> Path:
    """human-readable committed summary, so results diff readably in git."""
    s = ps.summary()
    L = [
        f"# {ps.plan.project} — results summary",
        "",
        f"_Generated from `{ps.plan.source.name}` (`{ps.plan.plan_hash}`). "
        "Do not edit by hand; regenerate with `bdtrack export`._",
        "",
        f"- Steps complete: **{s['steps_complete']}/{s['steps_total']}**",
        f"- Runs recorded: **{s['runs_total']}** "
        f"({s['runs_failed']} failed, {s['runs_dirty']} from a dirty tree)",
        f"- Gates open: **{s['gates_open']}** · steps missing pre-registration: "
        f"**{s['steps_missing_prereg']}**",
        "",
    ]

    for phase in ps.plan.phases:
        L += [f"## Phase {phase.id} — {phase.name}", ""]
        for step in phase.steps:
            st = ps.by_id(step.id)
            gate = " **[GATE]**" if step.gate else ""
            L.append(f"### {step.id} — {step.name}{gate} · `{st.state}`")
            if st.verdict:
                L.append(
                    f"- Verdict: **{st.verdict['decision']}** "
                    f"({st.verdict['decided_at']}) {st.verdict['rationale']}"
                )
            for t in st.thresholds:
                mark = {"pass": "✅", "fail": "❌", "no-data": "⬜"}[t.outcome]
                L.append(f"- {mark} `{t.description}` → observed `{_fmt(t.observed)}`")
            if st.missing_experiments:
                L.append(
                    "- Pending: " + ", ".join(f"`{m}`" for m in st.missing_experiments)
                )
            completed = [r for r in st.runs if r.status == "completed"]
            if completed:
                L.append("")
                L.append("| Run | Experiment | Variant | Metrics | Code |")
                L.append("|---|---|---|---|---|")
                for r in completed:
                    mets = ", ".join(f"{k}={_fmt(v)}" for k, v in r.metrics.items()) or "—"
                    sha = (r.git_sha[:7] if r.git_sha else "—") + (" ⚠dirty" if r.git_dirty else "")
                    L.append(
                        f"| `{r.short_id}` | {r.experiment_id} | "
                        f"{variant_label(r.variant)} | {mets} | {sha} |"
                    )
            L.append("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path
