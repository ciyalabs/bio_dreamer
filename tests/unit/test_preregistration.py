"""Pre-registration integrity — Step 0.6.

PREREGISTRATION.md is the human-readable commitment; plan.yaml is what
`bdtrack status` actually checks. If they drift, the committed thresholds stop
meaning anything and nobody notices. These tests tie them together and guard the
properties that make a pre-registration worth having at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from biodreamer.tracking import load_plan

PREREG = Path("PREREGISTRATION.md")

# every (step, metric) the document commits to, by hypothesis
COMMITTED = {
    "1.1": {"nonadditive_variance_fraction", "greedy_optimum_hit_rate"},
    "2.2": {
        "spearman_mean",
        "margin_over_additive_null",
        "frac_assays_beating_null",
        "spearman_mean_vs_frozen_regression",
    },
    "4.2": {
        "error_growth_slope_ratio",
        "error_growth_gap_over_seed_sd",
        "candidate_diversity_ratio",
        "n_seeds",
    },
    "6.2": {
        "policy_gap_over_seed_sd",
        "n_seeds",
        "spearman_calibration_vs_policy_rank",
        "eta_zero_reported",
    },
    "6.3": {
        "optimum_hit_rate_margin_over_best_baseline",
        "best_fitness_margin_over_best_baseline",
        "correlation_with_additive",
        "margin_on_negative_controls",
        "n_seeds",
    },
}


@pytest.fixture(scope="module")
def plan():
    return load_plan()


@pytest.fixture(scope="module")
def prereg_text():
    if not PREREG.exists():
        pytest.fail("PREREGISTRATION.md is missing — it is the deliverable of step 0.6")
    return PREREG.read_text()


def test_preregistration_exists_and_is_dated(prereg_text):
    assert re.search(r"\*\*Registered:\*\*\s*\d{4}-\d{2}-\d{2}", prereg_text), (
        "a pre-registration without a date is not a pre-registration"
    )


def test_no_step_marked_prereg_required_lacks_thresholds(plan):
    """The condition the tracker flags; it must be empty once 0.6 is done."""
    missing = [s.id for s in plan.steps() if s.prereg_required and not s.thresholds]
    assert not missing, f"steps still need thresholds: {missing}"


@pytest.mark.parametrize("step_id,metrics", sorted(COMMITTED.items()))
def test_plan_declares_every_committed_metric(plan, step_id, metrics):
    declared = {t.metric for t in plan.step(step_id).thresholds}
    assert metrics <= declared, (
        f"step {step_id}: PREREGISTRATION.md commits to {sorted(metrics - declared)} "
        "but plan.yaml does not declare it"
    )


@pytest.mark.parametrize("step_id", sorted(COMMITTED))
def test_plan_declares_no_undocumented_metric(plan, step_id):
    """A threshold in plan.yaml that the document does not mention is unregistered."""
    declared = {t.metric for t in plan.step(step_id).thresholds}
    extra = declared - COMMITTED[step_id]
    assert not extra, f"step {step_id}: plan.yaml has undocumented thresholds {sorted(extra)}"


def test_h4_falsification_threshold_is_present_and_strict(plan):
    """The one that can fail H4 regardless of how well the model scores."""
    t = next(
        x for x in plan.step("6.3").thresholds if x.metric == "correlation_with_additive"
    )
    assert t.op == "<=" and t.value == pytest.approx(0.95)
    assert t.aggregate == "max", "the most additive-looking landscape must decide"
    assert "falsif" in t.hypothesis.lower()


def test_h4_has_a_negative_control_condition(plan):
    """A planner winning on near-additive controls indicates reward exploitation."""
    t = next(
        x for x in plan.step("6.3").thresholds if x.metric == "margin_on_negative_controls"
    )
    assert t.op == "<=" and t.value <= 0.05


def test_step_1_1_gate_is_existential_not_universal(plan):
    """'all' would fail on the near-additive controls by design, which is wrong:
    the project needs one landscape with headroom, not every landscape."""
    by = {t.metric: t for t in plan.step("1.1").thresholds}
    assert by["nonadditive_variance_fraction"].aggregate == "max"
    assert by["greedy_optimum_hit_rate"].aggregate == "min"


def test_seed_counts_are_at_least_three_everywhere(plan):
    """No headline number from a single seed — a standing rule in the document."""
    for s in plan.steps():
        for t in s.thresholds:
            if t.metric == "n_seeds":
                assert t.value >= 3, f"step {s.id}: n_seeds >= {t.value} is too weak"


def test_document_records_the_pre_phase1_revisions(prereg_text):
    """H4 was reframed before Phase 1; that must be visible, not quietly applied."""
    for marker in ("greedy", "TEV", "oracle"):
        assert marker.lower() in prereg_text.lower()
    assert "Pre-Phase-1 revisions" in prereg_text


def test_amendment_table_exists(prereg_text):
    """Later edits must have somewhere to be logged, with the original kept visible."""
    assert "## Amendments" in prereg_text


def test_greedy_is_defined_as_restart_based(prereg_text):
    """Step 0.5 showed the restart count decides the comparison, so it is fixed."""
    assert re.search(r"5 random restarts", prereg_text), (
        "the document must fix which greedy variant is the baseline"
    )
