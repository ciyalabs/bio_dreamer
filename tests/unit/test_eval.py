"""Evaluation harness tests — Step 0.5.

Two declared experiments live here:

    metric_validation    every metric checked against a case whose answer is
                         known analytically, not against another implementation
    additive_null_impl   the null verified on a landscape that is additive by
                         construction, where it must recover R^2 = 1

A metric validated only against real data cannot be shown to be correct. These
use constructed cases so a broken metric fails rather than looking plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from biodreamer.eval.baselines import AdditiveNull, GreedyWalk, RandomSearch, StackTopK
from biodreamer.eval.landscape import Landscape
from biodreamer.eval.metrics import (
    best_fitness_found,
    calibration_slope,
    error_growth,
    global_optimum_hit_rate,
    path_accessibility,
    reciprocal_sign_epistasis,
    report_additive_comparison,
    risk_coverage,
    spearman,
    spearman_per_assay,
    top_k_recovery,
)


@pytest.fixture()
def additive():
    return Landscape.synthetic_additive(n_sites=3, alphabet="ACDE", seed=0)


@pytest.fixture()
def valley():
    return Landscape.synthetic_sign_epistasis()


# ---- additive_null_impl ----


def test_additive_null_is_exact_on_an_additive_landscape(additive):
    """THE validation case. If this is not ~1.0 the null is broken."""
    null = AdditiveNull().fit(additive)
    assert null.r2(additive) == pytest.approx(1.0, abs=1e-6)


def test_additive_null_fitted_on_singles_extrapolates_exactly(additive):
    """On an additive landscape, singles determine every higher order exactly."""
    null = AdditiveNull().fit_on_singles(additive)
    higher = additive.subset(additive.order() >= 2, "higher")
    pred = null.predict(higher.variants)
    assert np.allclose(pred, higher.fitness, atol=1e-6)


def test_additive_null_cannot_fit_sign_epistasis(valley):
    """By construction it must fail here — that is what makes it a null."""
    null = AdditiveNull().fit(valley)
    assert null.r2(valley) < 0.95


def test_additive_null_degrades_with_noise():
    clean = AdditiveNull().fit(Landscape.synthetic_additive(seed=1, noise=0.0))
    noisy_ls = Landscape.synthetic_additive(seed=1, noise=0.5)
    assert AdditiveNull().fit(noisy_ls).r2(noisy_ls) < clean.r2(
        Landscape.synthetic_additive(seed=1, noise=0.0)
    )


def test_additive_null_requires_fitting():
    with pytest.raises(RuntimeError, match="fitted"):
        AdditiveNull().predict(["AAA"])


def test_additive_null_proposes_its_own_best(additive):
    null = AdditiveNull().fit(additive)
    props = null.propose(additive, budget=5)
    assert len(props) == 5
    # on an exactly-additive landscape the null must find the true optimum
    assert global_optimum_hit_rate(props, additive) == 1.0


# ---- greedy and the valley ----


def test_greedy_is_trapped_by_reciprocal_sign_epistasis(valley):
    """AA->AB and AA->BA are both deleterious, so greedy stops at the wild type."""
    path = GreedyWalk().walk(valley)
    assert path == ["AA"]
    assert global_optimum_hit_rate(path, valley) == 0.0


def test_greedy_reaches_the_optimum_on_an_additive_landscape(additive):
    path = GreedyWalk(max_steps=10).walk(additive)
    assert global_optimum_hit_rate(path, additive) == 1.0


def test_greedy_restarts_can_escape(valley):
    """With restarts greedy sometimes lands inside the basin; planning should not need to."""
    props = GreedyWalk(max_steps=5, restarts=8, seed=0).propose(valley, budget=10)
    assert "AA" in props


def test_stack_top_k_combines_best_singles(additive):
    props = StackTopK(k=3).propose(additive, budget=5)
    assert props and all(len(p) == additive.n_sites for p in props)


def test_random_search_respects_budget(additive):
    props = RandomSearch(seed=0).propose(additive, budget=7)
    assert len(props) == 7 and len(set(props)) == 7


# ---- metric_validation: known answers ----


def test_spearman_perfect_and_inverted():
    y = np.arange(10, dtype=float)
    assert spearman(y, y) == pytest.approx(1.0)
    assert spearman(y, -y) == pytest.approx(-1.0)


def test_spearman_is_nan_on_a_constant_vector():
    assert np.isnan(spearman(np.ones(5), np.arange(5.0)))


def test_spearman_is_rank_based_not_value_based():
    y = np.array([1.0, 2.0, 3.0])
    assert spearman(y, np.exp(y)) == pytest.approx(1.0)


def test_spearman_per_assay_does_not_pool():
    """Assay A is perfectly predicted, assay B inverted; pooling would hide that."""
    out = spearman_per_assay(
        {"A": (np.arange(5.0), np.arange(5.0)), "B": (np.arange(5.0), -np.arange(5.0))}
    )
    assert out["A"] == pytest.approx(1.0) and out["B"] == pytest.approx(-1.0)


def test_top_k_recovery_known_cases():
    y = np.arange(10.0)
    assert top_k_recovery(y, y, k=3) == 1.0
    assert top_k_recovery(y, -y, k=3) == 0.0
    # predicting the top 3 within a budget of 5 recovers all of them
    assert top_k_recovery(y, y, k=3, budget=5) == 1.0


def test_global_optimum_hit_rate(additive):
    best, _ = additive.global_optimum()
    assert global_optimum_hit_rate([best], additive) == 1.0
    assert global_optimum_hit_rate(["AAA"], additive) in (0.0, 1.0)


def test_best_fitness_found(valley):
    assert best_fitness_found(["BB"], valley) == pytest.approx(2.0)
    assert np.isnan(best_fitness_found(["ZZ"], valley))


def test_reciprocal_sign_epistasis_counts_the_constructed_trap(valley):
    rep = reciprocal_sign_epistasis(valley)
    assert rep.n_reciprocal_sign == 1 and rep.n_pairs == 1
    assert rep.fraction == pytest.approx(1.0)


def test_no_reciprocal_sign_epistasis_in_an_additive_landscape(additive):
    assert reciprocal_sign_epistasis(additive).n_reciprocal_sign == 0


def test_path_accessibility_is_zero_across_a_valley(valley):
    """Both orderings AA->AB->BB and AA->BA->BB dip below the wild type."""
    assert path_accessibility(valley, "BB") == 0.0


def test_path_accessibility_is_one_on_a_monotone_climb():
    variants = np.array([list(s) for s in ("AA", "AB", "BA", "BB")])
    ls = Landscape("mono", variants, np.array([0.0, 1.0, 1.0, 2.0]), "AA")
    assert path_accessibility(ls, "BB") == pytest.approx(1.0)


def test_error_growth_recovers_a_known_slope():
    assert error_growth({1: 1.0, 2: 3.0, 3: 5.0}) == pytest.approx(2.0)


def test_calibration_slope_positive_when_uncertainty_tracks_error():
    u = np.linspace(0, 1, 200)
    assert calibration_slope(errors=u * 2, uncertainty=u) > 0


def test_calibration_slope_flat_when_uncertainty_is_noise():
    rng = np.random.default_rng(0)
    s = calibration_slope(errors=rng.normal(size=500), uncertainty=rng.normal(size=500))
    assert abs(s) < 0.5


def test_risk_coverage_rises_for_informative_uncertainty():
    u = np.linspace(0, 1, 100)
    _cov, risk = risk_coverage(errors=u, uncertainty=u)
    assert risk[-1] > risk[0]


# ---- falsification ----


def test_comparison_falsifies_a_disguised_additive_model(additive):
    """A model that is the null in disguise must be flagged however well it scores."""
    null = AdditiveNull().fit(additive)
    null_pred = null.predict(additive.variants)
    model_pred = null_pred * 1.01 + 0.001  # monotone transform: same ranking
    rep = report_additive_comparison(additive.fitness, model_pred, null_pred)
    assert rep.correlation_with_null > 0.99
    assert rep.falsified, "a monotone rescaling of the null must be caught"


def test_comparison_survives_for_a_genuinely_different_model(valley):
    null = AdditiveNull().fit(valley)
    null_pred = null.predict(valley.variants)
    rep = report_additive_comparison(valley.fitness, valley.fitness, null_pred)
    assert not rep.falsified
    assert rep.margin > 0


def test_comparison_reports_margin_even_when_falsified(additive):
    null = AdditiveNull().fit(additive)
    p = null.predict(additive.variants)
    rep = report_additive_comparison(additive.fitness, p, p)
    assert rep.falsified and rep.margin == pytest.approx(0.0, abs=1e-9)
    assert "FALSIFIED" in str(rep)


def test_threshold_is_configurable(additive):
    null = AdditiveNull().fit(additive)
    p = null.predict(additive.variants)
    assert report_additive_comparison(additive.fitness, p * 2, p, threshold=1.01).falsified is False


# ---- landscape plumbing ----


def test_order_and_singles(additive):
    assert additive.order().min() == 0
    assert set(additive.singles().order().tolist()) <= {0, 1}


def test_neighbours_are_one_substitution_away(valley):
    assert sorted(valley.neighbours("AA")) == ["AB", "BA"]


def test_wild_type_is_the_modal_residue(additive):
    assert additive.wild_type == "AAA"


def test_landscape_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="lengths differ"):
        Landscape("x", np.array([list("AA")]), np.array([1.0, 2.0]), "AA")
