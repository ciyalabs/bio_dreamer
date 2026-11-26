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

from pathlib import Path

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


# ---- oracle accounting (Step 1.1) ----


def test_counting_oracle_charges_once_per_distinct_sequence():
    """A campaign remembers what it measured, so a repeat query is free."""
    from biodreamer.eval.baselines import CountingOracle

    ls = Landscape.synthetic_additive(n_sites=2, alphabet="AC")
    oracle = CountingOracle(ls)
    seq = ls.sequences()[0]
    for _ in range(5):
        oracle(seq)
    assert oracle.queries == 5
    assert oracle.distinct == 1


def test_counting_oracle_returns_the_same_values_as_the_landscape_index():
    from biodreamer.eval.baselines import CountingOracle

    ls = Landscape.synthetic_additive(n_sites=3, alphabet="ACD")
    oracle = CountingOracle(ls)
    for seq, fit in zip(ls.sequences(), ls.fitness):
        assert oracle(seq) == pytest.approx(fit)


def test_counting_oracle_reports_the_best_sequence_it_measured():
    from biodreamer.eval.baselines import CountingOracle

    ls = Landscape.synthetic_sign_epistasis()
    oracle = CountingOracle(ls)
    for s in ("AA", "AB"):
        oracle(s)
    seq, fit = oracle.best()
    assert seq == "AA" and fit == pytest.approx(1.0)
    # the true optimum was never measured, so it cannot be reported
    assert fit < ls.global_optimum()[1]


def test_counting_oracle_is_empty_before_any_query():
    from biodreamer.eval.baselines import CountingOracle

    oracle = CountingOracle(Landscape.synthetic_sign_epistasis())
    assert oracle.distinct == 0
    assert oracle.best()[0] == ""


def test_greedy_costs_far_more_oracle_calls_than_the_registered_h4_budget():
    """The inconsistency Step 1.1 found: H4 fixes B=100 true evaluations, while
    "greedy" is fixed at 5 restarts of 20 steps. One step over 20 residues at four
    sites evaluates 76 neighbours, so the two definitions differ by orders of
    magnitude and cannot both hold."""
    from biodreamer.eval.baselines import CountingOracle, GreedyWalk

    ls = Landscape.synthetic_additive(n_sites=4, alphabet="ACDEFGHIKL")
    oracle = CountingOracle(ls)
    GreedyWalk(max_steps=20, restarts=5, seed=0).propose(ls, 10**6, oracle=oracle)
    assert oracle.distinct > 100


def test_stack_topk_costs_one_call_per_single_mutant():
    from biodreamer.eval.baselines import CountingOracle, StackTopK

    ls = Landscape.synthetic_additive(n_sites=3, alphabet="ACDE")
    oracle = CountingOracle(ls)
    StackTopK(k=3).propose(ls, 3, oracle=oracle)
    # 3 sites x 3 non-wild-type residues, plus whatever the stacking evaluates
    assert oracle.distinct >= 9


# ---- path accessibility sampling (Step 1.1) ----


def test_path_accessibility_sampling_approximates_the_exact_enumeration():
    """9! = 362,880 paths at the dimension-9 hypercubes, so the cap matters."""
    ls = Landscape.synthetic_additive(n_sites=4, alphabet="AC")
    exact = path_accessibility(ls)
    sampled = path_accessibility(ls, max_paths=200)
    assert sampled == pytest.approx(exact, abs=0.1)


def test_path_accessibility_is_exact_when_under_the_cap():
    ls = Landscape.synthetic_additive(n_sites=3, alphabet="AC")
    assert path_accessibility(ls, max_paths=10**6) == pytest.approx(
        path_accessibility(ls)
    )


def test_path_accessibility_on_a_constructed_valley_is_zero():
    """Reciprocal sign epistasis: every shortest path goes down before it goes up."""
    ls = Landscape.synthetic_sign_epistasis()
    assert path_accessibility(ls) == 0.0


# ---- wild-type resolution (added 2026-11-26) ----

SSMULA_DIR = Path("data/raw/ssmula/extracted/data")


def _csv(rel: str) -> Path:
    return SSMULA_DIR / rel


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
@pytest.mark.parametrize(
    "name,rel,wt,fitness",
    [
        ("GB1", "GB1/fitness_landscape/GB1.csv", "VDGV", 1.0),
        ("TEV", "TEV/fitness_landscape/TEV.csv", "TDHS", 0.4144),
        ("TrpB4", "TrpB/fitness_landscape/TrpB4.csv", "VFVS", 0.4081),
    ],
)
def test_the_real_wild_type_is_recovered_not_the_modal_variant(name, rel, wt, fitness):
    """The modal residue per site is meaningless on a combinatorially complete
    landscape -- every residue appears at every site by design -- and in practice it
    returns a dead variant. It gave WLVS at fitness 0.0039 for GB1 instead of VDGV at
    1.0000, and FLPP at -0.0162 for TEV instead of TDHS at 0.4144. Every
    wild-type-anchored metric was computed from a dead variant."""
    ls = Landscape.from_csv(_csv(rel), name)
    assert ls.wild_type == wt
    assert ls.wild_type_source != "modal"
    assert ls.wild_type_fitness() == pytest.approx(fitness, abs=1e-3)


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
def test_the_modal_wild_type_would_have_been_a_dead_variant():
    """Pins the magnitude of the error, so nobody reintroduces the shortcut."""
    ls = Landscape.from_csv(_csv("GB1/fitness_landscape/GB1.csv"), "GB1")
    idx = ls.index()
    assert idx["WLVS"] < 0.01, "the modal variant is dead"
    assert idx["VDGV"] == pytest.approx(1.0), "the real wild type is the reference"


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
def test_an_unresolvable_wild_type_warns_and_is_marked(caplog):
    """TrpB3A has no distance column, no registry entry, and no unique 1.0 anchor, so
    its wild type cannot be recovered from shipped data. That must be loud, not silent.
    ParD2 used to be the example here until the normalisation anchor resolved it."""
    import logging

    with caplog.at_level(logging.WARNING):
        ls = Landscape.from_csv(_csv("TrpB/fitness_landscape/TrpB3A.csv"), "TrpB3A")
    assert ls.wild_type_source == "modal"
    assert "modal residue" in caplog.text


def test_an_explicit_wild_type_overrides_inference():
    v = np.array([list(s) for s in ("AA", "AB", "BA", "BB")])
    ls = Landscape("x", v, np.array([1.0, 2.0, 3.0, 4.0]), "BB", wild_type_source="explicit")
    assert ls.wild_type == "BB"


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
def test_explicit_wild_type_is_honoured_by_from_csv():
    ls = Landscape.from_csv(_csv("GB1/fitness_landscape/GB1.csv"), "GB1", wild_type="WLVS")
    assert ls.wild_type == "WLVS"
    assert ls.wild_type_source == "explicit"


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
def test_a_wrong_length_wild_type_is_rejected():
    with pytest.raises(ValueError, match="has 3 sites, landscape has 4"):
        Landscape.from_csv(_csv("GB1/fitness_landscape/GB1.csv"), "GB1", wild_type="ABC")


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
@pytest.mark.parametrize(
    "name,rel,wt", [("ParD2", "ParD2/fitness_landscape/ParD2.csv", "ILK"),
                    ("ParD3", "ParD3/fitness_landscape/ParD3.csv", "DKE")]
)
def test_pard_wild_types_come_from_the_normalisation_anchor(name, rel, wt):
    """Neither ships a distance column. Both have exactly one variant at the literal
    float 1.0, and in neither case is it the maximum -- so the 1.0 is a reference
    point, not a ceiling. The same signature identifies GB1's VDGV, where the HD==0
    column confirms it independently."""
    ls = Landscape.from_csv(_csv(rel), name)
    assert ls.wild_type == wt
    assert ls.wild_type_source != "modal"
    assert ls.wild_type_fitness() == 1.0
    assert ls.global_optimum()[1] > 1.0, "the anchor must not also be the maximum"


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
def test_the_anchor_rule_declines_to_fire_where_it_should():
    """On TrpB4 the unique 1.0 IS the maximum, so it is a ceiling and says nothing
    about the wild type; TEV has no variant at exactly 1.0 at all. Both are resolved
    by other means, and a rule that fired here would be wrong."""
    trpb = Landscape.from_csv(_csv("TrpB/fitness_landscape/TrpB4.csv"), "TrpB4")
    assert trpb.global_optimum()[1] == 1.0
    assert trpb.wild_type_fitness() < 1.0
    tev = Landscape.from_csv(_csv("TEV/fitness_landscape/TEV.csv"), "TEV")
    assert not any(f == 1.0 for f in tev.index().values())


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
def test_a_plausible_wild_type_is_not_beaten_by_most_of_its_landscape():
    """The diagnostic that would have caught the dead-variant bug a month earlier.
    The Step 0.2 audit reported 45-57% and nobody looked."""
    for name, rel in [
        ("GB1", "GB1/fitness_landscape/GB1.csv"),
        ("TEV", "TEV/fitness_landscape/TEV.csv"),
        ("TrpB4", "TrpB/fitness_landscape/TrpB4.csv"),
        ("ParD2", "ParD2/fitness_landscape/ParD2.csv"),
        ("ParD3", "ParD3/fitness_landscape/ParD3.csv"),
    ]:
        ls = Landscape.from_csv(_csv(rel), name)
        frac = float((ls.fitness > ls.wild_type_fitness()).mean())
        assert frac < 0.20, f"{name}: {frac:.1%} of variants beat the wild type"


def test_variable_positions_spell_the_registered_wild_type():
    """Only landscapes whose positions are actually determined are registered."""
    from biodreamer.eval.landscape import KNOWN_WILD_TYPES, VARIABLE_POSITIONS

    assert set(VARIABLE_POSITIONS) == {"GB1", "TrpB4"}
    for name in VARIABLE_POSITIONS:
        assert len(VARIABLE_POSITIONS[name]) == len(KNOWN_WILD_TYPES[name][0])


# ---- full-sequence splicing ----


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
@pytest.mark.parametrize(
    "name,rel,parent_len", [("GB1", "GB1/fitness_landscape/GB1.csv", 56),
                            ("TrpB4", "TrpB/fitness_landscape/TrpB4.csv", 390)]
)
def test_variants_are_spliced_into_the_whole_parent_protein(name, rel, parent_len):
    """`sequences()` returns 3-4 character fragments. A protein language model embeds
    those without complaining and returns nothing useful, so the error is silent."""
    ls = Landscape.from_csv(_csv(rel), name)
    full = ls.full_sequences()
    assert len(full) == ls.n
    assert all(len(s) == parent_len for s in full[:50])
    assert len(ls.sequences()[0]) == ls.n_sites  # the fragment, for contrast


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
def test_splicing_the_wild_type_reproduces_the_parent():
    """The round trip that proves the positions are right, not merely consistent."""
    from biodreamer.eval.landscape import PARENT_FASTA, read_fasta

    ls = Landscape.from_csv(_csv("GB1/fitness_landscape/GB1.csv"), "GB1")
    parent = read_fasta(PARENT_FASTA["GB1"])
    i = ls.sequences().index(ls.wild_type)
    assert ls.full_sequences()[i] == parent


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
def test_splicing_changes_exactly_the_variable_positions():
    from biodreamer.eval.landscape import VARIABLE_POSITIONS, PARENT_FASTA, read_fasta

    ls = Landscape.from_csv(_csv("GB1/fitness_landscape/GB1.csv"), "GB1")
    parent = read_fasta(PARENT_FASTA["GB1"])
    at = {p - 1 for p in VARIABLE_POSITIONS["GB1"]}
    for s in ls.full_sequences()[:200]:
        differing = {i for i, (x, y) in enumerate(zip(parent, s)) if x != y}
        assert differing <= at


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
def test_splicing_refuses_when_the_positions_are_unknown():
    """Silently encoding fragments is the failure this guards."""
    ls = Landscape.from_csv(_csv("ParD2/fitness_landscape/ParD2.csv"), "ParD2")
    with pytest.raises(KeyError, match="variable positions unknown"):
        ls.full_sequences()


@pytest.mark.skipif(not SSMULA_DIR.exists(), reason="SSMuLA data not downloaded")
def test_splicing_refuses_a_parent_that_disagrees_with_the_wild_type():
    """Catches a wrong position set or a mismatched parent, rather than producing
    plausible-looking sequences with the wrong residues."""
    ls = Landscape.from_csv(_csv("GB1/fitness_landscape/GB1.csv"), "GB1")
    with pytest.raises(ValueError, match="positions and parent disagree"):
        ls.full_sequences(parent="A" * 56)
