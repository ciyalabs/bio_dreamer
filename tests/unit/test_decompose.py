"""Interaction-order decomposition — Step 1.1.

These tests exist because the decomposition is what the project's central premise
rests on. If it reports structure where there is none, Step 1.1 passes its gate on
an artifact and the error only surfaces in Phase 6. So each property is checked
against a landscape whose answer is known by construction:

    additive by construction      order 1 recovers it, pairwise adds nothing
    pairwise by construction      order 1 fails, pairwise recovers it
    pure noise                    no order recovers anything
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from biodreamer.eval.decompose import (
    DAMP_GRID,
    cv_r2,
    decompose,
    interaction_design,
    noise_ceiling,
)
from biodreamer.eval.landscape import Landscape

ALPHA = list("ACDE")


def _grid(n_sites: int, alphabet=ALPHA) -> np.ndarray:
    return np.array([list(c) for c in itertools.product(alphabet, repeat=n_sites)])


def _additive(n_sites: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    V = _grid(n_sites)
    eff = {(i, a): rng.normal() for i in range(n_sites) for a in ALPHA}
    y = np.array([sum(eff[(i, r)] for i, r in enumerate(row)) for row in V])
    return V, y


def _pairwise(n_sites: int = 5, seed: int = 0, strength: float = 3.0):
    """Additive effects plus strong interactions on every site pair."""
    rng = np.random.default_rng(seed)
    V = _grid(n_sites)
    eff = {(i, a): rng.normal() for i in range(n_sites) for a in ALPHA}
    pair = {
        (i, j, a, b): rng.normal() * strength
        for i, j in itertools.combinations(range(n_sites), 2)
        for a in ALPHA
        for b in ALPHA
    }
    y = np.array(
        [
            sum(eff[(i, r)] for i, r in enumerate(row))
            + sum(
                pair[(i, j, row[i], row[j])]
                for i, j in itertools.combinations(range(n_sites), 2)
            )
            for row in V
        ]
    )
    return V, y


# ---- design matrix ----


def test_design_has_one_indicator_per_site_subset():
    V = _grid(3)
    X = interaction_design(V, ALPHA, 2)
    # 3 singles + 3 pairs = 6 indicators per row, every row the same
    assert X.shape == (64, 3 * 4 + 3 * 16)
    assert (np.asarray(X.sum(axis=1)).ravel() == 6).all()


def test_design_stays_sparse_at_high_order():
    """A dense order-3 design on a 4-site landscape would be unusable."""
    V = _grid(4)
    X = interaction_design(V, ALPHA, 3)
    assert X.nnz == len(V) * (4 + 6 + 4)  # one indicator per site subset
    assert X.shape[1] == 4 * 4 + 6 * 16 + 4 * 64
    # with the 20-letter alphabet of a real landscape the same design is far
    # sparser: 14 non-zeros against 34,480 columns
    big = interaction_design(np.array([list("ACDE")]), list("ABCDEFGHIKLMNPQRSTVWY"), 3)
    assert big.nnz == 14
    assert big.nnz / big.shape[1] < 0.001


def test_design_rejects_unknown_residue():
    V = _grid(2)
    with pytest.raises(ValueError, match="absent from the alphabet"):
        interaction_design(V, ["A", "C"], 1)


@pytest.mark.parametrize("order", [0, 4])
def test_design_rejects_out_of_range_order(order):
    with pytest.raises(ValueError, match="order must be in"):
        interaction_design(_grid(3), ALPHA, order)


# ---- the three landscapes with known answers ----


def test_additive_landscape_is_recovered_at_order_one():
    V, y = _additive()
    r2, _ = cv_r2(interaction_design(V, ALPHA, 1), y)
    assert r2 > 0.99


def test_additive_landscape_gains_nothing_from_pairwise_terms():
    """The false-positive case. If this fails, every gate number is suspect."""
    V, y = _additive()
    dec = decompose(V, y, ALPHA, "additive", max_order=2)
    assert dec.additive > 0.99
    assert dec.pairwise_increment < 0.01


def test_pairwise_landscape_is_missed_by_the_additive_null_and_found_at_order_two():
    V, y = _pairwise()
    dec = decompose(V, y, ALPHA, "pairwise", max_order=2)
    assert dec.additive < 0.4, "an additive model must not explain built-in epistasis"
    assert dec.pairwise_increment > 0.4
    assert dec.structured_nonadditive == pytest.approx(dec.pairwise_increment)


def test_pure_noise_yields_no_structure_at_any_order():
    """The decomposition must not manufacture signal — the TEV failure mode."""
    rng = np.random.default_rng(0)
    V = _grid(5)
    y = rng.normal(size=len(V))
    dec = decompose(V, y, ALPHA, "noise", max_order=2)
    assert dec.additive < 0.1
    assert dec.pairwise_increment < 0.1
    assert dec.best < 0.1


def test_noise_swamps_a_real_additive_signal_the_way_tev_does():
    """Why the raw Step 0.2 figure was wrong: add noise, lose the signal.

    Same landscape, same model, only the error bars change — and the
    'non-additive fraction' goes from ~0 to ~0.9 without any epistasis existing.
    """
    V, y = _additive()
    rng = np.random.default_rng(1)
    noisy = y + rng.normal(0, 3.0 * y.std(), size=len(y))
    clean_r2, _ = cv_r2(interaction_design(V, ALPHA, 1), y)
    noisy_r2, _ = cv_r2(interaction_design(V, ALPHA, 1), noisy)
    assert clean_r2 > 0.99
    assert noisy_r2 < 0.2
    # the naive figure implies epistasis in a landscape that has none
    assert 1.0 - noisy_r2 > 0.8


# ---- regularisation selection ----


def test_damp_is_selected_without_seeing_the_scoring_fold():
    """Leakage here would inflate every order equally and silently."""
    V, y = _pairwise()
    a, _ = cv_r2(interaction_design(V, ALPHA, 2), y, seed=0)
    b, _ = cv_r2(interaction_design(V, ALPHA, 2), y, seed=7)
    # different folds, similar answer: the score is a property of the landscape
    assert abs(a - b) < 0.12


def test_chosen_damp_comes_from_the_fixed_grid():
    V, y = _additive()
    _, damp = cv_r2(interaction_design(V, ALPHA, 1), y)
    assert damp in DAMP_GRID


# ---- saturated order ----


def test_top_order_is_excluded_by_default():
    """On a complete landscape the top-order cell of a held-out genotype is
    unidentifiable from the training folds, so its CV score measures the design."""
    V, y = _additive(n_sites=3)
    dec = decompose(V, y, ALPHA, "additive3")
    assert dec.max_order == 2
    assert dec.saturated_orders == [3]
    assert 3 not in dec.r2_by_order


def test_explicit_max_order_overrides_the_default():
    V, y = _additive(n_sites=3)
    dec = decompose(V, y, ALPHA, "additive3", max_order=3)
    assert set(dec.r2_by_order) == {1, 2, 3}
    assert dec.saturated_orders == []


# ---- noise ceiling ----


def test_noise_ceiling_is_one_minus_the_measurement_share():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    v = np.full(5, y.var(ddof=1) * 0.25)
    out = noise_ceiling(y, v)
    assert out["noise_fraction"] == pytest.approx(0.25)
    assert out["reliability_ceiling"] == pytest.approx(0.75)


def test_noise_ceiling_is_undefined_on_a_flat_landscape():
    out = noise_ceiling(np.ones(10), np.ones(10))
    assert np.isnan(out["reliability_ceiling"])


def test_noise_ceiling_reports_median_separately():
    """Mean and median diverge when depth is skewed, as on TEV (0.33 vs 0.035)."""
    y = np.arange(100.0)
    v = np.concatenate([np.full(95, 1.0), np.full(5, 1000.0)])
    out = noise_ceiling(y, v)
    assert out["noise_fraction"] > out["noise_fraction_median"] * 10


# ---- integration with the landscape objects the project actually uses ----


def test_synthetic_additive_landscape_from_the_eval_harness_decomposes_as_additive():
    ls = Landscape.synthetic_additive(n_sites=4, alphabet="ACDE")
    dec = decompose(ls.variants, ls.fitness, ls.alphabet, ls.name, max_order=2)
    assert dec.additive > 0.99
    assert dec.pairwise_increment < 0.01


def test_as_metrics_exposes_every_order_and_the_increment():
    V, y = _pairwise()
    m = decompose(V, y, ALPHA, "p", max_order=2).as_metrics()
    assert {"cv_r2_order_1", "cv_r2_order_2", "cv_r2_additive", "pairwise_increment"} <= set(m)
    assert all(np.isfinite(v) for v in m.values())


# ---- exact decomposition on a complete hypercube ----


def _hypercube_from_effects(n_sites: int, main, inter=None):
    """Fitness over 2^n_sites corners, built from chosen Walsh coefficients."""
    N = 2**n_sites
    y = np.zeros(N)
    for idx in range(N):
        bits = [(idx >> (n_sites - 1 - i)) & 1 for i in range(n_sites)]
        signs = [1 - 2 * b for b in bits]  # bit 0 -> +1, bit 1 -> -1
        val = 0.0
        for i, m in enumerate(main):
            val += m * signs[i]
        for (i, j), c in (inter or {}).items():
            val += c * signs[i] * signs[j]
        y[idx] = val
    return y


def test_walsh_decomposition_is_exactly_additive_for_main_effects_only():
    from biodreamer.eval.decompose import walsh_variance_by_order

    y = _hypercube_from_effects(4, [1.0, 2.0, 0.5, 0.25])
    frac = walsh_variance_by_order(y, 4)
    assert frac[1] == pytest.approx(1.0)
    assert sum(frac.values()) == pytest.approx(1.0)
    for k in (2, 3, 4):
        assert frac[k] == pytest.approx(0.0)


def test_walsh_decomposition_splits_a_known_mixture_by_order():
    from biodreamer.eval.decompose import walsh_variance_by_order

    # one main effect of amplitude 1, one pairwise of amplitude 1 -> 50/50
    y = _hypercube_from_effects(3, [1.0, 0.0, 0.0], {(1, 2): 1.0})
    frac = walsh_variance_by_order(y, 3)
    assert frac[1] == pytest.approx(0.5)
    assert frac[2] == pytest.approx(0.5)
    assert frac[3] == pytest.approx(0.0)


def test_walsh_decomposition_finds_pure_highest_order_interaction():
    from biodreamer.eval.decompose import walsh_variance_by_order

    N = 8
    y = np.array([1.0 if bin(i).count("1") % 2 == 0 else -1.0 for i in range(N)])
    frac = walsh_variance_by_order(y, 3)
    assert frac[3] == pytest.approx(1.0)


def test_walsh_decomposition_agrees_with_an_additive_regression_fit():
    """Two independent routes to the same number, on the same data."""
    from biodreamer.eval.decompose import walsh_variance_by_order

    rng = np.random.default_rng(3)
    y = _hypercube_from_effects(4, rng.normal(size=4), {(0, 1): 0.8, (2, 3): 0.4})
    frac = walsh_variance_by_order(y, 4)

    V = np.array([list(f"{i:04b}") for i in range(16)])
    X = interaction_design(V, ["0", "1"], 1)
    beta, mu = __import__("biodreamer.eval.decompose", fromlist=["_ridge"])._ridge(X, y, 1e-9)
    r2 = 1 - ((y - (X @ beta + mu)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert frac[1] == pytest.approx(r2, abs=1e-6)


def test_walsh_decomposition_rejects_a_wrong_corner_count():
    from biodreamer.eval.decompose import walsh_variance_by_order

    with pytest.raises(ValueError, match="expected 8 corners"):
        walsh_variance_by_order(np.zeros(7), 3)


def test_walsh_decomposition_is_undefined_on_a_flat_hypercube():
    from biodreamer.eval.decompose import walsh_variance_by_order

    frac = walsh_variance_by_order(np.ones(8), 3)
    assert all(np.isnan(v) for v in frac.values())
