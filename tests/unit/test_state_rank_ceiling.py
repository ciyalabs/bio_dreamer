"""The rank-ceiling diagnostic underwrites the Step 3.2 gate amendment, so its one
non-obvious property -- that effective rank actually measures spread of variance, not
matrix rank -- is pinned here.
"""

import numpy as np
import pytest

from scripts.state_rank_ceiling import effective_rank


def test_isotropic_data_has_effective_rank_near_the_dimension():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 32))
    assert effective_rank(x) == pytest.approx(32, rel=0.1)


def test_one_direction_has_effective_rank_near_one():
    rng = np.random.default_rng(0)
    d = rng.normal(size=32)
    x = rng.normal(size=(4000, 1)) * d
    assert effective_rank(x) < 1.05


def test_full_matrix_rank_can_still_be_a_low_effective_rank():
    """The whole reason Step 3.2 uses effective rank: this matrix has hard rank 32 and
    numpy's matrix_rank says so, while nearly all its variance is in two directions."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 32))
    x[:, 2:] *= 1e-3
    assert np.linalg.matrix_rank(x) == 32
    assert effective_rank(x) < 4


def test_constant_data_has_zero_effective_rank():
    assert effective_rank(np.ones((100, 8))) == 0.0


def test_effective_rank_is_invariant_to_a_global_scale():
    """It is a ratio of eigenvalues, so doubling the data must not move it -- the reason
    `rank_preserved_vs_input` can compare a 256-d latent with a 1280-d input at all."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 16)) @ rng.normal(size=(16, 16))
    assert effective_rank(x) == pytest.approx(effective_rank(x * 7.5), rel=1e-9)
