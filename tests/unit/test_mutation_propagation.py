"""`mutation_propagation` decides whether the per-residue cache can be windowed, a
3x storage question, so its position arithmetic is pinned.
"""

import numpy as np
import pytest

from scripts.mutation_propagation import mutated_sites


def test_sites_are_zero_based():
    """The cache stores 1-based ProteinGym positions and the latent rows are 0-based.
    An off-by-one here would silently compare the wrong residue and make a window look
    worse than it is."""
    assert mutated_sites("A1G").tolist() == [0]
    assert mutated_sites("M1A:K118R").tolist() == [0, 117]


def test_multiple_substitutions_are_all_returned():
    assert mutated_sites("A5G:C10D:E20F").tolist() == [4, 9, 19]


def test_a_mutant_string_with_no_substitution_raises():
    """A silently-empty site list would make `dist` empty and the window trivially
    cover nothing, reporting 0% captured rather than failing."""
    with pytest.raises(ValueError, match="no substitutions"):
        mutated_sites("")


def test_window_union_is_what_storage_costs_not_one_window():
    """Two substitutions 4 apart share rows at radius 8, so the union is smaller than
    two independent windows. The storage column has to count the union."""
    L, r = 100, 8
    sites = np.array([20, 24])
    d = np.abs(np.arange(L)[:, None] - sites[None, :]).min(1)
    union = int((d <= r).sum())
    assert union == 21           # rows 12..32 inclusive
    assert union < 2 * (2 * r + 1)
