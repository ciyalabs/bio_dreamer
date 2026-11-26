"""HypercubeME output parsing — Step 1.1 path analysis past four sites.

A wrong parse here does not crash; it silently attaches one genotype's fitness to
a different corner and produces path statistics for landscapes that do not exist.
So the format is pinned down with real lines from the extraction, and the two
independent descriptions of each hypercube — the diagonal and the two corner
genotypes — are required to agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biodreamer.eval.hypercube import (
    WILD_TYPE,
    Hypercube,
    iter_lines,
    load_genotype_table,
    parse_genotype,
    parse_hypercube,
    sample,
    to_landscape,
)

# verbatim from data/processed/hypercubeme/out_140_160/
DIM2 = ("A143I:C141G", "140F:141C:142F:143A:144D", "140F:141G:142F:143I:144D")
# a dimension-6 line, where four of the six positions revert to wild type
DIM6 = (
    "A143I:D144N:F140Z:F142Z:S141Z:S157Z",
    "140F:141S:142F:143A:144D:157S",
    "143I:144N",
)


# ---- genotype specs ----


def test_empty_mutation_list_is_the_wild_type():
    assert parse_genotype("") == frozenset()
    assert parse_genotype("   ") == frozenset()


def test_genotype_spec_parses_to_position_residue_pairs():
    assert parse_genotype("140F:141G") == frozenset({(140, "F"), (141, "G")})


def test_genotype_spec_is_order_independent():
    assert parse_genotype("141G:140F") == parse_genotype("140F:141G")


@pytest.mark.parametrize("bad", ["A140", "140", "F140F", "140f", "140F:oops"])
def test_malformed_genotype_token_is_rejected(bad):
    with pytest.raises(ValueError, match="position"):
        parse_genotype(bad)


# ---- the genotype table ----


def _table_file(tmp_path: Path) -> Path:
    p = tmp_path / "genotypes.txt"
    p.write_text(
        "mut_list\tfitness\terror\n"
        "\t0.0\t0.0\n"
        "140F:141G\t0.5\t0.0\n"
        "140F\t0.25\t0.0\n"
        "bogus\t0.9\t0.0\n"  # unparsable: skipped, not fatal
    )
    return p


def test_genotype_table_includes_the_wild_type_and_skips_bad_rows(tmp_path):
    t = load_genotype_table(_table_file(tmp_path))
    assert t[frozenset()] == 0.0
    assert t[frozenset({(140, "F"), (141, "G")})] == 0.5
    assert len(t) == 3


def test_genotype_table_requires_a_fitness_header(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("a\tb\n1\t2\n")
    with pytest.raises(ValueError, match="fitness"):
        load_genotype_table(p)


# ---- hypercube lines ----


def test_dimension_two_line_parses_to_two_positions_and_a_shared_background():
    hc = parse_hypercube(*DIM2)
    assert hc.dimension == 2
    assert hc.positions == (141, 143)
    assert hc.state_a == ("C", "A")
    assert hc.state_b == ("G", "I")
    assert hc.background == frozenset({(140, "F"), (142, "F"), (144, "D")})


def test_z_in_the_diagonal_means_the_position_reverts_to_wild_type():
    hc = parse_hypercube(*DIM6)
    assert hc.dimension == 6
    assert hc.positions == (140, 141, 142, 143, 144, 157)
    assert hc.state_a == ("F", "S", "F", "A", "D", "S")
    assert hc.state_b == (WILD_TYPE, WILD_TYPE, WILD_TYPE, "I", "N", WILD_TYPE)
    assert hc.background == frozenset()


def test_a_diagonal_that_contradicts_the_corners_is_rejected():
    """The check that makes a silent misparse impossible."""
    with pytest.raises(ValueError, match="diagonal and corners disagree"):
        parse_hypercube("A143I:C141G", "140F:141C:142F:143A", "140F:141G:142F:143W")


def test_corners_differing_outside_the_diagonal_are_rejected():
    with pytest.raises(ValueError, match="differ outside the diagonal"):
        parse_hypercube("A143I:C141G", "141C:143A:150Q", "141G:143I:150R")


def test_a_position_with_the_same_residue_in_both_corners_is_not_a_dimension():
    with pytest.raises(ValueError, match="not a dimension"):
        parse_hypercube("A143A", "143A", "143A")


@pytest.mark.parametrize("bad", ["143I", "A143", "a143i"])
def test_malformed_diagonal_token_is_rejected(bad):
    with pytest.raises(ValueError, match="position"):
        parse_hypercube(bad, "143A", "143I")


# ---- corners ----


def test_a_dimension_d_hypercube_has_exactly_two_to_the_d_distinct_corners():
    hc = parse_hypercube(*DIM6)
    corners = list(hc.corners())
    assert len(corners) == 2**6
    assert len({enc for enc, _ in corners}) == 2**6
    assert len({key for _, key in corners}) == 2**6


def test_corners_carry_the_background_and_drop_wild_type_positions():
    hc = parse_hypercube(*DIM2)
    keys = {key for _, key in hc.corners()}
    assert frozenset({(140, "F"), (142, "F"), (144, "D"), (141, "C"), (143, "A")}) in keys
    # every corner contains the whole background
    assert all(hc.background <= k for k in keys)


def test_a_fully_reverted_corner_has_no_mutations_at_the_variable_sites():
    hc = parse_hypercube("F140Z:S141Z", "140F:141S", "")
    keys = {key for _, key in hc.corners()}
    assert frozenset() in keys


# ---- landscapes ----


def _complete_table(hc: Hypercube, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {key: float(rng.normal()) for _, key in hc.corners()}


def test_landscape_has_one_point_per_corner():
    hc = parse_hypercube(*DIM2)
    ls = to_landscape(hc, _complete_table(hc))
    assert ls is not None
    assert ls.n == 4
    assert ls.n_sites == 2


def test_landscape_wild_type_is_the_corner_closest_to_the_his3_wild_type():
    """A walk starts from the least-mutated corner, not an arbitrary one."""
    hc = parse_hypercube(*DIM6)
    ls = to_landscape(hc, _complete_table(hc))
    assert ls is not None
    # four of the six positions revert; at 143 (A/I) and 144 (D/N) neither state
    # is wild type, so those tie and the rule takes first_genotype's residue
    assert ls.wild_type == "---AD-"
    assert ls.wild_type in ls.sequences()
    assert ls.wild_type_fitness() in ls.fitness.tolist()


def test_a_hypercube_with_an_unmeasured_corner_yields_none():
    """Should not happen for output from the same genotype file; counted if it does."""
    hc = parse_hypercube(*DIM2)
    table = _complete_table(hc)
    table.pop(next(iter(table)))
    assert to_landscape(hc, table) is None


def test_landscape_fitness_matches_the_table_corner_by_corner():
    hc = parse_hypercube(*DIM2)
    table = _complete_table(hc)
    ls = to_landscape(hc, table)
    assert ls is not None
    by_enc = {enc: table[key] for enc, key in hc.corners()}
    for seq, fit in zip(ls.sequences(), ls.fitness):
        assert fit == pytest.approx(by_enc[seq])


# ---- streaming and sampling ----


def _hc_file(tmp_path: Path, n: int = 50) -> Path:
    p = tmp_path / "hypercubes_2.txt"
    lines = ["diagonal first_genotype last_genotype"]
    for i in range(n):
        lines.append(f"A143I:C141G 140F:141C:142F:143A:{200+i}D 140F:141G:142F:143I:{200+i}D")
    p.write_text("\n".join(lines) + "\n")
    return p


def test_iter_lines_requires_the_expected_header(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("nope\na b c\n")
    with pytest.raises(ValueError, match="diagonal"):
        list(iter_lines(p))


def test_iter_lines_yields_one_triple_per_hypercube(tmp_path):
    assert len(list(iter_lines(_hc_file(tmp_path, 7)))) == 7


def test_iter_lines_handles_an_empty_last_corner(tmp_path):
    p = tmp_path / "hypercubes_1.txt"
    p.write_text("diagonal first_genotype last_genotype\nF140Z 140F\n")
    assert list(iter_lines(p)) == [("F140Z", "140F", "")]


def test_sample_counts_what_it_saw_and_what_it_could_use(tmp_path):
    hcf = _hc_file(tmp_path, 50)
    # build a table covering every corner of every line
    table: dict = {}
    for d, f, l in iter_lines(hcf):
        table |= _complete_table(parse_hypercube(d, f, l))
    out, stats = sample(hcf, table, 10, seed=1)
    assert stats == {"seen": 50, "sampled": 10, "unparsed": 0, "incomplete": 0, "usable": 10}
    assert len(out) == 10


def test_sample_returns_everything_when_the_file_is_smaller_than_the_request(tmp_path):
    hcf = _hc_file(tmp_path, 3)
    table: dict = {}
    for d, f, l in iter_lines(hcf):
        table |= _complete_table(parse_hypercube(d, f, l))
    out, stats = sample(hcf, table, 25, seed=0)
    assert stats["seen"] == 3
    assert len(out) == 3


def test_sample_reports_incomplete_hypercubes_instead_of_dropping_them_silently(tmp_path):
    hcf = _hc_file(tmp_path, 5)
    out, stats = sample(hcf, {}, 5, seed=0)
    assert out == []
    assert stats["incomplete"] == 5
    assert stats["usable"] == 0


def test_wild_type_tie_break_prefers_the_first_genotype_corner():
    """Both corners mutated at every site: the choice must be stated, not incidental."""
    hc = parse_hypercube("A10I:D11N", "10A:11D", "10I:11N")
    ls = to_landscape(hc, _complete_table(hc))
    assert ls is not None
    assert ls.wild_type == "AD"


def test_the_wild_type_marker_is_not_a_mutation():
    """HypercubeME spells the wild type `0Z` in its output and empty in its input.

    Read as a substitution at position 0 it makes the corner miss the genotype
    table, which dropped every hypercube anchored at the true wild type.
    """
    assert parse_genotype("0Z") == frozenset()
    assert parse_genotype("140F:0Z") == frozenset({(140, "F")})


def test_a_hypercube_whose_corner_is_the_wild_type_parses():
    hc = parse_hypercube("E154Z:I145Z", "145I:154E", "0Z")
    assert hc.background == frozenset()
    assert hc.state_b == (WILD_TYPE, WILD_TYPE)
    keys = {k for _, k in hc.corners()}
    assert frozenset() in keys, "the wild-type corner must be reachable"
