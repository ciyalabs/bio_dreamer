"""Zero-shot baseline — Step 2.4.

The whole measurement rests on two things that fail silently. An off-by-one in the token
index reads the log-probabilities of a *neighbouring* residue and still returns a
plausible-looking Spearman; and the scoring rule must stay additive over substitutions,
because that additivity is what makes 1,201 forward passes cover 943,211 variants and is
also the limitation the step reports.

The model itself is not loaded here — these are the invariants that hold without it.
`scripts/zeroshot_baseline.py` validates the token alignment against the live tokenizer at
run time via `encode_wt`, which is the part that needs the real vocabulary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from zeroshot_baseline import AA, cache_path, rho, score_variants  # noqa: E402


def _logprobs(wt: str, table: dict[int, dict[str, float]]) -> dict[int, np.ndarray]:
    """Hand-built log-probability rows, so the expected score is arithmetic."""
    out = {}
    for pos, vals in table.items():
        row = np.full(len(AA), -10.0)
        for aa, v in vals.items():
            row[AA.index(aa)] = v
        out[pos] = row
    return out


def test_score_is_the_sum_of_per_substitution_log_odds():
    wt = "MKVAY"
    lp = _logprobs(wt, {2: {"K": -1.0, "R": -0.5}, 4: {"A": -2.0, "G": -0.25}})
    # K2R contributes -0.5 - (-1.0) = +0.5; A4G contributes -0.25 - (-2.0) = +1.75
    got = score_variants(np.array(["K2R", "A4G", "K2R:A4G"]), lp, wt)
    assert got == pytest.approx([0.5, 1.75, 2.25])


def test_additivity_is_exact_and_is_the_documented_limitation():
    """A triple must equal the sum of its singles -- no epistasis is representable."""
    wt = "MKVAYG"
    lp = _logprobs(
        wt, {2: {"K": -1.0, "R": -0.5}, 4: {"A": -2.0, "G": -0.25}, 6: {"G": -3.0, "W": -1.0}}
    )
    singles = score_variants(np.array(["K2R", "A4G", "G6W"]), lp, wt)
    triple = score_variants(np.array(["K2R:A4G:G6W"]), lp, wt)
    assert triple[0] == pytest.approx(singles.sum())


def test_a_wrong_reference_residue_raises_rather_than_scoring():
    """Catches an off-by-one or a mismatched wild type instead of returning a number."""
    wt = "MKVAY"
    lp = _logprobs(wt, {2: {"K": -1.0, "R": -0.5}})
    with pytest.raises(ValueError, match="wild type has"):
        score_variants(np.array(["V2R"]), lp, wt)  # position 2 is K, not V


def test_rho_is_zero_rather_than_nan_on_a_constant_prediction():
    """A degenerate assay must not poison the mean with a NaN."""
    assert rho(np.zeros(10), np.arange(10.0)) == 0.0


def test_cache_path_separates_backbone_and_rule():
    """Two rules must not overwrite each other's saved scores."""
    a = cache_path("saprot", "masked-marginals")
    b = cache_path("saprot", "wt-marginals")
    c = cache_path("esm2", "masked-marginals")
    assert a != b and a != c
    assert "-" not in a.name.removesuffix(".npz")
