"""The structure kernel is a rewrite of a double sum; these check it is the same number."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from structure_kernel import (  # noqa: E402
    assay_vocabulary,
    SiteData,
    StructureKernel,
    build_indicator,
    ca_coords,
    pair_distances,
    parse_mutations,
)


@pytest.fixture
def site() -> SiteData:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(30, 20))
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    return SiteData(logp=logp, coords=rng.normal(size=(30, 3)) * 10.0)


def test_parse_mutations_is_one_based():
    assert parse_mutations("A24G") == [(23, 5)]  # G is index 5 in ACDEFGHIKLMNPQRSTVWY
    assert parse_mutations("A1C:W3Y") == [(0, 1), (2, 19)]


def test_hellinger_is_zero_for_identical_sites_and_bounded(site):
    vocab = [(3, 0), (3, 7), (11, 2)]
    d_hell, _, d_euc = pair_distances(vocab, vocab, site)
    # same site compared with itself: the distributions are identical, distance 0
    assert d_hell[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert d_euc[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(d_hell >= -1e-12) and np.all(d_hell <= 1.0 + 1e-9)
    assert np.allclose(d_hell, d_hell.T)


def test_hellinger_matches_the_textbook_formula(site):
    vocab = [(1, 0), (9, 3), (20, 11)]
    d_hell, _, _ = pair_distances(vocab, vocab, site)
    p = np.exp(site.logp)
    for i, (si, _) in enumerate(vocab):
        for j, (sj, _) in enumerate(vocab):
            want = np.sqrt(0.5 * ((np.sqrt(p[si]) - np.sqrt(p[sj])) ** 2).sum())
            assert d_hell[i, j] == pytest.approx(want, abs=1e-10)


def test_bilinear_form_equals_the_naive_double_sum(site):
    """S @ Phi @ S.T must reproduce Kermut eq. 2 summed variant by variant.

    This is the whole reason the module is written the way it is, so it is the test that
    matters: if the rewrite is wrong, every downstream number is wrong and nothing else
    here would catch it.
    """
    mutants = np.array(["A2C:D10E", "A2C", "W21Y:D10E:A2C", "F5G"])
    S, vocab = build_indicator(mutants)
    keys = [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    dists = [torch.tensor(d) for d in pair_distances(keys, keys, site)]

    kern = StructureKernel().double()
    with torch.no_grad():
        kern.log_g1.fill_(0.3), kern.log_g2.fill_(-0.4), kern.log_g3.fill_(0.1)
        phi = kern.phi(*dists).numpy()
        fast = kern(torch.tensor(S.toarray()), torch.tensor(S.toarray()), dists).numpy()

    per_variant = [parse_mutations(m) for m in mutants]
    naive = np.zeros((len(mutants), len(mutants)))
    for a, ma in enumerate(per_variant):
        for b, mb in enumerate(per_variant):
            naive[a, b] = sum(phi[vocab[i], vocab[j]] for i in ma for j in mb)
    assert np.allclose(fast, naive, atol=1e-12)


def test_build_indicator_restricts_only_when_told_not_to_grow():
    train = np.array(["A2C", "D10E"])
    _, vocab = build_indicator(train)
    S_te, vocab2 = build_indicator(np.array(["A2C:F5G", "F5G"]), vocab, grow=False)
    assert vocab2 is vocab and S_te.shape[1] == 2
    assert S_te[0].sum() == 1.0 and S_te[1].sum() == 0.0


def test_cross_covariance_keeps_test_only_mutations(site):
    """A mutation seen only at test time still pairs with every training mutation.

    Phi[a, b] is a similarity between two MUTATIONS and is non-zero for any pair, so
    restricting the test indicator to the training vocabulary does not drop an unused
    term -- it computes a different kernel. On this protocol (train order <= 2, test
    order >= 3) test-only mutations are the normal case, so getting this wrong would
    quietly corrupt every structure-kernel number rather than failing loudly.
    """
    train = np.array(["A2C", "D10E"])
    test = np.array(["A2C:F5G", "W21Y"])  # W21Y shares no mutation with training
    S_tr, S_te, keys, train_cols = assay_vocabulary(train, test)
    assert S_tr.shape[1] == S_te.shape[1] == len(keys) == 4
    assert S_te[1].sum() == 1.0  # W21Y is present, not dropped

    dists = [torch.tensor(d) for d in pair_distances(keys, keys, site)]
    kern = StructureKernel().double()
    with torch.no_grad():
        kern.log_g1.fill_(0.2), kern.log_g2.fill_(0.1), kern.log_g3.fill_(-0.3)
        phi = kern.phi(*dists).numpy()
        cross = kern(
            torch.tensor(S_te.toarray()), torch.tensor(S_tr.toarray()), dists
        ).numpy()

    index = {k: i for i, k in enumerate(keys)}
    for a, mt in enumerate(test):
        for b, mr in enumerate(train):
            want = sum(
                phi[index[i], index[j]]
                for i in parse_mutations(mt)
                for j in parse_mutations(mr)
            )
            assert cross[a, b] == pytest.approx(want, abs=1e-12)
    # and the W21Y row is genuinely non-zero, which the dropping version would have zeroed
    assert abs(cross[1]).sum() > 0.0


def test_ca_coords_reads_one_row_per_residue(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  MET A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      3  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C\n"
        "TER\n"
    )
    c = ca_coords(pdb)
    assert c.shape == (2, 3)
    assert np.allclose(c[1], [4.0, 5.0, 6.0])
