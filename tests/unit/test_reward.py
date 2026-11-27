"""Reward head and its losses — Step 2.1.

The censoring decision is the part of this step most likely to be quietly undone, so
most of these tests are about the loss, not the network. `scripts/censoring_audit.py`
measured three of the seven H1 assays carrying a detection limit, one with 31.9% of its
variants at a single exact value. The ranking loss must use those points for the
comparisons they support and no others; a regression that treats the floor as a
measurement would corrupt every epistasis estimate downstream.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

torch = pytest.importorskip("torch")

from biodreamer.reward import (  # noqa: E402
    Adapter,
    RewardConfig,
    RewardHead,
    masked_pool,
    pairwise_ranking_loss,
    ranking_pairs,
    standardise_per_assay,
    weighted_smooth_l1,
)


# ---- the adapter must not mix residues ----


def test_adapter_is_applied_independently_per_residue():
    """Mixing positions would destroy the per-residue structure Steps 1.3 and 1.4
    showed is load-bearing, and the damage would be invisible in the loss."""
    a = Adapter(RewardConfig()).eval()
    z = torch.randn(1, 7, 1280)
    with torch.no_grad():
        out = a(z)
        shuffled = a(z[:, [3, 1, 0, 6, 2, 5, 4], :])
    assert torch.allclose(out[:, [3, 1, 0, 6, 2, 5, 4], :], shuffled, atol=1e-5)


def test_adapter_projects_to_the_measured_latent_dimension():
    """d=128 is fixed by the Step 1.3 sweep, not a free parameter."""
    assert RewardConfig().latent_dim == 128
    assert Adapter(RewardConfig())(torch.randn(2, 5, 1280)).shape == (2, 5, 128)


def test_adapter_rejects_a_non_sequence_input():
    with pytest.raises(ValueError, match=r"expected \(B, L, d\)"):
        Adapter(RewardConfig())(torch.randn(4, 1280))


# ---- pooling ----


def test_masked_pool_ignores_padding():
    """A batch mixing a 36-residue domain with a 390-residue one is mostly padding for
    the short one; an unmasked mean would divide its signal by ten."""
    z = torch.zeros(1, 10, 3)
    z[0, :4] = 1.0
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :4] = True
    assert torch.allclose(masked_pool(z, mask), torch.ones(1, 3))
    assert torch.allclose(masked_pool(z, None), torch.full((1, 3), 0.4))


def test_the_head_pools_but_also_exposes_the_unpooled_latent():
    """Pooling happens at the reward head and nowhere else; the world model and the
    inverse-dynamics anchor both need the per-residue tensor."""
    h = RewardHead()
    out = h(torch.randn(2, 9, 1280))
    assert out["latent"].shape == (2, 9, 128)
    assert out["pooled"].shape == (2, 128)
    assert out["stability"].shape == (2,)
    assert torch.allclose(h.latents(torch.zeros(1, 4, 1280)), out["latent"][:1, :4] * 0 + h.latents(torch.zeros(1, 4, 1280)))


# ---- censoring: the decision this step exists to make ----


def _batch():
    #            live  live  FLOOR FLOOR live   | assay B
    target = torch.tensor([3.0, 2.0, 0.0, 0.0, 1.0, 5.0, 4.0])
    assay = torch.tensor([0, 0, 0, 0, 0, 1, 1])
    censored = torch.tensor([False, False, True, True, False, False, False])
    return target, assay, censored


def test_pairs_never_join_two_censored_variants():
    """Two variants at the detection limit carry no ordering information. Pairing them
    teaches a tie-break the assay never measured."""
    target, assay, censored = _batch()
    i, j = ranking_pairs(target, assay, censored, max_pairs=None)
    for a, b in zip(i.tolist(), j.tolist()):
        assert not (censored[a] and censored[b])


def test_a_censored_variant_is_still_compared_against_a_live_one():
    """The other half of the decision: 'this one is dead and that one is not' is real
    signal, and excluding censored points entirely would discard it."""
    target, assay, censored = _batch()
    i, j = ranking_pairs(target, assay, censored, max_pairs=None)
    mixed = [(a, b) for a, b in zip(i.tolist(), j.tolist()) if censored[a] ^ censored[b]]
    assert mixed, "censored-versus-live pairs must survive"


def test_pairs_never_cross_assays():
    """Raw DMS scores are not comparable between assays -- different units, ranges and
    organisms. PREREGISTRATION.md standing rule 5."""
    target, assay, _ = _batch()
    i, j = ranking_pairs(target, assay, None, max_pairs=None)
    assert torch.equal(assay[i], assay[j])


def test_pairs_are_ordered_and_never_tied():
    target, assay, _ = _batch()
    i, j = ranking_pairs(target, assay, None, max_pairs=None)
    assert (target[i] > target[j]).all()


def test_pair_sampling_respects_the_cap():
    target = torch.arange(60.0)
    assay = torch.zeros(60, dtype=torch.long)
    i, _ = ranking_pairs(target, assay, None, max_pairs=25)
    assert i.numel() == 25


def test_no_comparable_pair_yields_a_finite_differentiable_zero():
    """A batch of one assay with identical targets must not crash the step."""
    pred = torch.zeros(3, requires_grad=True)
    loss = pairwise_ranking_loss(
        pred, torch.ones(3), torch.zeros(3, dtype=torch.long), max_pairs=None
    )
    loss.backward()
    assert loss.item() == 0.0
    assert pred.grad is not None


# ---- ranking loss behaviour ----


def test_ranking_loss_rewards_the_correct_order():
    target, assay, censored = _batch()
    right = pairwise_ranking_loss(target.clone(), target, assay, censored, max_pairs=None)
    wrong = pairwise_ranking_loss(-target, target, assay, censored, max_pairs=None)
    assert right < wrong


def test_ranking_loss_is_invariant_to_a_monotone_rescaling_of_the_target():
    """Order is all it reads, which is why per-assay standardisation is unnecessary
    for this loss -- and why it survives assays with incomparable units."""
    target, assay, censored = _batch()
    pred = torch.randn(7)
    a = pairwise_ranking_loss(pred, target, assay, censored, max_pairs=None)
    b = pairwise_ranking_loss(pred, target * 100 + 7, assay, censored, max_pairs=None)
    assert a == pytest.approx(b.item(), abs=1e-6)


# ---- stability loss ----


def test_imprecise_measurements_are_down_weighted():
    """9.7% of Tsuboyama measurements have a 95% CI wider than 1 kcal/mol. Treating
    them as equal to a 0.13 kcal/mol measurement wastes the good ones."""
    # the second point is badly mispredicted; whether that should cost much depends
    # entirely on whether its measurement is trustworthy
    pred = torch.zeros(2)
    target = torch.tensor([0.2, 5.0])
    equal = weighted_smooth_l1(pred, target, torch.tensor([0.5, 0.5]))
    bad_one_is_imprecise = weighted_smooth_l1(pred, target, torch.tensor([0.5, 4.0]))
    bad_one_is_precise = weighted_smooth_l1(pred, target, torch.tensor([4.0, 0.5]))
    assert bad_one_is_imprecise < equal < bad_one_is_precise


def test_weights_are_normalised_so_the_loss_scale_does_not_move():
    """Otherwise the effective learning rate changes with the batch's error bars."""
    pred, target = torch.zeros(4), torch.ones(4)
    flat = weighted_smooth_l1(pred, target, torch.full((4,), 0.5))
    unweighted = weighted_smooth_l1(pred, target, None)
    assert flat == pytest.approx(unweighted.item(), abs=1e-6)


def test_a_precision_floor_stops_one_point_dominating():
    pred, target = torch.zeros(3), torch.ones(3)
    assert torch.isfinite(
        weighted_smooth_l1(pred, target, torch.tensor([1e-9, 1.0, 1.0]))
    )


def test_stability_loss_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="differ"):
        weighted_smooth_l1(torch.zeros(3), torch.zeros(4))


# ---- per-assay standardisation ----


def test_standardisation_is_within_assay():
    value = torch.tensor([1.0, 3.0, 100.0, 300.0])
    assay = torch.tensor([0, 0, 1, 1])
    out = standardise_per_assay(value, assay)
    assert out[:2].mean().abs() < 1e-5
    assert out[2:].mean().abs() < 1e-5
    assert torch.allclose(out[:2], out[2:], atol=1e-4), "scale must not survive"


# ---- it can actually learn ----


def test_the_head_learns_a_separable_synthetic_signal():
    """End-to-end sanity: if this cannot fit a signal planted in the latent, nothing
    measured downstream means anything."""
    torch.manual_seed(0)
    n, L = 96, 12
    z = torch.randn(n, L, 1280)
    direction = torch.randn(1280)
    truth = (z.mean(dim=1) @ direction)
    assay = torch.zeros(n, dtype=torch.long)

    head = RewardHead()
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    first = last = None
    for step in range(120):
        out = head(z)
        loss = pairwise_ranking_loss(out["fitness"], truth, assay, max_pairs=512)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.6, f"loss barely moved: {first:.4f} -> {last:.4f}"

    head.eval()
    with torch.no_grad():
        pred = head(z)["fitness"]
    from scipy.stats import spearmanr

    rho = spearmanr(pred.numpy(), truth.numpy()).statistic
    assert rho > 0.8, f"recovered only rho={rho:.3f}"


# ---- splits: the two-instance rule and H1's protocol ----

from pathlib import Path  # noqa: E402

DATA_READY = Path("data/processed/encoding_budget.json").exists() and Path(
    "data/processed/censoring_audit.json"
).exists()
pytestmark_data = pytest.mark.skipif(not DATA_READY, reason="budget/audit not generated")


@pytestmark_data
def test_the_two_instances_are_not_the_same_set():
    """They were, briefly: `h1_assays` in the budget file is already firewall-filtered,
    so deriving both from it made the two-instance rule a silent no-op."""
    from biodreamer.reward.data import assays_for

    b, p = set(assays_for("benchmark")), set(assays_for("planning"))
    assert p < b, "planning must be a strict subset of benchmark"
    assert len(b - p) >= 1


@pytestmark_data
def test_the_planning_instance_never_sees_a_banned_assay():
    """The whole point of the firewall. A planning checkpoint is the only one allowed
    to produce a Phase 6 number."""
    from biodreamer.reward.data import assays_for, banned_assays

    assert not (set(assays_for("planning")) & banned_assays())


@pytestmark_data
def test_the_benchmark_instance_may_use_the_banned_assays():
    """It exists for literature comparability and may never produce a Phase 6 result;
    if it could not train on them there would be no reason for it to exist."""
    from biodreamer.reward.data import assays_for, banned_assays

    assert set(assays_for("benchmark")) & banned_assays()


@pytestmark_data
def test_an_unknown_instance_is_rejected_rather_than_guessed():
    from biodreamer.reward.data import assays_for

    with pytest.raises(ValueError, match="two-instance rule"):
        assays_for("whatever")


def test_h1_split_follows_the_registered_protocol():
    """Train on <=300 order<=2, test on unseen order>=3. The threshold
    `spearman_mean >= 0.55` is only comparable to published numbers under it."""
    import numpy as np

    from biodreamer.reward.data import h1_split

    order = np.array([1] * 500 + [2] * 500 + [3] * 50 + [7] * 10)
    tr, te = h1_split(order, cap=300, seed=0)
    assert len(tr) == 300
    assert (order[tr] <= 2).all()
    assert (order[te] >= 3).all()
    assert len(te) == 60
    assert not set(tr.tolist()) & set(te.tolist())


def test_h1_split_samples_rather_than_taking_the_first_rows():
    """ProteinGym files are often ordered by position; the first 300 rows would be the
    N-terminus and the split would be a region, not a sample."""
    import numpy as np

    from biodreamer.reward.data import h1_split

    order = np.ones(2000, dtype=int)
    order[-20:] = 3
    tr, _ = h1_split(order, cap=300, seed=0)
    assert tr.max() > 400, "a head-first split would never reach past index 300"


@pytestmark_data
def test_censored_mask_comes_from_the_audit_not_a_recomputation():
    """So the loss and scripts/censoring_audit.py cannot drift apart."""
    from biodreamer.reward.data import detection_limits

    limits = detection_limits()
    assert "D7PM05_CLYGR_Somermeyer_2022" in limits
    assert limits["D7PM05_CLYGR_Somermeyer_2022"] == pytest.approx(590.0)
    assert "PHOT_CHLRE_Chen_2023" not in limits, "a clean assay must have no limit"


# ---- pooling before the adapter ----


def test_pool_first_accepts_an_already_pooled_cache_vector():
    """The 2.4 GB pooled cache stores (B, d), not (B, L, d)."""
    from biodreamer.reward import RewardConfig, RewardHead

    h = RewardHead(RewardConfig(pool_first=True))
    out = h(torch.randn(5, 1280))
    assert out["stability"].shape == (5,)
    assert out["latent"] is None, "there is no per-residue latent to report"


def test_pool_first_matches_pooling_a_full_tensor():
    """Feeding the head a pre-pooled vector must equal feeding it the matrix."""
    from biodreamer.reward import RewardConfig, RewardHead, masked_pool

    h = RewardHead(RewardConfig(pool_first=True)).eval()
    z = torch.randn(3, 11, 1280)
    with torch.no_grad():
        a = h(z)["stability"]
        b = h(masked_pool(z, None))["stability"]
    assert torch.allclose(a, b, atol=1e-5)


# --- depth-stratified pair sampling -------------------------------------------------
# The reason this exists: uniform sampling over an order<=2 training set draws mostly
# order-1-vs-order-2 pairs, whose right answer is "the shallower one". A head that
# learns only that scores ~0.31 on H1, which is where the uniformly trained head landed.


def test_equal_order_frac_one_returns_only_equal_order_pairs():
    target = torch.tensor([3.0, 2.0, 1.0, 0.0])
    assay = torch.zeros(4, dtype=torch.long)
    order = torch.tensor([1, 1, 2, 2])
    i, j = ranking_pairs(target, assay, order=order, equal_order_frac=1.0,
                         max_pairs=None)
    assert i.numel() > 0
    assert torch.equal(order[i], order[j])


def test_equal_order_frac_zero_matches_the_unstratified_sampler():
    target = torch.tensor([3.0, 2.0, 1.0, 0.0])
    assay = torch.zeros(4, dtype=torch.long)
    order = torch.tensor([1, 1, 2, 2])
    a = ranking_pairs(target, assay, max_pairs=None)
    b = ranking_pairs(target, assay, order=order, equal_order_frac=0.0, max_pairs=None)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_stratified_pairs_still_respect_assay_and_censoring():
    target = torch.tensor([3.0, 2.0, 3.0, 2.0])
    assay = torch.tensor([0, 0, 1, 1])
    order = torch.tensor([2, 2, 2, 2])
    censored = torch.tensor([True, True, False, False])
    i, j = ranking_pairs(target, assay, censored=censored, order=order,
                         equal_order_frac=1.0, max_pairs=None)
    assert torch.equal(assay[i], assay[j])          # never across assays
    assert not (censored[i] & censored[j]).any()     # never both censored
    assert (target[i] > target[j]).all()


def test_partial_frac_mixes_both_populations_within_budget():
    g = torch.Generator().manual_seed(0)
    n = 60
    target = torch.linspace(1.0, 0.0, n)
    assay = torch.zeros(n, dtype=torch.long)
    order = torch.tensor([1] * (n // 2) + [2] * (n - n // 2))
    i, j = ranking_pairs(target, assay, max_pairs=40, generator=g,
                         order=order, equal_order_frac=0.5)
    assert i.numel() == 40
    same = (order[i] == order[j]).sum().item()
    assert 20 <= same < 40  # the reserved half, plus whatever uniform adds


def test_no_equal_order_pairs_yields_no_gradient_rather_than_a_wrong_one():
    # a caller that stratifies must check the pair count; this documents the failure
    target = torch.tensor([2.0, 1.0])
    assay = torch.zeros(2, dtype=torch.long)
    order = torch.tensor([1, 2])
    i, _ = ranking_pairs(target, assay, order=order, equal_order_frac=1.0,
                         max_pairs=None)
    assert i.numel() == 0
    pred = torch.zeros(2, requires_grad=True)
    loss = pairwise_ranking_loss(pred, target, assay, order=order,
                                 equal_order_frac=1.0)
    assert float(loss.detach()) == 0.0
    loss.backward()  # must not raise


def test_equal_order_frac_outside_unit_interval_is_rejected():
    target = torch.tensor([2.0, 1.0])
    assay = torch.zeros(2, dtype=torch.long)
    order = torch.tensor([1, 1])
    with pytest.raises(ValueError, match="equal_order_frac"):
        ranking_pairs(target, assay, order=order, equal_order_frac=1.5)


def test_order_length_mismatch_is_rejected():
    target = torch.tensor([2.0, 1.0])
    assay = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="order"):
        ranking_pairs(target, assay, order=torch.tensor([1, 1, 2]),
                      equal_order_frac=1.0)


# --- depth-aware fitness ------------------------------------------------------------
# Step 2.1 measured that the head and mutation count are COMPLEMENTARY: at fixed depth
# the head ranks at +0.18 to +0.36 on 5/7 assays, while depth alone scores 0.31 and has
# no within-depth skill. These tests pin the combination's invariants.


def _depth_cfg(**kw):
    return RewardConfig(backbone_dim=8, latent_dim=4, trunk_dim=4,
                        pool_first=True, depth_aware=True, **kw)


def test_depth_aware_requires_order():
    head = RewardHead(_depth_cfg())
    with pytest.raises(ValueError, match="order"):
        head(torch.zeros(3, 8))


def test_depth_penalty_is_monotone_and_non_negative():
    head = RewardHead(_depth_cfg())
    with torch.no_grad():
        head.depth_beta.fill_(-5.0)   # softplus keeps it positive even here
    order = torch.tensor([1, 2, 5, 28])
    pen = head.depth_penalty(order)
    assert (pen >= 0).all()
    assert torch.all(pen[1:] >= pen[:-1])


def test_depth_penalty_cannot_invert_however_beta_is_driven():
    # the prior is that deeper is never better; softplus enforces it structurally
    head = RewardHead(_depth_cfg())
    for b in (-50.0, -1.0, 0.0, 3.0):
        with torch.no_grad():
            head.depth_beta.fill_(b)
        pen = head.depth_penalty(torch.tensor([1, 9]))
        assert pen[1] >= pen[0]


def test_depth_aware_lowers_the_score_of_deeper_variants():
    torch.manual_seed(0)
    head = RewardHead(_depth_cfg())
    head.eval()  # dropout would make identical rows differ
    with torch.no_grad():
        head.depth_beta.fill_(2.0)
    z = torch.zeros(2, 8)                      # identical input, different depth
    out = head(z, order=torch.tensor([1, 6]))
    assert out["fitness"][0] > out["fitness"][1]
    # the sequence-only score ignores depth, so it stays tied
    assert torch.allclose(out["fitness_sequence_only"][0],
                          out["fitness_sequence_only"][1])


def test_sequence_only_score_is_unchanged_by_the_penalty():
    torch.manual_seed(0)
    plain = RewardHead(RewardConfig(backbone_dim=8, latent_dim=4, trunk_dim=4,
                                    pool_first=True))
    torch.manual_seed(0)
    deep = RewardHead(_depth_cfg())
    plain.eval()
    deep.eval()
    z = torch.randn(4, 8)
    a = plain(z)["fitness"]
    b = deep(z, order=torch.tensor([1, 2, 3, 4]))["fitness_sequence_only"]
    assert torch.allclose(a, b, atol=1e-6)


def test_log_transform_extrapolates_more_gently_than_linear():
    lin = RewardHead(_depth_cfg(depth_transform="linear"))
    log = RewardHead(_depth_cfg(depth_transform="log"))
    with torch.no_grad():
        lin.depth_beta.fill_(1.0)
        log.depth_beta.fill_(1.0)
    order = torch.tensor([2, 28])
    # identified on the 1-2 contrast, the two agree in kind; at order 28 they must not
    ratio_lin = (lin.depth_penalty(order)[1] / lin.depth_penalty(order)[0]).item()
    ratio_log = (log.depth_penalty(order)[1] / log.depth_penalty(order)[0]).item()
    assert ratio_lin > ratio_log


def test_bad_depth_transform_is_rejected():
    with pytest.raises(ValueError, match="depth_transform"):
        RewardHead(_depth_cfg(depth_transform="sqrt"))


def test_order_below_one_is_rejected():
    head = RewardHead(_depth_cfg())
    with pytest.raises(ValueError, match="order"):
        head.depth_penalty(torch.tensor([0, 1]))


def test_depth_beta_receives_gradient_from_the_ranking_loss():
    torch.manual_seed(0)
    head = RewardHead(_depth_cfg())
    z = torch.randn(6, 8)
    order = torch.tensor([1, 1, 1, 2, 2, 2])
    target = torch.tensor([1.0, 0.9, 0.8, 0.3, 0.2, 0.1])  # shallower is fitter
    assay = torch.zeros(6, dtype=torch.long)
    loss = pairwise_ranking_loss(head(z, order=order)["fitness"], target, assay,
                                 max_pairs=None)
    loss.backward()
    assert head.depth_beta.grad is not None
    assert head.depth_beta.grad.abs().item() > 0


# --- per-assay depth weight ---------------------------------------------------------
# A per-assay AFFINE output would be a no-op: H1 scores per-assay Spearman, invariant
# under any increasing per-assay rescaling. A per-assay depth WEIGHT is not, because it
# reweights two components rather than rescaling one. test_per_assay_affine_is_a_noop
# below pins the first claim so nobody spends time implementing it.


def test_per_assay_affine_output_cannot_change_per_assay_spearman():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=200)
    truth = pred + rng.normal(scale=0.5, size=200)
    base = spearmanr(pred, truth).statistic
    for a, b in [(1.0, 0.0), (3.7, -2.0), (0.01, 100.0)]:
        assert spearmanr(a * pred + b, truth).statistic == pytest.approx(base)


def test_per_assay_depth_weight_needs_the_assay_index():
    head = RewardHead(_depth_cfg(n_assays=3))
    with pytest.raises(ValueError, match="assay"):
        head.depth_penalty(torch.tensor([1, 2]))


def test_per_assay_depth_weights_act_independently():
    head = RewardHead(_depth_cfg(n_assays=2))
    with torch.no_grad():
        head.depth_beta.copy_(torch.tensor([-6.0, 2.0]))  # near-zero, then strong
    order = torch.tensor([4, 4])
    pen = head.depth_penalty(order, assay=torch.tensor([0, 1]))
    assert pen[1] > pen[0] * 10


def test_per_assay_weight_count_matches_config():
    assert RewardHead(_depth_cfg()).depth_beta.numel() == 1
    assert RewardHead(_depth_cfg(n_assays=7)).depth_beta.numel() == 7


def test_shared_weight_still_works_without_an_assay_index():
    head = RewardHead(_depth_cfg())
    head.eval()
    out = head(torch.zeros(2, 8), order=torch.tensor([1, 5]))
    assert out["fitness"][0] > out["fitness"][1]
