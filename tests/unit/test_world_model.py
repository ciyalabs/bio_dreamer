"""World Model B: the parts that would fail silently if wrong."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from biodreamer.world_model import SIGReg, WorldModelB, dynamics_weight, effective_rank
from biodreamer.world_model.transitions import _parse


def test_parse_action_is_one_based():
    assert _parse("A24G") == (23, 5)
    assert _parse("M1C") == (0, 1)


def test_effective_rank_detects_collapse():
    """The number the Step 3.2 gate is written on has to separate these two cases."""
    rng = np.random.default_rng(0)
    full = rng.normal(size=(512, 32))
    assert effective_rank(full) > 25  # isotropic: close to the full 32

    collapsed = np.zeros((512, 32))
    collapsed[:, 0] = rng.normal(size=512)  # all variance in one direction
    assert effective_rank(collapsed) == pytest.approx(1.0, abs=1e-6)

    constant = np.ones((512, 32))
    assert effective_rank(constant) == 0.0


def test_effective_rank_ignores_scale_but_not_shape():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(256, 16))
    assert effective_rank(x) == pytest.approx(effective_rank(x * 1000.0), rel=1e-6)
    y = x.copy()
    y[:, 8:] *= 1e-6  # half the directions nearly dead
    assert effective_rank(y) < 0.6 * effective_rank(x)


def test_sigreg_prefers_gaussian_to_collapsed():
    """SIGReg is the ONLY anti-collapse term in this model, so its sign must be right."""
    torch.manual_seed(0)
    sig = SIGReg(n_proj=32).double()
    gauss = torch.randn(1024, 24, dtype=torch.float64)
    spike = torch.randn(1024, 24, dtype=torch.float64)
    spike[:, 1:] *= 1e-4
    assert float(sig(gauss)) < float(sig(spike))


def test_dynamics_weight_schedule():
    # silent during warm-up, 1.0 after the ramp, monotone in between
    assert dynamics_weight(0, 3, 5) == 0.0
    assert dynamics_weight(2, 3, 5) == 0.0
    assert dynamics_weight(8, 3, 5) == 1.0
    mid = [dynamics_weight(e, 3, 5) for e in range(3, 9)]
    assert all(0.0 < m <= 1.0 for m in mid)
    assert mid == sorted(mid)
    assert dynamics_weight(0, 0, 0) == 1.0  # no warm-up requested


def test_step_is_residual_so_a_null_action_is_near_identity():
    """`step` adds a predicted edit. With zeroed dynamics output it must be exactly h."""
    torch.manual_seed(0)
    m = WorldModelB(d_in=32, d_model=16, length=10, hidden=24).double()
    for p in m.dynamics[-1].parameters():
        torch.nn.init.zeros_(p)
    h = torch.randn(4, 16, dtype=torch.float64)
    pos = torch.tensor([0, 1, 2, 3])
    aa = torch.tensor([0, 1, 2, 3])
    assert torch.allclose(m.step(h, pos, aa), h, atol=1e-12)


def test_action_changes_the_prediction():
    """A dynamics head that ignores its action would pass every loss-based check."""
    torch.manual_seed(0)
    m = WorldModelB(d_in=32, d_model=16, length=10, hidden=24).double()
    h = torch.randn(8, 16, dtype=torch.float64)
    p0 = m.step(h, torch.zeros(8, dtype=torch.long), torch.zeros(8, dtype=torch.long))
    p1 = m.step(h, torch.full((8,), 7), torch.full((8,), 11))
    assert not torch.allclose(p0, p1, atol=1e-6)


def test_inverse_head_shapes():
    m = WorldModelB(d_in=32, d_model=16, length=10, hidden=24).double()
    h = torch.randn(5, 16, dtype=torch.float64)
    lp, la = m.inverse_logits(h, h + 0.1)
    assert lp.shape == (5, 10) and la.shape == (5, 20)


def test_gradients_reach_the_adapter_only_through_the_prediction():
    """The target is stop-gradient: the adapter must not be trainable through it.

    Without this the model can drive the dynamics loss down by making the TARGET
    degenerate, which is collapse arriving disguised as progress.
    """
    torch.manual_seed(0)
    m = WorldModelB(d_in=8, d_model=8, length=4, hidden=16).double()
    z = torch.randn(6, 8, dtype=torch.float64)
    h = m.encode(z)
    pred = m.step(h[:3], torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    loss = (pred - h[3:].detach()).pow(2).sum()
    loss.backward()
    assert m.adapter[0].weight.grad is not None
    assert torch.isfinite(m.adapter[0].weight.grad).all()


def test_mmap_rows_reorders_without_materialising_the_whole_table(tmp_path):
    """`full[want]` on a memmap is fancy indexing and materialises every requested row
    at once -- 52 GB on CAPSD. `_MmapRows` keeps it lazy, and must still return rows in
    the order asked for, including with repeats and out-of-order reads.
    """
    import numpy as np

    from biodreamer.world_model.transitions import _MmapRows

    path = tmp_path / "t.npy"
    table = np.arange(12 * 3 * 2, dtype=np.float16).reshape(12, 3, 2)
    np.save(path, table)
    full = np.load(path, mmap_mode="r")

    want = np.array([7, 1, 7, 0, 11])
    view = _MmapRows(full, want)
    assert view.shape == (5, 3, 2)
    assert len(view) == 5
    np.testing.assert_array_equal(view[np.arange(5)], table[want])
    # a scalar index returns one state, not a length-1 batch
    np.testing.assert_array_equal(view[2], table[7])
    # a subset, out of order, still maps through `want`
    np.testing.assert_array_equal(view[np.array([4, 0])], table[[11, 7]])


def test_transition_index_needs_no_cache_and_finds_more_than_the_pooled_one():
    """The coverage gap: the pooled cache was built for the H1 split, which excludes the
    low-order PARENTS of transitions. The index computed from mutant strings alone must
    find strictly more, or encoding against it buys nothing.
    """
    from biodreamer.world_model import transition_index

    assay = "D7PM05_CLYGR_Somermeyer_2022"
    _, src_all, _, _, _, cov_all = transition_index(assay, present=None)
    assert cov_all == 1.0
    assert len(src_all) > 20000

    from biodreamer.world_model.transitions import BACKBONE, CACHE

    f = CACHE / f"{BACKBONE}__{assay}.npz"
    if f.exists():
        import numpy as np
        with np.load(f) as z:
            present = set(z["keys"])
        _, src_cached, _, _, _, cov_cached = transition_index(assay, present=present)
        assert cov_cached < 1.0
        assert len(src_cached) < len(src_all)


def test_build_rejects_an_unknown_state():
    import pytest

    from biodreamer.world_model import build

    with pytest.raises(ValueError, match="state must be"):
        build("D7PM05_CLYGR_Somermeyer_2022", state="per_residue")
