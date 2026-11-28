"""Step 3.1's per-residue model and the amended Step 3.2 gate's metric."""

import numpy as np
import pytest
import torch

from biodreamer.world_model import ResidueWorldModelB, residue_diagnostics

B, L, D_IN, D_MODEL = 6, 24, 32, 16


def make(seed=0, **kw):
    torch.manual_seed(seed)
    return ResidueWorldModelB(d_in=D_IN, d_model=D_MODEL, length=L, hidden=32,
                              d_action=8, max_offset=4, seed=seed, **kw)


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    z_src = torch.randn(B, L, D_IN, generator=g)
    z_dst = torch.randn(B, L, D_IN, generator=g)
    pos = torch.randint(0, L, (B,), generator=g)
    aa = torch.randint(0, 20, (B,), generator=g)
    return z_src, z_dst, pos, aa


def test_encode_keeps_the_residue_axis():
    """The whole point of Step 3.1: the state must stay (L, d), not collapse to (d,)."""
    m = make()
    h = m.encode(torch.randn(B, L, D_IN))
    assert h.shape == (B, L, D_MODEL)


def test_encode_rejects_a_pooled_state():
    """A (B, d) state would broadcast silently through the adapter and produce garbage
    with the right dtype, so it is refused loudly."""
    m = make()
    with pytest.raises(ValueError, match=r"expected \(B, L, d_in\)"):
        m.encode(torch.randn(B, D_IN))


def test_step_is_residual_so_a_zero_update_is_the_identity():
    m = make()
    for p in m.dynamics[-1].parameters():
        torch.nn.init.zeros_(p)
    h = m.encode(torch.randn(B, L, D_IN))
    _, _, pos, aa = batch()
    assert torch.allclose(m.step(h, pos, aa), h, atol=1e-6)


def test_the_action_changes_the_prediction():
    """If the dynamics head ignored the action, one-step prediction would be the
    identity baseline and `one_step_vs_identity` could not fall below 1."""
    m = make()
    h = m.encode(torch.randn(B, L, D_IN))
    _, _, pos, aa = batch()
    other = (aa + 1) % 20
    assert not torch.allclose(m.step(h, pos, aa), m.step(h, pos, other), atol=1e-5)


def test_position_logits_are_one_per_residue():
    """The per-residue state makes position a per-row readout; the pooled model had to
    predict it from a single vector."""
    m = make()
    z_src, z_dst, _, _ = batch()
    lp, la = m.inverse_logits(m.encode(z_src), m.encode(z_dst))
    assert lp.shape == (B, L)
    assert la.shape == (B, 20)


def test_offset_buckets_are_signed_and_share_one_bucket_beyond_the_clamp():
    """Upstream rows must not be confused with downstream ones -- a substitution's
    effect is not symmetric -- while everything past max_offset is one bucket, because
    distant rows were measured to move similarly."""
    m = make()
    idx = m._offset_bucket(L, torch.tensor([10]))[0]
    assert idx[10].item() == m.max_offset            # offset 0 -> centre
    assert idx[9].item() != idx[11].item()           # -1 and +1 differ
    far = 2 * m.max_offset + 1
    assert idx[0].item() == far and idx[L - 1].item() == far
    assert idx[10 - m.max_offset].item() != far      # exactly at the clamp is kept


def test_offset_buckets_stay_inside_the_embedding():
    m = make()
    for p in (0, 1, L - 1):
        idx = m._offset_bucket(L, torch.tensor([p]))
        assert idx.min() >= 0
        assert idx.max() < m.off_emb.num_embeddings


def test_rank_is_measured_on_rows_not_on_a_repooled_state():
    """The amended gate reads `rank_preserved_vs_input` on (B*L, d). Pooling over L
    before the SVD would reintroduce exactly the rank loss the per-residue state exists
    to avoid, and would silently report a much lower input rank."""
    m = make()
    z_src, z_dst, pos, aa = batch()
    d = residue_diagnostics(m, z_src, z_dst, pos, aa)
    rows = z_dst.reshape(-1, D_IN).numpy()
    pooled = z_dst.mean(1).numpy()

    from biodreamer.world_model import effective_rank
    assert d["input_effective_rank"] == pytest.approx(effective_rank(rows), rel=1e-5)
    assert d["input_effective_rank"] > effective_rank(pooled)


def test_preserved_ratio_is_one_when_the_adapter_is_an_isometry():
    """Sanity anchor for the gate's units. An adapter that loses nothing must score 1.

    The input is built INSIDE the projection's range: a projection is an isometry only
    on vectors it does not truncate, and on a random subspace it distorts the relative
    variances enough to move the effective rank a few percent.
    """
    m = make()
    torch.manual_seed(0)
    q = torch.linalg.qr(torch.randn(D_IN, D_IN))[0][:, :D_MODEL]   # (d_in, d_model)
    m.adapter = type("P", (torch.nn.Module,), {
        "forward": lambda self, x: x @ q})()
    coeff = torch.randn(B * L, 4) @ torch.randn(4, D_MODEL)
    z = (coeff @ q.T).reshape(B, L, D_IN)          # lies in q's column span
    _, _, pos, aa = batch()
    d = residue_diagnostics(m, z, z, pos, aa)
    assert d["rank_preserved_vs_input"] == pytest.approx(1.0, abs=1e-3)


def test_collapsed_adapter_is_caught_by_the_preserved_ratio():
    """The failure the gate exists to detect: an adapter mapping everything to one
    direction must score near zero, not near one."""
    m = make()
    m.adapter = type("C", (torch.nn.Module,), {
        "forward": lambda self, x: torch.cat(
            [x[..., :1], torch.zeros_like(x[..., : D_MODEL - 1])], dim=-1)})()
    z_src, z_dst, pos, aa = batch()
    d = residue_diagnostics(m, z_src, z_dst, pos, aa)
    assert d["rank_preserved_vs_input"] < 0.1
