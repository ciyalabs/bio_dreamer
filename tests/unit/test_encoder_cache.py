"""Encoder harness and cache tests — Step 0.4.

The declared experiment `per_residue_rank_test` lives here: the whole revision
turns on latents staying per-residue, and a silent pool anywhere would look like
a working pipeline while destroying the signal the world model is meant to learn.
These tests make that failure loud.

Backbone tests are marked `gpu` and skipped when weights or CUDA are absent, so
the suite stays runnable on any machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from biodreamer.protein_dreamer.cache import (
    Cache,
    estimate_bytes,
    format_bytes,
    sequence_hash,
)
from biodreamer.protein_dreamer.encoder import (
    UNKNOWN_3DI,
    BACKBONES,
    ProteinEncoder,
    interleave,
)


@pytest.fixture()
def cache(tmp_path):
    return Cache(root=tmp_path / "c")


# ---- the per-residue invariant ----


def test_cache_refuses_a_pooled_embedding(cache):
    """A (d,) vector is a pooled embedding; storing one must fail loudly."""
    pooled = np.zeros(1280, dtype=np.float16)
    with pytest.raises(ValueError, match="per-residue"):
        cache.put("some/backbone", "MKGY", pooled)


def test_cache_refuses_a_batched_three_d_array(cache):
    with pytest.raises(ValueError, match="per-residue"):
        cache.put("some/backbone", "MKGY", np.zeros((1, 4, 1280), dtype=np.float16))


def test_cache_accepts_per_residue(cache):
    arr = np.random.rand(10, 1280).astype(np.float16)
    p = cache.put("b", "MKGYFGPYGG", arr)
    assert p.exists()
    back = cache.get("b", "MKGYFGPYGG")
    assert back.shape == (10, 1280)


def test_error_message_names_the_reason(cache):
    """The message must say why, not just that — this is the failure that matters."""
    with pytest.raises(ValueError) as e:
        cache.put("b", "MK", np.zeros(8))
    assert "reward head" in str(e.value)


# ---- cache behaviour ----


def test_round_trip_preserves_values(cache):
    arr = np.random.rand(7, 64).astype(np.float32)
    cache.put("b", "SEQ", arr)
    assert np.allclose(cache.get("b", "SEQ"), arr.astype(np.float16), atol=1e-2)


def test_miss_then_hit_accounting(cache):
    assert cache.get("b", "NEW") is None
    assert cache.stats.misses == 1 and cache.stats.hits == 0
    cache.put("b", "NEW", np.zeros((3, 8)))
    assert cache.get("b", "NEW") is not None
    assert cache.stats.hits == 1


def test_backbones_do_not_collide(cache):
    a = np.ones((4, 8)); b = np.zeros((4, 8))
    cache.put("saprot", "SEQ", a)
    cache.put("esm2", "SEQ", b)
    assert cache.get("saprot", "SEQ").sum() == 32
    assert cache.get("esm2", "SEQ").sum() == 0


def test_sequence_hash_is_normalised():
    assert sequence_hash("mkgy") == sequence_hash(" MKGY ")
    assert sequence_hash("MKGY") != sequence_hash("MKGA")


def test_contains(cache):
    cache.put("b", "SEQ", np.zeros((2, 4)))
    assert ("b", "SEQ") in cache
    assert ("b", "OTHER") not in cache


def test_eviction_respects_budget(tmp_path):
    c = Cache(root=tmp_path / "c", max_bytes=20_000)
    for i in range(20):
        c.put("b", f"SEQ{i}", np.zeros((10, 100), dtype=np.float16))  # 2 KB each
    assert c.size_bytes() <= 20_000
    assert c.stats.evictions > 0


def test_no_partial_entry_after_a_failed_write(cache, monkeypatch):
    """A crash mid-write must not leave a readable but truncated entry."""
    import numpy as _np

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(_np, "save", boom)
    with pytest.raises(OSError):
        cache.put("b", "SEQ", np.zeros((4, 8)))
    assert cache.get("b", "SEQ") is None


# ---- storage budget ----


def test_estimate_matches_the_documented_gb1_figures():
    """The numbers quoted in cache.py and DATA_REGISTER.md must stay true."""
    raw = estimate_bytes(536_000, 56, 1280, "float16")
    assert 70 < raw / 2**30 < 75          # ~72 GiB at the raw backbone dim
    projected = estimate_bytes(536_000, 56, 128, "float16")
    assert 6.5 < projected / 2**30 < 7.5  # ~7.2 GiB after the adapter


def test_budget_report_flags_an_over_budget_workload(tmp_path):
    c = Cache(root=tmp_path / "c", max_bytes=1_000_000)
    report = c.budget_report({"gb1_doubles": (536_000, 56, 1280)})
    assert "OVER BUDGET" in report


def test_format_bytes():
    assert format_bytes(1536) == "1.5 KiB"
    assert format_bytes(2 * 2**30) == "2.0 GiB"


# ---- SaProt interleaving ----


def test_interleave_builds_saprot_input():
    assert interleave("MKG", "ddp") == "MdKdGp"


def test_interleave_without_structure_uses_the_unknown_token():
    out = interleave("MKG")
    assert out == f"M{UNKNOWN_3DI}K{UNKNOWN_3DI}G{UNKNOWN_3DI}"


def test_interleave_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="lengths differ"):
        interleave("MKGY", "dd")


def test_prosst_is_not_an_available_backbone():
    """Step 0.2 found ProSST publishes no licence; it must not be selectable."""
    assert "prosst" not in BACKBONES
    with pytest.raises(ValueError, match="no licence"):
        ProteinEncoder("prosst")


def test_unknown_backbone_lists_the_valid_ones():
    with pytest.raises(ValueError, match="saprot"):
        ProteinEncoder("not_a_model")


# ---- end to end, needs weights ----


@pytest.fixture(scope="module")
def saprot(tmp_path_factory):
    """One SaProt instance for the whole module.

    A 650M backbone is ~1.25 GiB on the GPU and this machine has 3.6 GiB, so a
    per-test instance exhausts VRAM by the second test — which is how the first
    version of these tests produced a spurious skip.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    enc = ProteinEncoder("saprot", cache=Cache(root=tmp_path_factory.mktemp("c")))
    try:
        enc.encode(["MKGY"])  # force the load here so failures surface once
    except Exception as e:
        pytest.skip(f"SaProt unavailable: {type(e).__name__}: {e}")
    return enc


@pytest.mark.gpu
def test_saprot_returns_per_residue_rows(saprot):
    out = saprot.encode(["MKGYFGPYGG", "MKGYFGPYGA"])
    assert [a.shape for a in out] == [(10, 1280), (10, 1280)]
    assert all(a.ndim == 2 for a in out), "backbone must not pool"


@pytest.mark.gpu
def test_a_single_mutation_moves_the_latent(saprot):
    """If one substitution does not move the per-residue latent, D2 is dead."""
    wt, mut = saprot.encode(["MKGYFGPYGG", "MKGYFGPYGA"])
    assert np.abs(wt - mut).mean() > 1e-4


@pytest.mark.gpu
def test_structure_changes_the_embedding(saprot):
    """3Di tokens must actually reach the model, or SaProt is just ESM-2."""
    plain = saprot.encode("MKGYFGPYGG", structure=None)
    withs = saprot.encode("MKGYFGPYGG", structure="ddpvvvvqql")
    assert np.abs(plain - withs).mean() > 1e-3
