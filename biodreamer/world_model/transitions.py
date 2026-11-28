"""Single-mutation transitions (z_t, action, z_{t+1}) from a DMS assay.

A world model needs (state, action, next state). A DMS assay gives measured variants, so
a transition exists wherever two measured variants differ by exactly one substitution and
one's mutation set is a subset of the other's: `A24G` -> `A24G:L56P` is the action
"substitute P at position 56". The `planning` instance alone yields 1,395,965 of these.

**On the state representation.** `state="pooled"` uses the mean-pooled 1280-d encoder
output, which is exact and is what the reward model consumes. `state="residue"` uses the
full per-residue latent Z in R^(L x d) that Step 3.1 registers, read from
`data/processed/residue_cache/` (built by `scripts/encode_residue_states.py`).

**Reconstructing the per-residue state instead of encoding it does not work, and both
cheap versions of that idea are refused in code rather than offered behind a flag.**

1. *Wild-type rows with the mutated rows substituted in* — i.e. what
   `positional_cache/` already holds. Measured on `CAPSD_AAV2S_Sinai_2021`:

       cosine(reconstructed pooled, true pooled)   0.9991
       ||true - wt||                               0.141
       ||reconstructed - wt||                      0.023
       fraction of the displacement captured       0.052
       direction agreement of that displacement    0.30

   The cosine looks excellent because both vectors are dominated by the wild type.
   Against the displacement that actually distinguishes one variant from another, the
   reconstruction keeps **5%**, pointing somewhere else.

2. *A window of radius r around each mutation.* Tested because it looked like a 3x
   storage saving, and refuted: on CAPSD a radius-16 window recovers 37.6% of the
   pooled displacement (cosine 0.697), because a substitution's effect on the pooled
   vector is a diffuse chain-wide shift rather than a local one. It reaches 93.6% on the
   118-residue PHOT only because a radius-16 window there already covers 90% of the
   chain. See `results/state_representation.md`.

So the per-residue cache stores the full tensor. A model trained on either shortcut would
be fitting a shadow of the transition while every cosine-based diagnostic reported
success.

**On coverage.** The transition STRUCTURE depends only on the assay's mutant strings, so
`transition_index` computes it with no cache at all. That matters: the pooled cache was
built for the H1 and literature splits and excludes most low-order variants, which are
precisely the PARENTS of transitions, so reading transitions out of it loses most of them
(`D7PM05` 55 usable of 20,513; `Q6WV12` 92 of 34,111). Encoding exactly the variants the
index names closes that gap by construction rather than by a larger encode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biodreamer.protein_dreamer.cache import sequence_hash
from biodreamer.reward.data import detection_limits, load_assay

CACHE = Path("data/processed/pooled_cache")
RESIDUE_CACHE = Path("data/processed/residue_cache")
BACKBONE = "westlake-repl_SaProt_650M_AF2"
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}


@dataclass
class Transitions:
    """One assay's transitions, as index arrays into `z`."""

    z: np.ndarray  # pooled: (n_variants, d). residue: (n_variants, L, d), may be a memmap
    src: np.ndarray  # (n_trans,) index of z_t
    dst: np.ndarray  # (n_trans,) index of z_{t+1}
    position: np.ndarray  # (n_trans,) 0-based residue index of the action
    aa: np.ndarray  # (n_trans,) amino-acid index of the action
    score: np.ndarray  # (n_variants,) measured fitness, for the reward term
    assay: str
    length: int
    coverage: float  # share of the assay's VARIANTS that had an exact cached state
    state: str = "pooled"  # "pooled" or "residue"
    # Share of the assay's FORMABLE transitions that survived the cache filter. This is
    # the quantity the Step 3.2 precondition means by "the coverage gap", and it is not
    # the same as `coverage`: a cache scoped to transition endpoints stores few of the
    # assay's variants (GCN4: 119 of 2,638, so coverage 5%) while losing no transitions
    # at all (144 of 144, so transition_coverage 1.0). Reading the variant-level number
    # as if it were the transition-level one would make a complete cache look starved.
    transition_coverage: float = float("nan")

    def __len__(self) -> int:
        return len(self.src)

    @property
    def per_residue(self) -> bool:
        return self.z.ndim == 3


def _parse(mut: str) -> tuple[int, int]:
    """'A24G' -> (23, index of G). Positions are 1-based in the assay files."""
    return int(mut[1:-1]) - 1, AA_INDEX[mut[-1]]


def transition_index(assay: str, present: set[str] | None = None,
                     max_transitions: int | None = None, seed: int = 0):
    """The transition structure of an assay, from its mutant strings alone.

    Needs no encoder cache, which is the point: `present` filters to variants whose state
    is actually available, and passing `None` asks what the assay could yield if every
    needed variant were encoded. `scripts/encode_residue_states.py` calls it with `None`
    to decide what to encode, so the resulting cache has 100% coverage by construction.

    Returns `(used, src, dst, pos, aa, coverage)` with `src`/`dst` indexing `used`.
    """
    a = load_assay(assay, detection_limits())
    index = {}
    for i, (m, s) in enumerate(zip(a.mutant, a.sequence)):
        if present is not None and sequence_hash(s) not in present:
            continue
        index[frozenset(str(m).split(":"))] = i
    coverage = len(index) / max(len(a.mutant), 1)

    src, dst, pos, aa = [], [], [], []
    for muts, j in index.items():
        if not muts:
            continue
        for added in muts:
            parent = index.get(muts - {added})
            if parent is None:
                continue
            p_, c = _parse(added)
            src.append(parent)
            dst.append(j)
            pos.append(p_)
            aa.append(c)

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    pos = np.asarray(pos, dtype=np.int64)
    aa = np.asarray(aa, dtype=np.int64)
    if max_transitions is not None and len(src) > max_transitions:
        keep = np.random.default_rng(seed).choice(
            len(src), size=max_transitions, replace=False
        )
        src, dst, pos, aa = src[keep], dst[keep], pos[keep], aa[keep]

    # only the variants a transition touches need a state; everything downstream, the
    # encode included, is scoped to these
    used = np.unique(np.concatenate([src, dst])) if len(src) else np.zeros(0, np.int64)
    remap = np.full(len(a.mutant), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return used, remap[src], remap[dst], pos, aa, coverage


def build(assay: str, max_transitions: int | None = None, seed: int = 0,
          state: str = "pooled", mmap: bool = True) -> Transitions:
    """Transitions with their endpoint states loaded.

    `state="residue"` returns `z` of shape `(n_used, L, d)`, memory-mapped by default --
    the three largest `planning` assays are 26-52 GB each and will not fit in RAM.
    """
    if state not in ("pooled", "residue"):
        raise ValueError(f"state must be 'pooled' or 'residue', got {state!r}")
    a = load_assay(assay, detection_limits())

    if state == "pooled":
        z = np.load(CACHE / f"{BACKBONE}__{assay}.npz")
        row = {k: i for i, k in enumerate(z["keys"])}
        emb = z["emb"]
        used, src, dst, pos, aa, coverage = transition_index(
            assay, set(row), max_transitions, seed
        )
        states = np.stack(
            [emb[row[sequence_hash(s)]] for s in a.sequence[used]]
        ).astype(np.float32)
    else:
        keys_path = RESIDUE_CACHE / f"{BACKBONE}__{assay}.keys.npy"
        data_path = RESIDUE_CACHE / f"{BACKBONE}__{assay}.npy"
        if not keys_path.exists() or not data_path.exists():
            raise FileNotFoundError(
                f"{data_path} missing -- run\n"
                f"  python -I scripts/encode_residue_states.py --assay {assay}"
            )
        keys = np.load(keys_path)
        row = {k: i for i, k in enumerate(keys)}
        used, src, dst, pos, aa, coverage = transition_index(
            assay, set(row), max_transitions, seed
        )
        full = np.load(data_path, mmap_mode="r" if mmap else None)
        # the cache rows are already in `transition_index` order for this assay, but the
        # cap or a partial encode can reorder them, so go through the hash every time
        # rather than assuming the two orderings agree
        want = np.array([row[sequence_hash(s)] for s in a.sequence[used]])
        states = full[want] if not mmap else _MmapRows(full, want)

    # cheap verification rather than an assumption: how many transitions this assay
    # could form if every variant were cached. Seconds even on PHOT's 1.2M.
    _, src_all, _, _, _, _ = transition_index(assay, present=None)
    n_formable = len(src_all) if max_transitions is None else min(
        len(src_all), max_transitions
    )

    return Transitions(
        z=states,
        src=src,
        dst=dst,
        position=pos,
        aa=aa,
        score=a.score[used].astype(np.float32),
        assay=assay,
        length=len(str(a.sequence[0])),
        coverage=coverage,
        state=state,
        transition_coverage=len(src) / max(n_formable, 1),
    )


class _MmapRows:
    """A reordered view of a memmap that reads only the rows asked for.

    `full[want]` on a memmap is fancy indexing, which materialises every row at once --
    52 GB on CAPSD. This keeps the mapping lazy so a batch of 64 states costs 64 reads.
    """

    def __init__(self, full: np.ndarray, want: np.ndarray) -> None:
        self._full = full
        self._want = want
        self.ndim = full.ndim
        self.shape = (len(want),) + full.shape[1:]
        self.dtype = full.dtype

    def __len__(self) -> int:
        return len(self._want)

    def __getitem__(self, key):
        idx = self._want[key]
        if np.isscalar(idx):
            return self._full[idx]
        # sort before reading so the mmap is touched in file order, then restore
        order = np.argsort(idx)
        out = np.empty((len(idx),) + self._full.shape[1:], self._full.dtype)
        out[order] = self._full[idx[order]]
        return out


def split(n: int, seed: int = 0, frac_val: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Transition-level train/validation split."""
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1.0 - frac_val))
    return perm[:cut], perm[cut:]
