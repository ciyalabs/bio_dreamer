"""Single-mutation transitions (z_t, action, z_{t+1}) from a DMS assay.

A world model needs (state, action, next state). A DMS assay gives measured variants, so
a transition exists wherever two measured variants differ by exactly one substitution and
one's mutation set is a subset of the other's: `A24G` -> `A24G:L56P` is the action
"substitute P at position 56". The `planning` instance alone yields 1,395,965 of these.

**On the state representation.** `--state pooled` uses the mean-pooled 1280-d encoder
output, which is exact and is what the reward model consumes. `--state residue` wants the
full per-residue latent Z in R^(L x d), which this project does NOT currently have for
variants: `data/processed/positional_cache/` stores each variant's latents only at its own
mutated positions, plus one full wild-type.

Reconstructing a variant state as "wild-type rows, with the mutated rows substituted in"
is the obvious workaround and it does not work. Measured on `CAPSD_AAV2S_Sinai_2021`:

    cosine(reconstructed pooled, true pooled)   0.9991
    ||true - wt||                               0.141
    ||reconstructed - wt||                      0.023
    fraction of the displacement captured       0.052
    direction agreement of that displacement    0.30

The cosine looks excellent because both vectors are dominated by the wild type. Against
the displacement that actually distinguishes one variant from another, the reconstruction
keeps **5%**, pointing somewhere else. This is Step 1.4's finding — only 10.6% of the
encoder's movement lands on the substituted residue — showing up at the pooled level. A
model trained on such states would be fitting a shadow of the transition while every
cosine-based diagnostic reported success, so the option is refused rather than offered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biodreamer.protein_dreamer.cache import sequence_hash
from biodreamer.reward.data import detection_limits, load_assay

CACHE = Path("data/processed/pooled_cache")
BACKBONE = "westlake-repl_SaProt_650M_AF2"
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}


@dataclass
class Transitions:
    """One assay's transitions, as index arrays into `z`."""

    z: np.ndarray  # (n_variants, d) encoder states
    src: np.ndarray  # (n_trans,) index of z_t
    dst: np.ndarray  # (n_trans,) index of z_{t+1}
    position: np.ndarray  # (n_trans,) 0-based residue index of the action
    aa: np.ndarray  # (n_trans,) amino-acid index of the action
    score: np.ndarray  # (n_variants,) measured fitness, for the reward term
    assay: str
    length: int
    coverage: float  # share of the assay's variants that had an exact cached state

    def __len__(self) -> int:
        return len(self.src)


def _parse(mut: str) -> tuple[int, int]:
    """'A24G' -> (23, index of G). Positions are 1-based in the assay files."""
    return int(mut[1:-1]) - 1, AA_INDEX[mut[-1]]


def build(assay: str, max_transitions: int | None = None,
          seed: int = 0) -> Transitions:
    a = load_assay(assay, detection_limits())
    z = np.load(CACHE / f"{BACKBONE}__{assay}.npz")
    row = {k: i for i, k in enumerate(z["keys"])}
    emb = z["emb"]

    # The pooled cache was built for the H1 and literature splits, not for whole assays,
    # so coverage runs 46% .. 100% here. A transition is only formed when BOTH of its
    # endpoints have an exact cached state: the alternative is re-deriving the missing
    # one, and the reconstruction this module's docstring measures keeps 5% of the
    # displacement. Fewer, exact transitions beat more, wrong ones.
    index = {}
    n_missing = 0
    for i, (m, s) in enumerate(zip(a.mutant, a.sequence)):
        if sequence_hash(s) not in row:
            n_missing += 1
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
            p, c = _parse(added)
            src.append(parent)
            dst.append(j)
            pos.append(p)
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

    # only the variants a transition touches need an encoder state in memory
    used = np.unique(np.concatenate([src, dst]))
    remap = np.full(len(a.mutant), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    states = np.stack(
        [emb[row[sequence_hash(s)]] for s in a.sequence[used]]
    ).astype(np.float32)

    return Transitions(
        z=states,
        src=remap[src],
        dst=remap[dst],
        position=pos,
        aa=aa,
        score=a.score[used].astype(np.float32),
        assay=assay,
        length=len(str(a.sequence[0])),
        coverage=coverage,
    )


def split(n: int, seed: int = 0, frac_val: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Transition-level train/validation split."""
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1.0 - frac_val))
    return perm[:cut], perm[cut:]
