"""Kermut's structure kernel — the component this project tested as a scalar and never built.

Step 2.4 scored ProteinMPNN as a **mean term**: one number per variant, summed over its
mutated positions, added to `m(x)`. It bought nothing (−0.002 on within-order skill). That
result stands, and it does not say what it looked like it said, because Kermut does not use
inverse folding that way. Kermut compares **sites**: two variants are similar when they
mutate positions whose structural preferences are similar and which are close in space.
Collapsing that to a scalar throws away both halves.

The definition reproduced here is Kermut eq. 1–2 (Groth et al., NeurIPS 2024), checked
against the reference implementation in `kermut/kernels/_structure_kernel.py`:

    k_struct(x, x') = sum_{i in M(x)} sum_{j in M(x')} lambda * k_H(i,j) * k_p(i,j) * k_d(i,j)

      k_H(i,j) = exp(-g1 * d_Hellinger(p_i, p_j))   p_i = ProteinMPNN p(aa | structure) at site i
      k_p(i,j) = exp(-g2 * |log p_i(a_i) - log p_j(a_j)|)   the observed mutant residue
      k_d(i,j) = exp(-g3 * ||c_i - c_j||)           Euclidean CA distance in Angstrom

    d_Hellinger(p,q) = sqrt(0.5 * sum_a (sqrt(p_a) - sqrt(q_a))^2)

g1, g2, g3 and lambda are learned by marginal likelihood, as everywhere else here.

**Computing it as written is the wrong way round.** A naive implementation loops over pairs
of variants and, inside that, over pairs of mutated sites: for 300 training against 494,494
test variants at ~5 mutations each that is order 10^10 kernel evaluations per optimiser
step. But every term depends on the two *mutations*, never on the variants carrying them,
and a variant is just a **set** of mutations. So with `S` the sparse binary matrix saying
which mutations each variant has, and `Phi` the mutation-by-mutation matrix of
k_H * k_p * k_d,

    k_struct(X, X') = S @ Phi @ S'.T

exactly. The mutation vocabulary is at most L * 19 (4,180 for the largest assay here), so
`Phi` is small and fixed in shape, the per-step cost stops depending on how many variants
there are, and the three distance matrices underneath it are precomputed once because only
the gammas change between steps.

Two honest deviations from the published setup, both recorded rather than papered over:

  1. **Structures are ESMFold predictions**, not the AlphaFold2 structures ProteinGym ships
     that Kermut used. Native-sequence recovery across these 13 runs 14.2% … 49.3%.
  2. **ProteinMPNN is run with `--unconditional_probs_only`** (one pass, no sequence
     context), where Kermut decodes the wild type ten times and averages. Both estimate
     p(aa | structure); they are not the same estimator.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.spatial import distance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroshot_baseline import AA  # noqa: E402

PDB_DIR = Path("data/processed/wt_structures")
MPNN = Path("data/processed/mpnn_matrix.npz")
AA_INDEX = {a: i for i, a in enumerate(AA)}


def ca_coords(pdb: Path) -> np.ndarray:
    """Alpha-carbon coordinates, one row per residue, in file order."""
    out = []
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            out.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class SiteData:
    """Per-site structural quantities for one assay's wild type."""

    logp: np.ndarray  # (L, 20) ProteinMPNN log p(aa | structure), AA order
    coords: np.ndarray  # (L, 3) CA coordinates

    @property
    def length(self) -> int:
        return self.logp.shape[0]


def load_site_data(assay: str) -> SiteData:
    with np.load(MPNN) as m:
        if assay not in m.files:
            raise KeyError(f"{MPNN} has no ProteinMPNN matrix for {assay}")
        logp = m[assay].astype(np.float64)
    pdb = PDB_DIR / f"{assay}.pdb"
    if not pdb.exists():
        raise FileNotFoundError(f"{pdb} missing -- fold the wild type first")
    coords = ca_coords(pdb)
    if coords.shape[0] != logp.shape[0]:
        raise ValueError(
            f"{assay}: {coords.shape[0]} CA atoms but ProteinMPNN has {logp.shape[0]} "
            "sites -- the structure and the matrix are not the same protein"
        )
    return SiteData(logp=logp, coords=coords)


def parse_mutations(mutant: str) -> list[tuple[int, int]]:
    """'A24G:L56P' -> [(23, aa_index['G']), (55, aa_index['P'])]; positions are 1-based."""
    out = []
    for tok in mutant.split(":"):
        tok = tok.strip()
        if not tok:
            continue
        out.append((int(tok[1:-1]) - 1, AA_INDEX[tok[-1]]))
    return out


def build_indicator(
    mutants: np.ndarray,
    vocab: dict[tuple[int, int], int] | None = None,
    grow: bool | None = None,
) -> tuple[sparse.csr_matrix, dict[tuple[int, int], int]]:
    """Sparse (n_variants, n_mutations) binary matrix, plus the vocabulary it indexes.

    Passing an existing `vocab` restricts to it and drops anything else. **That is only
    safe when the vocabulary already covers every mutation the caller will score**, which
    is why `assay_vocabulary` below builds one over train and test together.

    It is tempting to build the vocabulary from the training set alone and let test-only
    mutations fall out, on the grounds that the model never saw them. That is wrong here.
    `Phi[a, b]` is a similarity between two *mutations* and is non-zero for every pair, so
    a test-only mutation still contributes `Phi[a, b]` against each training mutation `b`
    in the cross covariance. Dropping it does not zero an unused term, it silently
    computes a different kernel -- and on this protocol, where the test set is order >= 3
    and the training set order <= 2, test-only mutations are the common case.
    """
    grow = (vocab is None) if grow is None else grow
    vocab = {} if vocab is None else vocab
    rows, cols = [], []
    for r, mut in enumerate(mutants):
        for key in parse_mutations(str(mut)):
            idx = vocab.get(key)
            if idx is None:
                if not grow:
                    continue
                idx = vocab[key] = len(vocab)
            rows.append(r)
            cols.append(idx)
    S = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(mutants), max(len(vocab), 1))
    )
    return S, vocab


def assay_vocabulary(
    mutants_train: np.ndarray, mutants_test: np.ndarray
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[tuple[int, int]], np.ndarray]:
    """One vocabulary over train and test, so no pair of mutations is silently dropped.

    Returns the two indicator matrices, the vocabulary as an ordered list, and the
    positions within it of the training mutations -- the cross kernel only ever needs
    `Phi[:, train_cols]`, which is at most (L*19) x (2 * n_train) and therefore small
    however many test variants there are.
    """
    S_tr, vocab = build_indicator(mutants_train)
    n_train_mut = len(vocab)
    S_te, vocab = build_indicator(mutants_test, vocab, grow=True)
    # build_indicator sized S_tr before the test mutations were added; widen it
    S_tr = sparse.csr_matrix(
        (S_tr.data, S_tr.indices, S_tr.indptr), shape=(S_tr.shape[0], len(vocab))
    )
    keys = [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    return S_tr, S_te, keys, np.arange(n_train_mut)


def pair_distances(
    va: list[tuple[int, int]], vb: list[tuple[int, int]], site: SiteData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three mutation-by-mutation distance matrices the gammas scale.

    Precomputed because they do not depend on any hyperparameter: an optimiser step only
    re-exponentiates them.
    """
    sa = np.array([s for s, _ in va])
    sb = np.array([s for s, _ in vb])
    aa_a = np.array([a for _, a in va])
    aa_b = np.array([a for _, a in vb])

    # `cdist` rather than the Gram expansion ||u-v||^2 = |u|^2 + |v|^2 - 2<u,v>. The
    # expansion is the fast way and it is wrong here: for two sites with the SAME
    # distribution it subtracts two near-equal numbers and returns ~1e-16 instead of 0,
    # whose square root is ~1e-8 -- a distance eight orders of magnitude larger than the
    # error, sitting exactly on the diagonal the kernel most depends on. A unit test
    # caught it. These matrices are built once per assay, so the direct form is free.
    root = np.sqrt(np.exp(site.logp))  # (L, 20)
    d_hell = distance.cdist(root[sa], root[sb], "euclidean") / np.sqrt(2.0)

    lp_a = site.logp[sa, aa_a]
    lp_b = site.logp[sb, aa_b]
    d_lp = np.abs(lp_a[:, None] - lp_b[None, :])

    d_euc = distance.cdist(site.coords[sa], site.coords[sb], "euclidean")
    return d_hell, d_lp, d_euc


class StructureKernel(torch.nn.Module):
    """k_H * k_p * k_d over mutation pairs, applied as S @ Phi @ S'.T."""

    def __init__(self) -> None:
        super().__init__()
        # Kermut initialises all three lengthscales at 1.0; these are their logs.
        self.log_g1 = torch.nn.Parameter(torch.zeros(1))
        self.log_g2 = torch.nn.Parameter(torch.zeros(1))
        self.log_g3 = torch.nn.Parameter(torch.zeros(1))

    def phi(self, d_hell: torch.Tensor, d_lp: torch.Tensor, d_euc: torch.Tensor):
        return torch.exp(
            -self.log_g1.exp() * d_hell
            - self.log_g2.exp() * d_lp
            - self.log_g3.exp() * d_euc
        )

    def forward(self, sa, sb: torch.Tensor, dists) -> torch.Tensor:
        """k_struct for every (row of `sa`, row of `sb`) pair.

        `sb` is always the DENSE training indicator, narrow by construction (at most two
        mutations per training variant). `sa` may be dense, for the training block where
        gradients are needed, or a scipy sparse matrix at prediction time: a test chunk
        spans the whole mutation vocabulary, and densifying 20,000 x 4,180 would cost
        668 MB per block to hold mostly zeros. Under `no_grad` the sparse product gives
        the same numbers for a few MB.
        """
        phi = self.phi(*dists)
        if sparse.issparse(sa):
            return torch.as_tensor(sa @ phi.detach().numpy()) @ sb.T
        return sa @ phi @ sb.T
