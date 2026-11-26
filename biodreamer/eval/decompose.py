"""Interaction-order variance decomposition — the measurement Step 1.1 turns on.

A low additive R^2 is the project's central premise, and on its own it means
nothing. Three different things produce one:

    real epistasis        structure an additive model cannot represent
    measurement noise     variance no model can represent
    floor censoring       a distribution piled against the assay's detection limit

Step 0.2 reported `1 - additive_r2` on a full-data fit and called it the
non-additive fraction. On TEV that number is 0.952, and almost all of it is read
noise: restricted to variants with above-median read depth the additive model
goes from R^2 0.047 to 0.231, and a pairwise model reaches 0.485. The raw figure
was not a measure of epistasis.

What separates the three is whether the non-additive variance is *predictable*.
Fit models of increasing interaction order and score them out of fold:

    order 1    site-independent (the additive null)
    order 2    + all pairwise terms
    order 3    + all triplet terms

Real epistasis shows up as a large out-of-fold increment from order 1 to order 2.
Noise shows up as no increment at any order. A saturated order is excluded
rather than reported: on a combinatorially complete landscape the top-order cell
of a held-out genotype is unidentifiable from the training folds, so its CV score
is an artifact of the design, not a property of the landscape.

Regularisation is selected inside each outer fold, on an inner split of that
fold's training data only, so no test point influences the model that scores it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

# Fixed in advance and applied identically to every landscape and every order.
# Selected per outer fold on an inner split, never on the scoring fold.
DAMP_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def interaction_design(variants: np.ndarray, alphabet: list[str], order: int) -> sp.csr_matrix:
    """One-hot design matrix with every interaction term up to `order`.

    Each row has exactly one indicator per site-subset, so the matrix is sparse
    with `sum_k C(n_sites, k)` non-zeros per row regardless of alphabet size.
    A dense version of the order-3 TEV design would be 44 GB; this is 19 MB.
    """
    n, n_sites = variants.shape
    if not 1 <= order <= n_sites:
        raise ValueError(f"order must be in 1..{n_sites}, got {order}")

    alpha = np.array(alphabet)
    # searchsorted returns len(alpha) for a residue past the end, so clip before
    # indexing and let the equality check report it
    codes = np.clip(np.searchsorted(alpha, variants), 0, len(alpha) - 1)
    if not (alpha[codes] == variants).all():
        missing = sorted(set(variants[alpha[codes] != variants].tolist()))
        raise ValueError(f"variants contain residues absent from the alphabet: {missing}")

    A = len(alphabet)
    col_blocks: list[np.ndarray] = []
    offset = 0
    for k in range(1, order + 1):
        for sites in combinations(range(n_sites), k):
            c = np.zeros(n, dtype=np.int64)
            for s in sites:
                c = c * A + codes[:, s]
            col_blocks.append(c + offset)
            offset += A**k

    cols = np.concatenate(col_blocks)
    rows = np.tile(np.arange(n), len(col_blocks))
    return sp.csr_matrix(
        (np.ones(len(cols)), (rows, cols)), shape=(n, offset)
    )


def _ridge(X: sp.csr_matrix, y: np.ndarray, damp: float) -> tuple[np.ndarray, float]:
    """Ridge fit with the intercept absorbed by centring, so it is not penalised."""
    mu = float(y.mean())
    beta = lsqr(X, y - mu, damp=damp, atol=1e-8, btol=1e-8, iter_lim=500)[0]
    return beta, mu


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - float(((y - pred) ** 2).sum()) / ss_tot


def cv_r2(
    X: sp.csr_matrix,
    y: np.ndarray,
    folds: int = 5,
    damps: tuple[float, ...] = DAMP_GRID,
    seed: int = 0,
) -> tuple[float, float]:
    """Out-of-fold R^2, with ridge strength chosen inside each fold.

    Returns (R^2 over the concatenated out-of-fold predictions, median chosen damp).
    """
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(y)), folds)
    pred = np.empty(len(y), dtype=float)
    chosen: list[float] = []

    for i in range(folds):
        test = parts[i]
        train = np.concatenate([parts[j] for j in range(folds) if j != i])
        # inner split of the training data only — the test fold is never consulted
        cut = int(0.8 * len(train))
        inner_fit, inner_val = train[:cut], train[cut:]

        best_damp, best_score = damps[0], -np.inf
        for d in damps:
            beta, mu = _ridge(X[inner_fit], y[inner_fit], d)
            score = _r2(y[inner_val], X[inner_val] @ beta + mu)
            if np.isfinite(score) and score > best_score:
                best_damp, best_score = d, score

        beta, mu = _ridge(X[train], y[train], best_damp)
        pred[test] = X[test] @ beta + mu
        chosen.append(best_damp)

    return _r2(y, pred), float(np.median(chosen))


@dataclass
class OrderDecomposition:
    """Out-of-fold R^2 per interaction order, and the increment each one buys."""

    name: str
    n: int
    n_sites: int
    max_order: int
    r2_by_order: dict[int, float] = field(default_factory=dict)
    damp_by_order: dict[int, float] = field(default_factory=dict)
    saturated_orders: list[int] = field(default_factory=list)

    @property
    def additive(self) -> float:
        return self.r2_by_order.get(1, float("nan"))

    @property
    def pairwise_increment(self) -> float:
        """How much predictable variance pairwise terms add over site-independence.

        The number that separates epistasis from noise: positive and large means
        structure, ~zero means the non-additive variance is unpredictable.
        """
        if 2 not in self.r2_by_order:
            return float("nan")
        return self.r2_by_order[2] - self.r2_by_order[1]

    @property
    def best(self) -> float:
        return max(self.r2_by_order.values()) if self.r2_by_order else float("nan")

    @property
    def structured_nonadditive(self) -> float:
        """Predictable variance beyond site-independence — epistasis, net of noise."""
        return self.best - self.additive if self.r2_by_order else float("nan")

    def as_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.r2_by_order.items():
            out[f"cv_r2_order_{k}"] = v
        out["cv_r2_additive"] = self.additive
        out["cv_r2_best"] = self.best
        out["pairwise_increment"] = self.pairwise_increment
        out["structured_nonadditive_fraction"] = self.structured_nonadditive
        return out


def decompose(
    variants: np.ndarray,
    fitness: np.ndarray,
    alphabet: list[str],
    name: str = "",
    max_order: int | None = None,
    folds: int = 5,
    seed: int = 0,
) -> OrderDecomposition:
    """Fit orders 1..max_order and score each out of fold.

    `max_order` defaults to `n_sites - 1`, excluding the saturated top order: on a
    complete landscape the top-order term of a held-out genotype appears in no
    training row, so its CV score measures the design, not the landscape. On
    TrpB4 the excluded order 4 moves CV R^2 by +0.004 over order 3, confirming it
    carries nothing.
    """
    n_sites = variants.shape[1]
    cap = n_sites - 1 if n_sites > 1 else 1
    top = min(max_order if max_order is not None else cap, n_sites)
    dec = OrderDecomposition(
        name=name,
        n=len(fitness),
        n_sites=n_sites,
        max_order=top,
        saturated_orders=[o for o in range(top + 1, n_sites + 1)],
    )
    for order in range(1, top + 1):
        X = interaction_design(variants, alphabet, order)
        r2, damp = cv_r2(X, fitness, folds=folds, seed=seed)
        dec.r2_by_order[order] = r2
        dec.damp_by_order[order] = damp
    return dec


def noise_ceiling(fitness: np.ndarray, measurement_variance: np.ndarray) -> dict[str, float]:
    """The highest R^2 any model could reach, given the assay's own error bars.

    `measurement_variance` is the published per-variant variance of the fitness
    estimate. Its mean over variants is the part of the total variance that no
    model can explain, so `1 - mean(var)/var(fitness)` bounds every R^2 on this
    landscape from above. Reporting a non-additive fraction without it is how
    read noise gets published as epistasis.
    """
    y = np.asarray(fitness, float)
    v = np.asarray(measurement_variance, float)
    total = float(y.var(ddof=1))
    if total <= 0:
        return {"noise_fraction": float("nan"), "reliability_ceiling": float("nan")}
    frac = float(np.nanmean(v)) / total
    return {
        "noise_fraction": frac,
        "reliability_ceiling": 1.0 - frac,
        "noise_fraction_median": float(np.nanmedian(v)) / total,
    }


def walsh_variance_by_order(fitness: np.ndarray, n_sites: int) -> dict[int, float]:
    """Exact variance decomposition by interaction order, for a complete 2^d landscape.

    Cross-validated regression is the right tool when the landscape is large and
    incomplete. A hypercube is neither: every one of its 2^d corners is measured
    and each site has exactly two states, so the Walsh-Hadamard basis diagonalises
    the problem and the decomposition is exact and orthogonal — no model fitting,
    no regularisation, nothing to tune.

    Returns {order: fraction of variance}, summing to 1 over orders 1..n_sites.
    The constant term is excluded, so these are fractions of the variance, not of
    the raw second moment.

    `fitness` must be in binary corner order: corner index bit `n_sites-1-i`
    selects site `i`'s second state, which is the order `Hypercube.corners()`
    emits.
    """
    y = np.asarray(fitness, float)
    if len(y) != 2**n_sites:
        raise ValueError(f"expected {2**n_sites} corners for {n_sites} sites, got {len(y)}")

    a = y.copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            lo = a[i : i + h].copy()
            hi = a[i + h : i + 2 * h].copy()
            a[i : i + h] = lo + hi
            a[i + h : i + 2 * h] = lo - hi
        h *= 2
    a /= len(y)

    power = a**2
    total = float(power[1:].sum())  # == var(y); the i=0 term is the mean
    if total <= 0:
        return {k: float("nan") for k in range(1, n_sites + 1)}

    order = np.array([bin(i).count("1") for i in range(len(y))])
    return {k: float(power[order == k].sum()) / total for k in range(1, n_sites + 1)}
