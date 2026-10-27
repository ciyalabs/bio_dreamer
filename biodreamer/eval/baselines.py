"""Baselines every planning claim must beat — Step 0.5 of RESEARCH_PLAN.md.

These are implemented **before** the model, deliberately. Built afterwards, the
additive null gets written late, under pressure, by someone who wants it to lose.
\\citet{visani2026additive} re-analysed a Science paper's epistasis claims and
found its multimutant predictions almost perfectly correlated with an additive
model's, so the engineering reduced to combining the largest individual effects —
a strategy four decades old. Any claim this project makes about epistasis or
valley crossing is measured against these.

    AdditiveNull     linear model over single-mutant effects. The decisive one.
    StackTopK        combine the k individually most beneficial substitutions.
    GreedyWalk       hill-climb one substitution at a time.
    RandomSearch     sample within the same mutational budget.

All four share one interface: given a landscape and an oracle, propose candidates
under a budget. That makes them interchangeable with the learned policy in Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .landscape import Landscape

Oracle = Callable[[str], float]


def _one_hot(variants: np.ndarray, alphabet: list[str]) -> np.ndarray:
    """(n, n_sites) residues -> (n, n_sites * |alphabet|) indicator matrix."""
    n, n_sites = variants.shape
    idx = {a: j for j, a in enumerate(alphabet)}
    X = np.zeros((n, n_sites * len(alphabet)), dtype=float)
    for i in range(n):
        for s in range(n_sites):
            a = variants[i, s]
            if a in idx:
                X[i, s * len(alphabet) + idx[a]] = 1.0
    return X


@dataclass
class AdditiveNull:
    """Additive (site-independent) model of fitness.

    By construction it cannot represent epistasis, which is exactly what makes it
    the right null: if a model's predictions correlate near-perfectly with this
    one, the model is not using epistasis whatever its architecture.

    Fit either on everything (`fit`) or on single mutants only
    (`fit_on_singles`). The second is the honest setting for H4: it asks whether
    higher-order fitness is predictable from single-mutant effects alone.
    """

    alphabet: list[str] = field(default_factory=list)
    coef: np.ndarray | None = None
    intercept: float = 0.0
    n_sites: int = 0
    ridge: float = 1e-6

    def fit(self, landscape: Landscape) -> "AdditiveNull":
        self.alphabet = landscape.alphabet
        self.n_sites = landscape.n_sites
        X = _one_hot(landscape.variants, self.alphabet)
        A = np.c_[X, np.ones(len(X))]
        # ridge keeps the fit defined when a residue is unobserved at a site
        reg = self.ridge * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        beta = np.linalg.solve(A.T @ A + reg, A.T @ landscape.fitness)
        self.coef, self.intercept = beta[:-1], float(beta[-1])
        return self

    def fit_on_singles(self, landscape: Landscape) -> "AdditiveNull":
        """Fit using only the wild type and single mutants.

        The alphabet is taken from the full landscape so the model can score
        higher-order variants it never saw during fitting.
        """
        singles = landscape.singles()
        full_alphabet = landscape.alphabet
        self.alphabet = full_alphabet
        self.n_sites = landscape.n_sites
        X = _one_hot(singles.variants, full_alphabet)
        A = np.c_[X, np.ones(len(X))]
        reg = self.ridge * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        beta = np.linalg.solve(A.T @ A + reg, A.T @ singles.fitness)
        self.coef, self.intercept = beta[:-1], float(beta[-1])
        return self

    def predict(self, variants: np.ndarray | list[str]) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("AdditiveNull must be fitted before predicting")
        if isinstance(variants, list):
            variants = np.array([list(s) for s in variants])
        X = _one_hot(variants, self.alphabet)
        return X @ self.coef + self.intercept

    def r2(self, landscape: Landscape) -> float:
        pred = self.predict(landscape.variants)
        y = landscape.fitness
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot == 0:
            return float("nan")
        return 1.0 - float(((y - pred) ** 2).sum()) / ss_tot

    def propose(self, landscape: Landscape, budget: int) -> list[str]:
        """Top `budget` sequences by additive prediction — the null's design campaign."""
        pred = self.predict(landscape.variants)
        order = np.argsort(pred)[::-1][:budget]
        return ["".join(landscape.variants[i]) for i in order]


@dataclass
class StackTopK:
    """Combine the k individually most beneficial substitutions.

    The forty-year-old heuristic. Distinct from AdditiveNull: it does not fit a
    model, it just ranks single mutants by measured effect and stacks the best.
    """

    k: int = 4

    def propose(self, landscape: Landscape, budget: int, oracle: Oracle | None = None) -> list[str]:
        idx = landscape.index()
        oracle = oracle or (lambda s: idx.get(s, float("-inf")))
        wt = landscape.wild_type
        wt_fit = landscape.wild_type_fitness()

        # measured effect of each single substitution
        effects: list[tuple[float, int, str]] = []
        for i in range(landscape.n_sites):
            for a in landscape.alphabet:
                if a == wt[i]:
                    continue
                s = wt[:i] + a + wt[i + 1 :]
                f = oracle(s)
                if np.isfinite(f):
                    effects.append((f - wt_fit, i, a))
        effects.sort(reverse=True)

        # stack the best substitution at each distinct site, deepest first
        out: list[str] = []
        used: dict[int, str] = {}
        for _eff, i, a in effects:
            if i in used:
                continue
            used[i] = a
            seq = list(wt)
            for j, b in used.items():
                seq[j] = b
            out.append("".join(seq))
            if len(used) >= min(self.k, landscape.n_sites) or len(out) >= budget:
                break
        return out[:budget]


@dataclass
class GreedyWalk:
    """Hill-climb one substitution at a time, taking the best improving neighbour.

    The control H4 is defined against: it isolates the contribution of planning
    from the contribution of the reward model, because it runs on the same oracle.
    """

    max_steps: int = 10
    restarts: int = 1
    seed: int = 0

    def walk(self, landscape: Landscape, oracle: Oracle | None = None, start: str | None = None) -> list[str]:
        idx = landscape.index()
        oracle = oracle or (lambda s: idx.get(s, float("-inf")))
        cur = start or landscape.wild_type
        path = [cur]
        for _ in range(self.max_steps):
            nbrs = landscape.neighbours(cur)
            if not nbrs:
                break
            scores = [oracle(s) for s in nbrs]
            best = int(np.argmax(scores))
            if scores[best] <= oracle(cur):
                break  # local optimum
            cur = nbrs[best]
            path.append(cur)
        return path

    def propose(self, landscape: Landscape, budget: int, oracle: Oracle | None = None) -> list[str]:
        rng = np.random.default_rng(self.seed)
        seqs = landscape.sequences()
        out: list[str] = []
        for r in range(max(1, self.restarts)):
            start = landscape.wild_type if r == 0 else seqs[int(rng.integers(len(seqs)))]
            out.extend(self.walk(landscape, oracle, start))
            if len(out) >= budget:
                break
        # de-duplicate, preserving order
        seen: set[str] = set()
        uniq = [s for s in out if not (s in seen or seen.add(s))]
        return uniq[:budget]


@dataclass
class RandomSearch:
    """Uniform sampling within the same budget. The floor every method must clear."""

    seed: int = 0

    def propose(self, landscape: Landscape, budget: int) -> list[str]:
        rng = np.random.default_rng(self.seed)
        seqs = landscape.sequences()
        k = min(budget, len(seqs))
        return [seqs[i] for i in rng.choice(len(seqs), size=k, replace=False)]
