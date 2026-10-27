"""Evaluation metrics — Step 0.5 of RESEARCH_PLAN.md.

Four families, matching the plan:

    fitness      per-assay Spearman, top-k recovery, global-optimum hit rate
    trajectory   reciprocal-sign-epistasis traps, path accessibility, error growth
    uncertainty  calibration curve, risk-coverage
    falsification correlation between a model and the additive null

The last one is not a diagnostic, it is a verdict. A model whose predictions
correlate near-perfectly with the additive null is not using epistasis, whatever
it scores, and H4 is falsified regardless of absolute performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.stats import spearmanr

from .landscape import Landscape


# ---- fitness ----


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank correlation; nan when either side is constant."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def spearman_per_assay(groups: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Spearman within each assay. Never pool raw fitness across assays."""
    return {k: spearman(t, p) for k, (t, p) in groups.items()}


def top_k_recovery(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 10, budget: int | None = None
) -> float:
    """Fraction of the true top-k found in the predicted top-`budget`."""
    budget = budget or k
    true_top = set(np.argsort(y_true)[::-1][:k].tolist())
    pred_top = set(np.argsort(y_pred)[::-1][:budget].tolist())
    return len(true_top & pred_top) / k


def global_optimum_hit_rate(proposals: list[str], landscape: Landscape) -> float:
    """1.0 if the landscape's global optimum is among the proposals, else 0.0."""
    best, _ = landscape.global_optimum()
    return float(best in set(proposals))


def best_fitness_found(proposals: list[str], landscape: Landscape) -> float:
    idx = landscape.index()
    vals = [idx[s] for s in proposals if s in idx]
    return float(max(vals)) if vals else float("nan")


# ---- trajectory ----


@dataclass
class TrapReport:
    n_pairs: int
    n_reciprocal_sign: int

    @property
    def fraction(self) -> float:
        return self.n_reciprocal_sign / self.n_pairs if self.n_pairs else float("nan")


def reciprocal_sign_epistasis(landscape: Landscape, max_pairs: int = 20_000) -> TrapReport:
    """Count site-pairs where both single steps are deleterious but the double is better.

    This is the trap H4 is about: greedy hill-climbing cannot cross it.
    """
    idx = landscape.index()
    wt = landscape.wild_type
    wt_f = landscape.wild_type_fitness()
    n_pairs = n_rse = 0
    rng = np.random.default_rng(0)

    sites = range(landscape.n_sites)
    for i in sites:
        for j in sites:
            if j <= i:
                continue
            for a in landscape.alphabet:
                if a == wt[i]:
                    continue
                for b in landscape.alphabet:
                    if b == wt[j]:
                        continue
                    if n_pairs >= max_pairs:
                        return TrapReport(n_pairs, n_rse)
                    s_i = wt[:i] + a + wt[i + 1 :]
                    s_j = wt[:j] + b + wt[j + 1 :]
                    d = list(wt)
                    d[i], d[j] = a, b
                    s_ij = "".join(d)
                    if not all(s in idx for s in (s_i, s_j, s_ij)):
                        continue
                    n_pairs += 1
                    if idx[s_i] < wt_f and idx[s_j] < wt_f and idx[s_ij] > wt_f:
                        n_rse += 1
    return TrapReport(n_pairs, n_rse)


def path_accessibility(landscape: Landscape, target: str | None = None) -> float:
    """Fraction of shortest mutational paths to `target` that increase monotonically.

    Low accessibility means the target is reachable only by crossing a valley,
    which is precisely where planning should beat greedy.
    """
    target = target or landscape.global_optimum()[0]
    wt = landscape.wild_type
    idx = landscape.index()
    diff = [i for i in range(landscape.n_sites) if wt[i] != target[i]]
    if not diff:
        return float("nan")

    total = accessible = 0
    for order in permutations(diff):
        cur = list(wt)
        prev = idx.get(wt, float("-inf"))
        ok = True
        for i in order:
            cur[i] = target[i]
            f = idx.get("".join(cur))
            if f is None or f <= prev:
                ok = False
                break
            prev = f
        total += 1
        accessible += int(ok)
    return accessible / total if total else float("nan")


def error_growth(errors_by_step: dict[int, float]) -> float:
    """Slope of error against rollout length — how fast prediction compounds."""
    steps = np.array(sorted(errors_by_step))
    vals = np.array([errors_by_step[s] for s in steps], float)
    if len(steps) < 2:
        return float("nan")
    return float(np.polyfit(steps, vals, 1)[0])


# ---- uncertainty ----


def calibration_curve(
    errors: np.ndarray, uncertainty: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Mean error per uncertainty bin. A calibrated model gives a rising curve."""
    errors, uncertainty = np.asarray(errors, float), np.asarray(uncertainty, float)
    order = np.argsort(uncertainty)
    errors, uncertainty = errors[order], uncertainty[order]
    bins = np.array_split(np.arange(len(errors)), n_bins)
    return (
        np.array([uncertainty[b].mean() for b in bins if len(b)]),
        np.array([errors[b].mean() for b in bins if len(b)]),
    )


def risk_coverage(errors: np.ndarray, uncertainty: np.ndarray, n_points: int = 20):
    """Error among the most-confident fraction, as coverage increases.

    A useful uncertainty estimate gives rising risk with coverage; a flat curve
    means the uncertainty carries no information about correctness.
    """
    errors, uncertainty = np.asarray(errors, float), np.asarray(uncertainty, float)
    order = np.argsort(uncertainty)
    errors = errors[order]
    cov = np.linspace(1 / n_points, 1.0, n_points)
    risk = np.array([errors[: max(1, int(c * len(errors)))].mean() for c in cov])
    return cov, risk


def calibration_slope(errors: np.ndarray, uncertainty: np.ndarray) -> float:
    """Single number for a calibration curve; <=0 means uninformative uncertainty."""
    u, e = calibration_curve(errors, uncertainty)
    if len(u) < 2 or np.ptp(u) == 0:
        return float("nan")
    return float(np.polyfit(u, e, 1)[0])


# ---- falsification ----


@dataclass
class AdditiveComparison:
    """The verdict an epistasis claim must survive."""

    model_score: float
    null_score: float
    correlation_with_null: float
    margin: float
    falsified: bool
    threshold: float

    def __str__(self) -> str:
        v = "FALSIFIED" if self.falsified else "survives"
        return (
            f"model {self.model_score:.4f} vs additive null {self.null_score:.4f} "
            f"(margin {self.margin:+.4f}); corr with null {self.correlation_with_null:.4f} "
            f"[limit {self.threshold:.2f}] -> {v}"
        )


def report_additive_comparison(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    null_pred: np.ndarray,
    threshold: float = 0.95,
    metric=spearman,
) -> AdditiveComparison:
    """Compare a model against the additive null, and say whether the claim survives.

    Every planning experiment calls this. Reporting only the model's score is how
    a reparameterised additive model gets published as epistasis learning.
    """
    m = metric(y_true, model_pred)
    n = metric(y_true, null_pred)
    corr = spearman(model_pred, null_pred)
    return AdditiveComparison(
        model_score=m,
        null_score=n,
        correlation_with_null=corr,
        margin=m - n,
        falsified=bool(np.isfinite(corr) and corr >= threshold),
        threshold=threshold,
    )
