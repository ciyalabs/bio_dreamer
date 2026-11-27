"""Losses for the reward head — Step 2.1.

Two targets, two losses, because the data has two different problems.

**Fitness: pairwise ranking, with censored pairs dropped.** Raw DMS scores are not
comparable across assays -- different units, different dynamic ranges, different
organisms -- so regressing them jointly is meaningless however the values are scaled.
Ranking within an assay sidesteps that entirely, and it is also what H1 is scored on
(per-assay Spearman).

It also resolves the censoring problem measured in `scripts/censoring_audit.py`. Three
of the seven H1 assays carry a detection limit, D7PM05 with **31.9% of its variants at
one exact value**. A pair of two floored variants carries no ordering information and
would teach a false tie-break; a pair of one floored and one live variant carries the
real signal that one is dead. So censored points are kept, and used only for the
comparisons they can support.

**Stability: inverse-variance weighted SmoothL1.** Tsuboyama's dG is rail-clipped to
[-1, 5] but only 0.01% of values reach a rail, so there is nothing to censor. What
varies is precision -- 9.7% of measurements have a 95% CI wider than 1 kcal/mol -- and
weighting by it uses every point in proportion to what it is worth. SmoothL1 rather
than MSE because the ddG distribution has a long destabilising tail that squared error
would let dominate the gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    ci95: torch.Tensor | None = None,
    beta: float = 0.5,
    ci_floor: float = 0.1,
) -> torch.Tensor:
    """SmoothL1 on stability, each point weighted by the precision of its measurement.

    `ci95` is the published 95% confidence interval width in kcal/mol. Weight is
    `1 / max(ci95, ci_floor)^2` -- inverse variance, with a floor so that a handful of
    implausibly precise measurements cannot dominate the batch. Weights are normalised
    to mean 1, so `beta` and the learning rate keep their meaning whether or not
    confidence intervals are supplied.

    With `ci95=None` every point is weighted equally, which is the right behaviour for
    a dataset that does not publish error bars rather than a reason to invent them.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ")
    per_point = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    if ci95 is None:
        return per_point.mean()
    if ci95.shape != target.shape:
        raise ValueError(f"ci95 {tuple(ci95.shape)} does not match target")
    w = 1.0 / torch.clamp(ci95, min=ci_floor) ** 2
    w = w / w.mean().clamp(min=1e-12)
    return (per_point * w).mean()


def ranking_pairs(
    target: torch.Tensor,
    assay: torch.Tensor,
    censored: torch.Tensor | None = None,
    max_pairs: int | None = 4096,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices (i, j) of comparable pairs, with target[i] > target[j].

    A pair is comparable when both members come from the same assay, their targets
    differ, and they are not **both** censored. Returns two index tensors; empty when
    nothing in the batch is comparable, which a caller must handle rather than assume
    away.

    `max_pairs` subsamples uniformly. A batch of n variants from one assay has O(n^2)
    pairs, which at n=1024 is a million comparisons per step for no extra signal.
    """
    if target.ndim != 1:
        raise ValueError(f"target must be 1-D, got {tuple(target.shape)}")
    if assay.shape != target.shape:
        raise ValueError("assay must have one entry per target")
    n = target.numel()
    if n < 2:
        empty = torch.empty(0, dtype=torch.long, device=target.device)
        return empty, empty

    same = assay[:, None] == assay[None, :]
    greater = target[:, None] > target[None, :]
    ok = same & greater
    if censored is not None:
        if censored.shape != target.shape:
            raise ValueError("censored must have one entry per target")
        # drop only pairs where BOTH are at the detection limit; a censored point
        # still orders correctly against a live one
        both = censored[:, None] & censored[None, :]
        ok = ok & ~both

    i, j = torch.nonzero(ok, as_tuple=True)
    if max_pairs is not None and i.numel() > max_pairs:
        sel = torch.randperm(i.numel(), device=i.device, generator=generator)[:max_pairs]
        i, j = i[sel], j[sel]
    return i, j


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    assay: torch.Tensor,
    censored: torch.Tensor | None = None,
    max_pairs: int | None = 4096,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """RankNet loss over comparable pairs: -log sigma(pred_i - pred_j).

    Scale-free and monotone-invariant, so per-assay standardisation of the target is
    not needed for this loss -- the ordering is all it reads. Returns a zero that still
    carries a gradient path when no pair is comparable, so a degenerate batch does not
    break the step.
    """
    i, j = ranking_pairs(target, assay, censored, max_pairs, generator)
    if i.numel() == 0:
        return pred.sum() * 0.0
    return F.softplus(-(pred[i] - pred[j])).mean()


def standardise_per_assay(
    value: torch.Tensor, assay: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Zero-mean, unit-variance within each assay.

    The ranking loss does not need this -- it reads order only. It is here for the
    diagnostics that *do* pool across assays, where raw DMS scores would make an
    assay with a wide dynamic range dominate any aggregate. `PREREGISTRATION.md`
    standing rule 5: never pool raw fitness across assays.
    """
    out = torch.empty_like(value, dtype=torch.float32)
    for a in torch.unique(assay):
        m = assay == a
        v = value[m].float()
        out[m] = (v - v.mean()) / v.std().clamp(min=eps) if v.numel() > 1 else 0.0
    return out
