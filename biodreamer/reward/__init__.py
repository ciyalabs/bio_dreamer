"""Reward head — Step 2.1 of RESEARCH_PLAN.md.

Grounds the latent in experimental measurement and supplies the signal every later
phase optimises. Two targets with different data problems, so two losses: see
`losses.py` for why fitness is ranked and stability is weighted.
"""

from .head import Adapter, RewardConfig, RewardHead, masked_pool
from .losses import (
    pairwise_ranking_loss,
    ranking_pairs,
    standardise_per_assay,
    weighted_smooth_l1,
)

__all__ = [
    "Adapter",
    "RewardConfig",
    "RewardHead",
    "masked_pool",
    "pairwise_ranking_loss",
    "ranking_pairs",
    "standardise_per_assay",
    "weighted_smooth_l1",
]
