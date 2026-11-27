"""Reward head — Step 2.1.

    Z_t (L x 1280)  ->  adapter (per residue)  ->  Z_t (L x 128)
                                                      |
                                                    pool          <- the ONLY pooling
                                                      |
                                                    trunk
                                                   /      \\
                                           stability     fitness

**Pooling happens here and nowhere else.** The world model and the inverse-dynamics
anchor both read the per-residue latent; Step 1.4 measured why, and it is categorical
rather than quantitative: a pooled latent has no position to predict, so the anchor
cannot be formulated on it at all. Only the reward head needs a single number per
protein, so only the reward head pools.

`d = 128` is fixed by Step 1.3, not chosen: an out-of-fold sweep gave 0.449 / 0.473 /
0.493 / 0.511 at d = 64 / 128 / 256 / 512 against 0.459 for the unprojected 1280, so
128 already matches the full backbone dimension and a fourfold larger latent buys at
most 0.04. It also makes the cache fit -- 170 GiB for ProteinGym instead of 1.7 TiB.

The backbone is frozen. Only the adapter, trunk and heads train, which is what makes
training on a cached latent possible at all: the 650M encoder never runs during a
reward-head epoch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RewardConfig:
    """Shapes and capacity. Defaults are the measured ones, not guesses."""

    backbone_dim: int = 1280  # ESM-2 650M and SaProt 650M both
    latent_dim: int = 128     # fixed by step 1.3
    trunk_dim: int = 256
    dropout: float = 0.1
    pool: str = "mean"
    # Adapt each residue then pool (the architecture the world model shares), or pool
    # the backbone output first and adapt the single vector. The second makes the
    # cache 512x smaller -- 2.4 GB instead of 615 GB for the benchmark instance -- and
    # is only legitimate if it costs nothing, which scripts/train_reward.py measures.
    pool_first: bool = False


class Adapter(nn.Module):
    """Per-residue projection from the frozen backbone to the project's latent.

    Applied independently at every position -- it must not mix residues, because the
    per-residue structure is what Steps 1.3 and 1.4 showed is load-bearing. A LayerNorm
    first, because frozen backbone activations are not normalised for downstream use
    and their scale varies by orders of magnitude between layers and checkpoints.
    """

    def __init__(self, cfg: RewardConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.backbone_dim),
            nn.Linear(cfg.backbone_dim, cfg.latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, L, backbone_dim) -> (B, L, latent_dim)."""
        if z.ndim != 3:
            raise ValueError(f"expected (B, L, d), got {tuple(z.shape)}")
        return self.net(z)


def masked_pool(z: torch.Tensor, mask: torch.Tensor | None, how: str = "mean") -> torch.Tensor:
    """(B, L, d) -> (B, d), ignoring padded positions.

    Padding is not optional bookkeeping here: a batch mixing a 36-residue domain with a
    390-residue one is 90% padding for the short one, and an unmasked mean would divide
    its signal by ten.
    """
    if mask is None:
        return z.mean(dim=1) if how == "mean" else z.max(dim=1).values
    m = mask.unsqueeze(-1).to(z.dtype)
    if how == "mean":
        return (z * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    return z.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values


class RewardHead(nn.Module):
    """Shared trunk on the pooled latent, with a scalar head per target.

    Two heads rather than the legacy four. ARCHITECTURE_DELTA.md marked the old
    `reward.py` for rewrite: it predicted ddG, K_d and k_cat, and the project has
    training data for the first and for DMS fitness, not for the other two. A head
    with no data is not an ablation, it is dead weight that still consumes capacity.
    """

    def __init__(self, cfg: RewardConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RewardConfig()
        self.adapter = Adapter(self.cfg)
        self.trunk = nn.Sequential(
            nn.Linear(self.cfg.latent_dim, self.cfg.trunk_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.trunk_dim, self.cfg.trunk_dim),
            nn.GELU(),
        )
        self.stability = nn.Linear(self.cfg.trunk_dim, 1)
        self.fitness = nn.Linear(self.cfg.trunk_dim, 1)

    def latents(self, z: torch.Tensor) -> torch.Tensor:
        """The per-residue latent the world model consumes. No pooling."""
        return self.adapter(z)

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """(B, L, backbone_dim) -> scalar predictions plus the latent, for reuse."""
        if self.cfg.pool_first:
            # z may already be (B, d) if it came from a pooled cache
            pooled_backbone = z if z.ndim == 2 else masked_pool(z, mask, self.cfg.pool)
            pooled = self.adapter.net(pooled_backbone)
            per_residue = None
        else:
            per_residue = self.adapter(z)
            pooled = masked_pool(per_residue, mask, self.cfg.pool)
        h = self.trunk(pooled)
        return {
            "stability": self.stability(h).squeeze(-1),
            "fitness": self.fitness(h).squeeze(-1),
            "latent": per_residue,
            "pooled": pooled,
        }

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
