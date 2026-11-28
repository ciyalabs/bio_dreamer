"""World Model B — deterministic latent dynamics over a frozen encoder.

Step 3.1. The pieces the plan fixes, and what each is for:

  adapter          trainable map from the frozen encoder state into the latent the
                   dynamics live in. The encoder stays frozen; this is the only thing
                   that shapes the representation.
  dynamics         deterministic predictor h_{t+1} = f(h_t, a). No EMA target -- the
                   target is `adapter(z_{t+1})` under stop-gradient, following the
                   LeWorldModel design where SIGReg replaces teacher-student machinery.
  inverse head     factorised cross-entropy over position and amino acid, recovering the
                   action from (h_t, h_{t+1}). The action space factorises as L x 20, so
                   two softmaxes are both simpler and stronger than a batch-contrastive
                   objective here.
  SIGReg           pushes the latent distribution towards an isotropic Gaussian. This is
                   the anti-collapse mechanism, and with no EMA it is the ONLY one.

**A warning the plan already carries.** Step 1.4 measured inverse-dynamics recovery at
99.9% top-1 for position and residue from these latents. The anchor task is nearly free,
so its gradient is nearly zero, and it should not be expected to hold the representation
up on its own. That is why collapse is instrumented in `collapse.py` rather than assumed
away, and why the remedy ladder is written down in RESEARCH_PLAN.md before it is needed.
"""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Sketched isotropic-Gaussian regularisation via random 1-D projections.

    The latent batch is projected onto `n_proj` random unit directions and each
    projection is compared against a standard normal with the Epps-Pulley statistic --
    the squared L2 distance between empirical and target characteristic functions,
    integrated against a Gaussian weight. It is the cheap sketch of "is this
    distribution isotropic Gaussian", which is what makes the full test tractable at
    batch size: a collapsed latent concentrates its projections and the statistic rises
    on every direction at once.

    Reported as well as optimised: `collapse.py` logs it over training, because a
    regulariser that is being satisfied while the rank falls is the failure mode worth
    catching.
    """

    def __init__(self, n_proj: int = 64, n_freq: int = 16, seed: int = 0) -> None:
        super().__init__()
        self.n_proj = n_proj
        g = torch.Generator().manual_seed(seed)
        # frequencies on a fixed grid; the Gaussian weight is folded into the mean
        self.register_buffer("t", torch.linspace(0.1, 3.0, n_freq))
        self.register_buffer("_g_seed", torch.tensor(seed))
        self._gen = g

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n, d = h.shape
        # drawn on CPU from the seeded generator and then moved: a CUDA generator would
        # make the directions depend on the device, so the same run on CPU and GPU would
        # report different SIGReg values for the same latent
        w = torch.randn(d, self.n_proj, generator=self._gen, dtype=h.dtype).to(h.device)
        w = w / w.norm(dim=0, keepdim=True).clamp(min=1e-12)
        # centre globally and divide by ONE shared scale, never per direction.
        # Standardising each projection to unit variance is the obvious thing and it
        # silently destroys the test: a latent whose variance lives in a single
        # direction projects to a scaled Gaussian on every direction, and once each is
        # rescaled to unit variance it is indistinguishable from isotropic. The
        # statistic then reads ~0 for total collapse. A unit test pins this.
        # For isotropic h with per-element variance s^2 and unit w, w^T h has variance
        # s^2, so the shared scale is exactly what makes the target N(0, 1).
        hc = h - h.mean(0, keepdim=True)
        p = (hc @ w) / (hc.std() + 1e-6)
        # empirical characteristic function at each frequency, against exp(-t^2/2)
        tp = self.t.view(1, 1, -1) * p.unsqueeze(-1)  # (n, n_proj, n_freq)
        re = tp.cos().mean(0) - torch.exp(-0.5 * self.t.pow(2)).view(1, -1)
        im = tp.sin().mean(0)
        weight = torch.exp(-0.5 * self.t.pow(2)).view(1, -1)
        return ((re.pow(2) + im.pow(2)) * weight).sum(-1).mean()


class WorldModelB(nn.Module):
    """Frozen encoder state -> latent -> deterministic one-step dynamics."""

    def __init__(self, d_in: int = 1280, d_model: int = 256, length: int = 256,
                 n_aa: int = 20, d_action: int = 64, hidden: int = 512,
                 seed: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.adapter = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        self.pos_emb = nn.Embedding(length, d_action)
        self.aa_emb = nn.Embedding(n_aa, d_action)
        self.dynamics = nn.Sequential(
            nn.Linear(d_model + 2 * d_action, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d_model),
        )
        # inverse dynamics reads the PAIR, so the difference is available to it directly
        self.inverse = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.GELU()
        )
        self.inv_pos = nn.Linear(hidden, length)
        self.inv_aa = nn.Linear(hidden, n_aa)
        # reward is read off the latent, so the warm-up has something to shape it with
        self.reward = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.sigreg = SIGReg(seed=seed)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.adapter(z)

    def step(self, h: torch.Tensor, pos: torch.Tensor, aa: torch.Tensor) -> torch.Tensor:
        """One deterministic transition in latent space. Residual: the action EDITS."""
        a = torch.cat([self.pos_emb(pos), self.aa_emb(aa)], dim=-1)
        return h + self.dynamics(torch.cat([h, a], dim=-1))

    def inverse_logits(self, h_t: torch.Tensor, h_next: torch.Tensor):
        g = self.inverse(torch.cat([h_t, h_next - h_t], dim=-1))
        return self.inv_pos(g), self.inv_aa(g)

    def predict_reward(self, h: torch.Tensor) -> torch.Tensor:
        return self.reward(h).squeeze(-1)


def dynamics_weight(epoch: int, warmup: int, ramp: int) -> float:
    """Warm-up schedule: reward + inverse only, then ramp the dynamics loss in.

    The proposal's limitations name the moving-target interaction: the dynamics loss is
    computed against a target produced by the same adapter it is training, so letting it
    dominate from step zero lets the model satisfy it by making the target boring. The
    ramp gives the inverse head and the reward head a chance to put structure into the
    latent first.
    """
    if epoch < warmup:
        return 0.0
    if epoch >= warmup + ramp:
        return 1.0
    return (epoch - warmup + 1) / (ramp + 1)


class ResidueWorldModelB(nn.Module):
    """World Model B on the PER-RESIDUE state Step 3.1 registers.

    `WorldModelB` above operates on the mean-pooled state. That was a pipeline validation
    and is kept because its results are published in `results/world_model_B.md`, but
    pooling is the wrong state for this model for a measured reason: the frozen encoder's
    pooled states have effective rank 6-37 of 1280, against 26-70 for its per-residue rows
    (`scripts/state_rank_ceiling.py`). Pooling discards about two thirds of the directions
    the dynamics could use.

    The differences from the pooled model are all forced by the state's shape:

    adapter      applied per row, so a state is `(L, d_model)` rather than `(d_model,)`.
    dynamics     an action is one substitution at one position, so the per-row update is
                 conditioned on that row's OFFSET from the mutated site as well as on the
                 action itself. The offset is bucketed and clamped rather than exact: the
                 propagation measurement found a substitution's effect is a diffuse
                 chain-wide shift (on 735-residue CAPSD a radius-16 window captures 37.6%
                 of it), so distant rows do move, and they move similarly.
    inverse      position is read off per row -- one scalar logit per residue -- which is
                 what the per-residue state makes available and the pooled state cannot.
                 The amino acid is a global property of the edit, so it is read from the
                 mean-pooled difference.
    SIGReg       applied to ROWS, treating `(B * L, d_model)` as the sample. This matches
                 how `rank_preserved_vs_input` is measured for the gate, so the
                 regulariser and the diagnostic are reading the same object.
    """

    def __init__(self, d_in: int = 1280, d_model: int = 256, length: int = 1024,
                 n_aa: int = 20, d_action: int = 64, hidden: int = 512,
                 max_offset: int = 64, seed: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.length = length
        self.max_offset = max_offset
        self.adapter = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        self.pos_emb = nn.Embedding(length, d_action)
        self.aa_emb = nn.Embedding(n_aa, d_action)
        # 2 * max_offset + 1 signed buckets, plus one shared bucket for "further than
        # max_offset in either direction"
        self.off_emb = nn.Embedding(2 * max_offset + 2, d_action)
        self.dynamics = nn.Sequential(
            nn.Linear(d_model + 3 * d_action, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d_model),
        )
        self.inverse = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.inv_aa = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, n_aa)
        )
        self.reward = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.sigreg = SIGReg(seed=seed)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """`(B, L, d_in)` -> `(B, L, d_model)`."""
        if z.ndim != 3:
            raise ValueError(f"expected (B, L, d_in), got {tuple(z.shape)}")
        return self.adapter(z)

    def _offset_bucket(self, length: int, pos: torch.Tensor) -> torch.Tensor:
        """Signed, clamped row offset from the mutated site -> embedding index."""
        rows = torch.arange(length, device=pos.device).view(1, -1)
        off = rows - pos.view(-1, 1)
        far = off.abs() > self.max_offset
        idx = off.clamp(-self.max_offset, self.max_offset) + self.max_offset
        return idx.masked_fill(far, 2 * self.max_offset + 1)

    def step(self, h: torch.Tensor, pos: torch.Tensor, aa: torch.Tensor) -> torch.Tensor:
        """One deterministic transition. Residual, per row: the action EDITS the state."""
        b, length, _ = h.shape
        a = torch.cat([self.pos_emb(pos), self.aa_emb(aa)], dim=-1)  # (B, 2*d_action)
        a = a.unsqueeze(1).expand(b, length, a.shape[-1])
        o = self.off_emb(self._offset_bucket(length, pos))           # (B, L, d_action)
        return h + self.dynamics(torch.cat([h, a, o], dim=-1))

    def inverse_logits(self, h_t: torch.Tensor, h_next: torch.Tensor):
        pair = torch.cat([h_t, h_next - h_t], dim=-1)        # (B, L, 2*d_model)
        pos_logits = self.inverse(pair).squeeze(-1)          # (B, L) -- one per residue
        aa_logits = self.inv_aa(pair.mean(1))                # (B, n_aa)
        return pos_logits, aa_logits

    def predict_reward(self, h: torch.Tensor) -> torch.Tensor:
        return self.reward(h.mean(1)).squeeze(-1)

    def rows(self, h: torch.Tensor) -> torch.Tensor:
        """`(B, L, d)` -> `(B * L, d)`, the sample SIGReg and the rank gate both use."""
        return h.reshape(-1, h.shape[-1])
