"""Step 3.2 — collapse diagnostics. The gate for Phase 3.

ProteinJEPA found that latent prediction without a masked-language-modelling anchor
collapses in nearly every experiment on protein data, with gains that did not replicate
across seeds. World Model B removes the EMA target deliberately, on the LeWorldModel
design where SIGReg replaces teacher-student machinery, so the one thing that must not be
assumed is that it worked.

The registered threshold is `latent_effective_rank_ratio >= 0.5`.

**Why effective rank and not the loss.** A collapsed model has a *low* prediction loss --
predicting a constant is easy when the target is nearly constant. Training loss moving in
the right direction is consistent with total collapse, which is exactly why the gate is
written on the representation rather than on the objective.
"""

from __future__ import annotations

import numpy as np
import torch


def effective_rank(h: np.ndarray) -> float:
    """exp(entropy of the normalised eigenspectrum) -- the participation ratio of
    variance across directions.

    Hard rank counts non-zero eigenvalues and so reports `d` for a latent whose variance
    sits almost entirely in two directions. Effective rank reports ~2, which is the
    number a collapse diagnostic has to produce to be worth logging.
    """
    x = h - h.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    ev = s**2
    tot = ev.sum()
    if tot <= 0:
        return 0.0
    p = ev / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def diagnostics(model, z: torch.Tensor, src, dst, pos, aa, limit: int = 4096) -> dict:
    """Everything Step 3.2 asks to be logged every run."""
    model.eval()
    n = min(limit, len(src))
    h = model.encode(z)
    h_t, h_next = h[src[:n]], h[dst[:n]]
    pred = model.step(h_t, pos[:n], aa[:n])

    hn = h.detach().cpu().numpy()
    er = effective_rank(hn)
    ratio = er / hn.shape[1]

    lp, la = model.inverse_logits(h_t, h_next)
    pos_acc = (lp.argmax(-1) == pos[:n]).double().mean().item()
    aa_acc = (la.argmax(-1) == aa[:n]).double().mean().item()

    # a one-step predictor that ignores the action entirely is the control that matters:
    # if predicting "no change" scores as well, the dynamics head has learned nothing
    err = (pred - h_next).pow(2).sum(-1)
    null = (h_t - h_next).pow(2).sum(-1)
    scale = h_next.var(0).sum().clamp(min=1e-12)

    model.train()
    return {
        "latent_effective_rank": er,
        "latent_effective_rank_ratio": ratio,
        "latent_dim": float(hn.shape[1]),
        "sigreg_stat": float(model.sigreg(h[:limit]).item()),
        "inverse_position_top1": pos_acc,
        "inverse_aa_top1": aa_acc,
        "one_step_mse": float(err.mean().item()),
        "identity_baseline_mse": float(null.mean().item()),
        # < 1 means the dynamics head beats "nothing happened"; >= 1 means it does not
        "one_step_vs_identity": float((err.mean() / null.mean().clamp(min=1e-12)).item()),
        "one_step_normalised": float((err.mean() / scale).item()),
        "latent_std": float(h.std().item()),
    }
